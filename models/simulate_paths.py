import os
import random
import numpy as np
import torch
import torchaudio

import rir_generator
import pyroomacoustics as pra

from models.reverberate import reverberate


PRIMARY_PATH = 1
SECONDARY_PATH = 2


# TODO: can we delete _simulate and __simulate_v2 since v=3?
def _simulate(signal_batch, rir, device, padding="same"):
    signal_batch = signal_batch.to(device).unsqueeze(1)
    rir = rir.to(device).unsqueeze(1)
    processed_signals = torch.nn.functional.conv1d(signal_batch, rir, padding=padding)
    processed_signals = processed_signals.squeeze(1)
    return processed_signals


def _simulate_v2(signal_batch, rir, device, padding="same"):
    signal_batch = signal_batch.to(device).unsqueeze(1)
    rir = rir.to(device).unsqueeze(1)
    # Apply the filter in the forward direction
    processed_signals = torch.nn.functional.conv1d(signal_batch, rir, padding=padding)

    # Reverse the filtered signal
    processed_signals = torch.flip(processed_signals, [2])

    # Apply the filter again in the forward direction
    processed_signals = torch.nn.functional.conv1d(processed_signals, rir, padding=padding)

    # Reverse the signal back to its original order
    processed_signals = torch.flip(processed_signals, [2])

    processed_signals = processed_signals.squeeze(1)
    return processed_signals


class RIRGenSimulator:
    def __init__(
            self, sr, reverbation_times, device,
            rir_samples=512, hp_filter=False, c=343, v=3):

        self.sr = sr
        self.device = device
        self.room_dim = [3, 4, 2]
        self.ref_mic = [1.5, 1, 1]
        self.ls_source = [1.5, 2.5, 1]
        self.error_mic = [1.5, 3, 1]
        self.reverbation_times = reverbation_times
        self.rir_length = rir_samples
        self.hp_filter = hp_filter
        self.c = c
        self.v = v
        self.rirs = self.get_rirs()

    def get_rirs(self):
        rirs = dict()
        for t60 in self.reverbation_times:
            for rir_type in [PRIMARY_PATH, SECONDARY_PATH]:
                if rir_type == PRIMARY_PATH:
                    pos_src = self.ref_mic
                    pos_rcv = self.error_mic
                elif rir_type == SECONDARY_PATH:
                    pos_src = self.ls_source
                    pos_rcv = self.error_mic
                rir = rir_generator.generate(  # consider hp_filter = False
                    c=self.c,
                    fs=self.sr,
                    s=pos_src,
                    r=[pos_rcv],
                    L=self.room_dim,
                    reverberation_time=t60,
                    nsample=self.rir_length,
                    hp_filter=self.hp_filter
                )
                rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir)).to(self.device).view(1, 1, -1).float()
        return rirs

    def simulate(self, signal_batch, t60, signal_type, padding="same"):
        rir = self.rirs[(t60, signal_type)]
        if self.v == 1:
            return _simulate(signal_batch, rir, self.device, padding)
        elif self.v == 2:
            return _simulate_v2(signal_batch, rir, self.device, padding)
        elif self.v == 3:
            return torchaudio.transforms.FFTConvolve(mode="same")(signal_batch, rir.squeeze(0))
        elif self.v == 4:
            return torchaudio.transforms.Convolve(mode="same")(signal_batch, rir)
        elif self.v == 5:
            return reverberate(signal_batch, rir.squeeze(0))
        else:
            raise ValueError("Does not support this version.")


class PyRoomSimulator:
    def __init__(self, sr, reverbation_times, device, rir_samples=512, v=3):
        self.sr = sr
        self.device = device
        self.room_dim = np.array([3, 4, 2])
        self.ref_mic = np.array([1.5, 1, 1])
        self.ls_source = np.array([1.5, 2.5, 1])
        self.error_mic = np.array([1.5, 3, 1])
        self.reverbation_times = reverbation_times
        self.rir_length = rir_samples
        self.rirs = self.get_rirs()
        self.v = v

    def get_rirs(self):
        rirs = dict()

        for t60 in self.reverbation_times:
            e_absorption, max_order = pra.inverse_sabine(t60, self.room_dim)
            for rir_type in [PRIMARY_PATH, SECONDARY_PATH]:
                room = pra.ShoeBox(
                    self.room_dim, fs=self.sr, materials=pra.Material(e_absorption), max_order=max_order)

                if rir_type == PRIMARY_PATH:
                    pos_src = self.ref_mic
                    pos_rcv = self.error_mic
                elif rir_type == SECONDARY_PATH:
                    pos_src = self.ls_source
                    pos_rcv = self.error_mic
                room.add_source(pos_src)
                mic = pra.MicrophoneArray(pos_rcv.reshape((-1, 1)), self.sr)
                room.add_microphone_array(mic)

                room.compute_rir()
                rir = room.rir[0][0]
                # make RIR adjustments to ISM model
                # (by pyroomacooustics maintainer https://github.com/DavidDiazGuerra/gpuRIR/issues/61)
                # rir_ism = rir[40:40+self.rir_length] * (1/(torch.pi * 4))
                # rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir_ism)).to(self.device).view(1, -1).float()
                rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir)).to(self.device).view(1, -1).float()
        return rirs

    def simulate(self, signal_batch, t60, signal_type, padding="same"):
        rir = self.rirs[(t60, signal_type)]
        if self.v == 1:
            return _simulate(signal_batch, rir, self.device, padding)
        elif self.v == 2:
            return _simulate_v2(signal_batch, rir, self.device, padding)
        elif self.v == 3:
            return torchaudio.transforms.FFTConvolve(mode="same")(signal_batch, rir)
        elif self.v == 4:
            return torchaudio.transforms.Convolve(mode="same")(signal_batch, rir)
        elif self.v == 5:
            return reverberate(signal_batch, rir.squeeze(0))
        else:
            raise ValueError("Does not support this version.")


class PreGeneratedSimulator:
    def __init__(
            self, sr, device, rir_files_path, rir_samples=-1, v=3):
        self.sr = sr
        self.device = device
        self.rir_length = rir_samples
        self.v = v
        self.base_path = rir_files_path
        self.rir_files = self.get_all_wav_files()

    def get_all_wav_files(self):
        # Recursively find all .wav files under self.base_path
        return [os.path.join(root, file)
                for root, _, files in os.walk(self.base_path)
                for file in files if file.endswith('.wav')]

    def load_rir(self, rir_file):
        rir, sr = torchaudio.load(rir_file)

        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            rir = resampler(rir)

        if self.rir_length > 0:
            rir = rir[:, :self.rir_length]

        rir = rir.to(self.device).view(1, 1, -1).float()
        return rir

    def sample_rir(self):
        rir_file = random.choice(self.rir_files)
        # print(f"Loading RIR from {rir_file}")
        rir = self.load_rir(rir_file)
        return rir

    def simulate(self, signal_batch, padding="same", rir_file=None):
        if rir_file is None:
            rir = self.sample_rir()
        else:
            rir = self.load_rir(rir_file)
        if self.v == 1:
            return _simulate(signal_batch, rir, self.device, padding)
        elif self.v == 2:
            return _simulate_v2(signal_batch, rir, self.device, padding)
        elif self.v == 3:
            return torchaudio.transforms.FFTConvolve(mode="same")(signal_batch, rir.squeeze(0))
        elif self.v == 4:
            return torchaudio.transforms.Convolve(mode="same")(signal_batch, rir)
        elif self.v == 5:
            return reverberate(signal_batch, rir.squeeze(0))
        else:
            raise ValueError("Does not support this version.")


def get_random_factor():
    factors = [0.1**0.5, 1**0.5, 10**0.5, "inf"]
    return factors[random.randint(0, len(factors) - 1)]


def sef(tensor, factor="inf"):
    """
    Non-linearity function for the secondary path.
    "inf" means inference.
    """
    # u = x/(sqrt(2)*factor), du = 1/(sqrt(2)*factor) dx => dx = sqrt(2)*factor * du
    # integral(e^(-(x^2)/(2*factor^2)), 0, tensor) dx =
    # integral(e^(-(x/(sqrt(2)*factor))^2), 0, tensor) dx =
    # integral(e^(-u^2) * sqrt(2)*factor, 0, tensor-u) du =
    # sqrt(2)*factor * integral(e^(-u^2), 0, tensor-u) du =
    # sqrt(2)*factor * ((sqrt(pi)/2) * erf(tensor-u) - (sqrt(pi)/2) * erf(0))  =
    # sqrt(2)*factor * sqrt(pi)/2 * (erf(tensor-u) - erf(0))  =
    # factor * sqrt(pi/2) * (erf(tensor/(sqrt(2)*factor)) - erf(0))) [because u = x/(sqrt(2)*factor)]
    sqrt_2 = 2 ** 0.5
    sqrt_pi_2 = (torch.pi/2) ** 0.5
    if factor == "random":
        factor = get_random_factor()

    if factor == "inf":
        return tensor

    return factor * sqrt_pi_2 * (torch.erf(tensor/(sqrt_2*factor)))


def randomize_reverberation_time(reverberation_times):
    """
    Randomly selects and returns a reverberation time from the provided list.

    Parameters:
    reverberation_times (list): A list of reverberation times to choose from.

    Returns:
    float: A randomly selected reverberation time from the list.
    """
    t60 = reverberation_times[random.randint(0, len(reverberation_times) - 1)]

    return t60


def process_signals_through_primary_path(signals: torch.Tensor,
                                         simulator: RIRGenSimulator | PyRoomSimulator, t60: float):
    """
    Processes signals through the primary path using a simulator and a reverberation time.
    Args:
        signals (torch.Tensor): The input signals to be processed through the primary path.
        simulator (RIRGenSimulator or PyRoomSimulator): The simulator object used to simulate the signal paths.
        t60 (float): The reverberation time to be used in the simulation.
    Returns:
        torch.Tensor: The processed signals through the primary path.
    """
    return simulator.simulate(signals, t60, signal_type=PRIMARY_PATH)


def process_signals_through_secondary_path(signals: torch.Tensor,
                                           simulator: RIRGenSimulator | PyRoomSimulator,
                                           t60: float, sef_factor="random"):
    """
    Processes signals through the secondary path using a simulator and a reverberation time.
    Args:
        signals (torch.Tensor): The input signals to be processed through the secondary path.
        simulator (RIRGenSimulator): The simulator object used to simulate the signal paths.
        t60 (float): The reverberation time to be used in the simulation.
        sef_factor (str, optional): The factor to be used in the sef function for non-linearity. Defaults to "random".
    Returns:
        torch.Tensor: The processed signals through the secondary path.
    """
    signals = sef(signals, sef_factor)  # None-linearity function
    return simulator.simulate(signals.float(), t60, signal_type=SECONDARY_PATH).type(signals.type())


def instance_simulator(simulator_type, sr=16000, reverberation_times=None,
                       rir_samples=512, device="cuda",  hp_filter=True, version=3):
    """
    Initializes an instance of RIRGenSimulator or PyRoomSimulator with the given parameters.

    Parameters:
    simulator_type (str): Simulator type name. Currently support "RIR" or "PyRoom".
    reverberation_times (list, optional): List of reverberation times to simulate.
        Defaults to [0.15, 0.175, 0.2, 0.225, 0.25].
    sr (int, optional): Sampling rate for the simulation. Defaults to 16000.
    rir_samples (int, optional): Number of samples for the RIR. Defaults to 512.
    device (str, optional): Device to run the simulation on, e.g., "cuda" for GPU. Defaults to "cuda".
    hp_filter (bool, optional): Whether to apply a high-pass filter. Defaults to True. Not used for PyRoomSimulator.
    version (int, optional): Version of the RIR generator simulate function to use.
        Defaults to 3. Not used for PyRoomSimulator.


    Returns:
    tuple: A tuple containing the list of reverberation times
        and an instance of RIRGenSimulator initialized with the given parameters.
    """
    if reverberation_times is None:
        reverberation_times = [0.15, 0.175, 0.2, 0.225, 0.25]
    if simulator_type == "RIR":
        return reverberation_times, RIRGenSimulator(sr, reverberation_times, device, hp_filter=hp_filter, v=version,
                                                    rir_samples=rir_samples)
    elif simulator_type == "PyRoom":
        return reverberation_times, PyRoomSimulator(sr, reverberation_times, device, rir_samples=rir_samples, v=version)
    else:
        raise ValueError("Invalid simulator type. Please choose either 'RIR' or 'PyRoom'.")
