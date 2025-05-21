import os
import json
import random

import torch
import torch.utils.data
import librosa

from models.stfts import mag_phase_stft
from models.pcs400 import cal_pcs
from models.simulate_paths import PreGeneratedSimulator


def list_files_in_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files


def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


class VCTKDemandReverbDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and reverb audio files.

    Args:
        clean_json (str): Json containing clean audio files.
        simulator (PreGeneratedSimulator): Simulator for generating reverb audio.
        sampling_rate (int, optional): Sampling rate of the audio files. Defaults to 16000.
        segment_size (int, optional): Size of the audio segments. Defaults to 32000.
        n_fft (int, optional): FFT size. Defaults to 400.
        hop_size (int, optional): Hop size. Defaults to 100.
        win_size (int, optional): Window size. Defaults to 400.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        split (bool, optional): Whether to split the audio into segments. Defaults to True.
        n_cache_reuse (int, optional): Number of times to reuse cached audio. Defaults to 1.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        pcs (bool, optional): Use PCS in training period. Defaults to False.
        normalize (bool, optional): Normalize the audio. Defaults to True.
    """
    def __init__(
        self,
        clean_json,
        simulator: PreGeneratedSimulator,
        sampling_rate=16000,
        segment_size=32000,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=1.0,
        split=True,
        n_cache_reuse=1,
        shuffle=True,
        pcs=False,
        normalize=True,
        const_rirs_files=None
    ):
        self.clean_wavs_path = load_json_file(clean_json)

        random.seed(1234)
        if shuffle:
            random.shuffle(self.clean_wavs_path)

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.cached_clean_wav = None
        self._cache_ref_count = 0
        self.pcs = pcs
        self.normalize = normalize

        self.simulator = simulator

        self.const_rirs_files = const_rirs_files
        self.rir_index = -1

    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """
        if self.const_rirs_files is not None:
            self.rir_index += 1
            if self.rir_index >= len(self.const_rirs_files):
                raise IndexError("Index out of range for constant RIR files.")
            rir_file = self.const_rirs_files[self.rir_index]

        if self._cache_ref_count == 0:
            clean_path = self.clean_wavs_path[index]
            clean_audio, _ = librosa.load(clean_path, sr=self.sampling_rate)
            if self.pcs:
                clean_audio = cal_pcs(clean_audio)
            self.cached_clean_wav = clean_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            self._cache_ref_count -= 1

        clean_audio = torch.FloatTensor(clean_audio).unsqueeze(0)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(
                    clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')

        noisy_audio = self.simulator.simulate(clean_audio, rir_file=rir_file)
        if self.normalize:
            norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
            clean_audio = (clean_audio * norm_factor)
            noisy_audio = (noisy_audio * norm_factor)
        else:
            norm_factor = 1.0

        assert clean_audio.size(1) == noisy_audio.size(1)

        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(),
                noisy_audio.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze(), norm_factor)

    def __len__(self):
        return len(self.clean_wavs_path)
