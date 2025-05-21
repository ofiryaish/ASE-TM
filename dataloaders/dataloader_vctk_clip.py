import os
import json
import random

import torch
import torch.utils.data
import librosa

from models.stfts import mag_phase_stft
from models.pcs400 import cal_pcs


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


class VCTKDemandClipDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and clip audio files.

    Args:
        clean_json (str): Json containing clean audio files.
        simulator (PreGeneratedSimulator): Simulator for generating clip audio.
        const_clip_value (float, optional): Constant clip value. Defaults to None. If not None, randomly
            select a clip value from range [min_clip_value, max_clip_value] for each audio.
        min_clip_value (float, optional): Minimum clip value. Defaults to None.
        max_clip_value (float, optional): Maximum clip value. Defaults to None.
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
        const_clip_value=None,
        min_clip_value=None,
        max_clip_value=None,
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
        normalize=True
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

        self.const_clip_value = const_clip_value
        self.min_clip_value = min_clip_value
        self.max_clip_value = max_clip_value

        if self.const_clip_value is None and self.min_clip_value is None and self.max_clip_value is None:
            raise ValueError("const_clip_value must be provided or min_clip_value and or max_clip_value.")

    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """
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

        # Create noisy audio by clipping
        clip_value = self.const_clip_value if self.const_clip_value else random.uniform(
            self.min_clip_value, self.max_clip_value)
        noisy_audio = torch.clamp(clean_audio, min=-clip_value, max=clip_value)

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
