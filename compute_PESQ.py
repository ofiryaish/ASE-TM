import argparse
import os
import librosa
import numpy as np
from pesq import pesq
import torch

from models.simulate_paths import instance_simulator, process_signals_through_primary_path
from utils.util import load_config
from joblib import Parallel, delayed


def pesq_score(utts_r, utts_g, cfg):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score for pairs of reference and generated utterances.

    Args:
        utts_r (list of torch.Tensor): List of reference utterances.
        utts_g (list of torch.Tensor): List of generated utterances.
        h (object): Configuration object containing parameters like sampling_rate.

    Returns:
        float: Mean PESQ score across all pairs of utterances.
    """
    def eval_pesq(clean_utt, esti_utt, sr):
        """
        Evaluate PESQ score for a single pair of clean and estimated utterances.

        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.

        Returns:
            float: PESQ score or -1 in case of an error.
        """
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt)
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1
        return pesq_score

    # Parallel processing of PESQ score computation
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze(),
        utts_g[i].squeeze(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))

    # Calculate mean PESQ score
    pesq_score = np.mean(pesq_scores)
    return pesq_score


def run_pesq(args, cfg):
    # Create simulator
    device = torch.device('cuda')
    if cfg['rir_cfg']['type'] == "RIR":
        reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=cfg['stft_cfg']['sampling_rate'],
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, hp_filter=cfg['rir_cfg']['hp_filter'], version=cfg['rir_cfg']['version'])
    elif cfg['rir_cfg']['type'] == "PyRoom":
        reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=cfg['stft_cfg']['sampling_rate'],
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, version=cfg['rir_cfg']['version'])
    else:
        raise ValueError("Unknown simulator type")

    audios_r = []
    audios_g = []
    t60 = 0.15
    for i, fname in enumerate(os.listdir(args.noisy_input_folder)):
        noisy_wav, _ = librosa.load(os.path.join(args.noisy_input_folder, fname), sr=cfg['stft_cfg']['sampling_rate'])
        audios_r.append(noisy_wav)
    for i, fname in enumerate(os.listdir(args.clean_input_folder)):
        clean_wav, _ = librosa.load(os.path.join(args.clean_input_folder, fname), sr=cfg['stft_cfg']['sampling_rate'])
        clean_wav = torch.FloatTensor(clean_wav).to(device).unsqueeze(0)
        clean_wav = process_signals_through_primary_path(clean_wav, simulator, t60)
        audios_g.append(clean_wav.cpu().numpy())

    val_pesq_score = pesq_score(audios_r, audios_g, cfg).item()

    return val_pesq_score


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_input_folder', default='exp/SEMamba_active_v9/checkpoint_g_00256000_results_t60_015')
    parser.add_argument('--clean_input_folder', default='data/VCTK_dataset/clean_testset_wav')
    parser.add_argument('--config', default='exp/SEMamba_active_v9/config.yaml')

    args = parser.parse_args()
    cfg = load_config(args.config)

    print(run_pesq(args, cfg))


if __name__ == '__main__':
    main()

# SEMamba_active_v5/checkpoint_g_00127000_results: 2.742228848696912
# SEMamba_active_v6/checkpoint_g_00051000_results_t60_025: 2.609804112645029
