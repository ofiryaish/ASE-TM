import os
import argparse
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.pcs400 import cal_pcs
import soundfile as sf

from models.simulate_paths import (
    instance_simulator, process_signals_through_primary_path,
    process_signals_through_secondary_path, randomize_reverberation_time)
from utils.util import load_config

from compute_PESQ import pesq_score


def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    # Create simulator
    if cfg['rir_cfg']['type'] == "RIR":
        reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=sampling_rate,
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, hp_filter=cfg['rir_cfg']['hp_filter'], version=cfg['rir_cfg']['version'])
    elif cfg['rir_cfg']['type'] == "PyRoom":
        reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=sampling_rate,
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, version=cfg['rir_cfg']['version'])
    else:
        raise ValueError("Unknown simulator type")

    if args.evaluate_on_testset:
        audios_r, audios_g = [], []

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ):
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir(args.input_folder)):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join(args.input_folder, fname), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            # max_original = noisy_wav.abs().max()
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor=1)
            # Simulator part
            # TODO: without t60 I could process the signals through the primary path beforehead
            # Process signals through primary path
            t60 = randomize_reverberation_time(reverberation_times)
            audio_g = process_signals_through_secondary_path(audio_g, simulator, t60, sef_factor="random")
            noisy_wav = process_signals_through_primary_path(noisy_wav, simulator, t60)
            audio_g = audio_g + noisy_wav[:, :audio_g.shape[1]]  # TODO: This is not an ideal solution
            # End simulator part
            audio_g = audio_g / norm_factor
            # print(audio_g.abs().max())
            # audio_g = (audio_g / audio_g.abs().max()) * max_original

            output_file = os.path.join(args.output_folder, fname)

            if args.post_processing_PCS:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
            else:
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')

            if args.evaluate_on_testset:
                audios_r.append(audio_g.squeeze().cpu().numpy())
                clean_wav, _ = librosa.load(os.path.join(args.clean_input_folder, fname), sr=sampling_rate)
                clean_wav = torch.FloatTensor(clean_wav).to(device).unsqueeze(0)
                clean_wav = process_signals_through_primary_path(clean_wav, simulator, t60)
                audios_g.append(clean_wav.cpu().numpy())

    if args.evaluate_on_testset:
        val_pesq_score = pesq_score(audios_r, audios_g, cfg).item()
        print("Evaluate on testset - PESQ:", val_pesq_score)


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='data/VCTK_dataset/noisy_testset_wav')  # EnhancedSamples/ref_noisy/
    parser.add_argument('--output_folder', default='exp/SEMamba_active_v9/checkpoint_g_00256000_results')
    parser.add_argument('--clean_input_folder', default='data/VCTK_dataset/clean_testset_wav')
    parser.add_argument('--config', default='exp/SEMamba_active_v9/config.yaml')
    parser.add_argument('--checkpoint_file', default='exp/SEMamba_active_v9/g_00256000.pth')
    parser.add_argument('--post_processing_PCS', default=False, action='store_true')
    parser.add_argument('--evaluate_on_testset', default=False, action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    inference(args, device)


if __name__ == '__main__':
    main()
