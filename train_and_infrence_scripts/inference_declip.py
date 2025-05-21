
import os
import argparse
import torch
from models.stfts import mag_phase_istft
from models.generator import (SEMamba, SEAtMamba, HyperSEMamba, SEMambaAt, SEMambaReAt,
                              SEMambaCo2dReAt, SEMambaCoDe2dReAt, SEHyperMambaCoDe2dReAt)
import soundfile as sf

from models.simulate_paths import (
    instance_simulator, process_signals_through_primary_path,
    process_signals_through_secondary_path)
from utils.util import load_config


from dataloaders.train_dataset_clip_utilis import create_dataloader, create_dataset


from baselines.ARN.main import predict
from baselines.ARN.model import SHARNN
from baselines.helpers import load_model


from baselines.DeepANC.model import get_model, Model


def predict_inferance(
        model, simulator, sampling_rate, device,
        n_fft, hop_size, win_size, compress_factor, cfg, t60=None, output_folder=None,
        save_noisy=True, sef_factor="inf"):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder + "_clean", exist_ok=True)
    if save_noisy:
        os.makedirs(output_folder + "_noise", exist_ok=True)
    # load files paths
    validset = create_dataset(cfg, train=False, split=False, test=True)
    validation_loader = create_dataloader(validset, cfg, train=False)

    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            clean_audio, _clean_mag, _clean_pha, _clean_com, \
                noisy_audio, noisy_mag, noisy_pha, norm_factor = batch  # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            norm_factor = torch.autograd.Variable(norm_factor.to(device, non_blocking=True))

            # Process noise through primary path
            noisy_audio = process_signals_through_primary_path(noisy_audio, simulator, t60)
            # Process the generated signal through secondary path
            if cfg['model_cfg']['model_type'] == "ARN":
                audio_g = predict(noisy_audio, model, device)
            elif cfg['model_cfg']['model_type'] == "DeepANC":
                audio_g = model.predict(noisy_audio, norm=False)[1]
            else:
                mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
                audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            audio_g = process_signals_through_secondary_path(audio_g, simulator, t60, sef_factor=sef_factor)
            audio_g = audio_g + noisy_audio[:, :audio_g.size(1)]

            audio_g = audio_g / norm_factor
            clean_audio = clean_audio / norm_factor

            if output_folder is not None:
                fname = str(j) + ".wav"
                generated_output_file = os.path.join(output_folder, fname)
                sf.write(generated_output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
                clean_ouput_file = os.path.join(output_folder + "_clean", fname)
                sf.write(
                    clean_ouput_file, clean_audio[:, :audio_g.size(1)].cpu().squeeze().numpy(), sampling_rate, 'PCM_16')

                if save_noisy:
                    noisy_audio = noisy_audio / norm_factor
                    sf.write(os.path.join(
                        output_folder + "_noise", fname), noisy_audio.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')


def inference(args, device, t60=0.25, sef_factor="inf"):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    # Create simulator
    if cfg['rir_cfg']['type'] == "RIR":
        _reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=sampling_rate,
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, hp_filter=cfg['rir_cfg']['hp_filter'], version=cfg['rir_cfg']['version'])
    elif cfg['rir_cfg']['type'] == "PyRoom":
        _reverberation_times, simulator = instance_simulator(
            simulator_type=cfg['rir_cfg']['type'], sr=sampling_rate,
            reverberation_times=cfg['rir_cfg']['reverberation_times'], rir_samples=cfg['rir_cfg']['rir_samples'],
            device=device, version=cfg['rir_cfg']['version'])
    else:
        raise ValueError("Unknown simulator type")

    print("here 1")
    if cfg['model_cfg']['model_type'] == "ARN":
        window_size = 256
        N = 512
        model = SHARNN(window_size, N, 4*N, cfg["model_cfg"]["nlayers"], cfg["training_cfg"]["dropouth"])
        model_state_dict, _, _, _ = load_model(optimizer=None, scheduler=None, cp_file_path=args.checkpoint_file)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
    elif cfg['model_cfg']['model_type'] == "DeepANC":
        model: Model = get_model(
            cfg['rir_cfg']['reverberation_times'], simulator, cfg, exp_path=args.checkpoint_file)[0]
        model.net = model.net.to(device)
        model.net.eval()
    else:
        if cfg['model_cfg']['model_type'] == "SEHyperMambaCoDe2dReAt":
            model = SEHyperMambaCoDe2dReAt(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "SEMambaCoDe2dReAt":
            # SEMambaCoDe2dReAt is the ASE-TM model
            model = SEMambaCoDe2dReAt(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "SEMambaCo2dReAt":
            model = SEMambaCo2dReAt(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "SEMambaReAt":
            model = SEMambaReAt(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "SEAtMamba":
            model = SEAtMamba(cfg).to(device)
        elif cfg['model_cfg']['model_type'] in ["SEMamba", "SEHyperMamba"]:
            model = SEMamba(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "SEMambaAt":
            model = SEMambaAt(cfg).to(device)
        elif cfg['model_cfg']['model_type'] == "HyperSEMamba":
            model = HyperSEMamba(cfg).to(device)
        else:
            raise ValueError("Unknown model type")
        print("here 2")
        state_dict = torch.load(args.checkpoint_file, map_location=device)
        model.load_state_dict(state_dict['generator'])
        model.eval()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Set the clipping value
    cfg['training_cfg']['const_clip_value'] = args.clip_value
    predict_inferance(
        model=model,
        simulator=simulator,
        sampling_rate=sampling_rate,
        device=device,
        n_fft=n_fft, hop_size=hop_size, win_size=win_size, compress_factor=compress_factor,
        cfg=cfg, t60=t60,
        output_folder=args.output_folder,
        sef_factor=sef_factor)


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', help='Path to output folder for the generated audio files')
    parser.add_argument('--config', help='Path to the config file')
    parser.add_argument('--checkpoint_file', help='Path to the checkpoint file')
    parser.add_argument('--t60', type=float, default=0.25, help='T60 time for the simulation')
    parser.add_argument('--sef_factor', type=str, default="inf", help='SEF factor for the simulation')
    parser.add_argument('--clip_value', type=float, default=0.1, help='Clipping value for the dataset generation')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    sef_factor_val = float(args.sef_factor) ** 0.5 if args.sef_factor != "inf" else args.sef_factor
    t60 = float(args.t60)
    inference(args, device, t60=t60, sef_factor=sef_factor_val)


if __name__ == '__main__':
    main()
