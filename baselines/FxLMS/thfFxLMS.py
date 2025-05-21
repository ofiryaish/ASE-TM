import argparse
import os
import torch

import soundfile as sf

from models.simulate_paths import (
    instance_simulator, process_signals_through_primary_path,
    process_signals_through_secondary_path, sef, SECONDARY_PATH)
from utils.util import load_config
from utils.compute_metrics import nmse


from dataloaders.train_dataset_utilis import create_dataloader, create_dataset

from pesq import pesq


class FxNLMS():
    def __init__(self, w_len, mu, device='cuda'):
        self.grads = 0
        self.w = torch.zeros(1, w_len, dtype=torch.float).to(device)
        self.x_buf = torch.zeros(1, w_len, dtype=torch.float).to(device)
        self.st_buf = torch.zeros(1, w_len, dtype=torch.float).to(device)
        self.mu = mu

    def predict(self, x, st):
        self.x_buf = torch.roll(self.x_buf, 1, 1)
        self.x_buf[0, 0] = x
        yt = self.w @ self.x_buf.t()

        self.st_buf = torch.roll(self.st_buf, 1, 1)
        self.st_buf[0, 0] = st
        power = self.st_buf @ self.st_buf.t()  # FxNLMS different from FxLMS
        return yt, power

    def step(self, loss):
        loss = torch.clamp(loss, -1e-03, 1e-03)
        grad = self.mu * loss * self.st_buf.flip(1)
        self.w += grad


def get_score(noisy_signal, anti_signal, clean_signal, simulator, t60):
    noisy_pt = process_signals_through_primary_path(
        noisy_signal, simulator, t60).squeeze()
    anti_st = process_signals_through_secondary_path(
        anti_signal, simulator, t60, sef_factor="inf").squeeze()
    est_clean_speech = noisy_pt + anti_st
    clean_pt = process_signals_through_primary_path(
        clean_signal, simulator, t60).squeeze()
    return nmse(clean_pt, est_clean_speech).item()


def execute_algo(clean_audio, noisy_audio, mu, simulator,
                 gama="inf", t60=0.25, thf=False, rir_samples=512, device="cuda",
                 sampling_rate=16000):
    fxlms = FxNLMS(w_len=rir_samples, mu=mu)
    # Process audio through paths
    padded_noisy_audio = torch.nn.functional.pad(noisy_audio, (rir_samples//2, 0), mode='constant', value=0)
    pt_signal = process_signals_through_primary_path(
        padded_noisy_audio, simulator, t60).squeeze()
    padded_noisy_audio = torch.tanh(padded_noisy_audio) if thf else padded_noisy_audio
    st_signal = process_signals_through_secondary_path(
        padded_noisy_audio, simulator, t60, sef_factor="inf").squeeze()

    padded_clean_audio = torch.nn.functional.pad(clean_audio, (rir_samples//2, 0), mode='constant', value=0)
    pt_clean_signal = process_signals_through_primary_path(
        padded_clean_audio, simulator, t60).squeeze()

    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)

    st = simulator.rirs[(t60, SECONDARY_PATH)].squeeze(0).to(device)
    ys = []
    len_data = pt_signal.shape[0]
    for i in range(len_data - rir_samples//2):
        # Feedfoward
        xin = st_signal[i]
        dis = pt_signal[i]

        y, power = fxlms.predict(noisy_audio[0, i], xin)

        y_buf = torch.roll(y_buf, -1, 0)
        y_buf[0, -1] = y
        y_buf_sef = sef(y_buf, factor=gama)
        sy = st @ y_buf_sef.t().flip(0)

        # e = dis-sy
        loss = dis + sy - pt_clean_signal[i]
        # loss = (e**2)
        # Progress shown

        fxlms.step(loss)
        ys.append(y.item())

        if i > 0 and i % 1000 == 0:
            noisy_audio_ = noisy_audio[0, :i+1 - rir_samples//2].unsqueeze(0)
            anti_signal_ = torch.tensor(ys)[rir_samples//2:].unsqueeze(0).to(device)
            clean_audio_ = clean_audio[0, :i+1 - rir_samples//2].unsqueeze(0)
            score = get_score(noisy_audio_, anti_signal_, clean_audio_,
                              simulator=simulator, t60=t60)
            print(f"Step {i} - Loss: {score:.5f}", flush=True)
        # Progress shown
    noisy_pt = process_signals_through_primary_path(
        noisy_audio_, simulator, t60).squeeze()
    anti_st = process_signals_through_secondary_path(
        anti_signal_, simulator, t60, sef_factor="inf").squeeze()
    est_clean_speech = noisy_pt + anti_st
    pt_clean_speech = process_signals_through_primary_path(
        clean_audio_, simulator, t60).squeeze()

    pt_clean_speech = pt_clean_speech.cpu().numpy()
    est_clean_speech = est_clean_speech.cpu().numpy()
    print(pesq(sampling_rate, pt_clean_speech, est_clean_speech, 'wb'))

    return pt_clean_speech, est_clean_speech


def main():
    t60 = 0.25
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='exp/thfFxLMS/t025_new_inference_inf')
    parser.add_argument('--config', default='recipes/FxLMS/thfFxLMS.yaml')
    args = parser.parse_args()

    device = "cuda"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder + "_clean", exist_ok=True)

    cfg = load_config(args.config)
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

    validset = create_dataset(cfg, train=False, split=False, test=True, normalize=False)
    validation_loader = create_dataloader(validset, cfg, train=False)

    for j, batch in enumerate(validation_loader):
        clean_audio, noisy_audio, _noisy_mag, _noisy_pha, _norm_factor = \
                batch  # [B, 1, F, T], F = nfft // 2+ 1, T = nframes

        noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
        clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))

        pt_clean_speech, est_clean_speech = execute_algo(
            clean_audio, noisy_audio, mu=0.1,
            simulator=simulator, gama="inf",
            t60=t60, thf=cfg['model_cfg']['thf'],
            rir_samples=cfg['rir_cfg']['rir_samples'],
            sampling_rate=sampling_rate
        )

        if output_folder is not None:
            fname = str(j) + ".wav"
            generated_output_file = os.path.join(output_folder, fname)
            sf.write(generated_output_file, est_clean_speech, sampling_rate, 'PCM_16')
            clean_ouput_file = os.path.join(output_folder + "_clean", fname)
            sf.write(
                clean_ouput_file, pt_clean_speech, sampling_rate, 'PCM_16')


if __name__ == "__main__":
    main()
