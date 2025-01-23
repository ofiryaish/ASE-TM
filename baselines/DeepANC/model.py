import argparse
import os

from datetime import datetime

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dataloaders.train_dataset_utilis import create_dataloader, create_dataset
from models.simulate_paths import (instance_simulator, process_signals_through_primary_path,
                                   process_signals_through_secondary_path, randomize_reverberation_time)
from baselines.DeepANC.pipeline_modules import (NetFeeder, Resynthesizer, normalize,
                                                extract_norm_params, denormalize, get_device)
from baselines.DeepANC.networks import Net
from baselines.helpers import save_model, load_model, delay_signal
from utils.util import load_config, scan_checkpoint


# Convert to string
now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
print("NOW STR: ", now_str)


class Model(object):
    def __init__(self, reverberation_times, simulator, win_len, hop_len, sr):
        self.device = get_device()
        self.sr = sr
        self.win_size = int(win_len * sr)
        self.hop_size = int(hop_len * sr)

        self.feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        self.resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)
        self.net = Net().to(self.device)

        self.reverberation_times = reverberation_times
        self.simulator = simulator

    def predict(self, noises, norm, denorm=False, frames_delay=0):
        delayed_noises = delay_signal(self.hop_size, frames_delay, noises)

        if norm:
            norm_params = extract_norm_params(noises)
            noises = normalize(noises, norm_params)

            d_norm_params = extract_norm_params(delayed_noises)
            delayed_noises = normalize(delayed_noises, d_norm_params)

        feat = self.feeder(delayed_noises)
        est = self.net(feat)
        anti_signals = self.resynthesizer(est, delayed_noises.shape).to(self.device)  # TODO shape should be of pt.shape

        if norm and denorm:
            anti_signals = denormalize(anti_signals, d_norm_params)
            noises = denormalize(noises, norm_params)

        return noises, anti_signals

    def train(
            self, tr_loader, ts_loader, optimizer, scheduler,
            start_epoch, max_epochs, norm, clip, sw, save_folder_path=None):
        criterion = torch.nn.MSELoss(reduction="mean")
        clip_norm = clip
        total_loss = 0
        total_items = 0
        best_validation_total_loss = np.inf

        # train model
        steps = 0
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_loss = 0
            epoch_items = 0
            for n_iter, batch in enumerate(tr_loader):  # tr_loader loads after resampling
                optimizer.zero_grad()
                # get the inputs and load to device
                clean_audio, noisy_audio, _noisy_mag, _noisy_pha = batch
                clean_audio = clean_audio.to(self.device)
                noisy_audio = noisy_audio.to(self.device)

                # TODO: without t60 I can process the signals through the primary path beforehead
                t60 = randomize_reverberation_time(self.reverberation_times)
                # Process signals through primary path
                clean_audio = process_signals_through_primary_path(clean_audio, self.simulator, t60)
                noisy_audio, audio_g = self.predict(noisy_audio, norm=norm)
                noisy_audio = process_signals_through_primary_path(noisy_audio,  self.simulator, t60)

                # Process the generated signal through secondary path
                audio_g = process_signals_through_secondary_path(audio_g,  self.simulator, t60, sef_factor="random")
                audio_g = audio_g + noisy_audio

                loss = criterion(clean_audio, audio_g)

                total_items += noisy_audio.shape[0]
                epoch_items += noisy_audio.shape[0]

                loss.backward()
                if clip_norm >= 0.0:
                    clip_grad_norm_(self.net.parameters(), clip_norm)
                optimizer.step()
                # calculate loss
                running_loss = loss.item()
                epoch_loss += running_loss * noisy_audio.shape[0]
                total_loss += running_loss * noisy_audio.shape[0]

                steps += 1
                if n_iter % 300 == 0:
                    print('Epoch [{}/{}], Iter [{}], epoch_loss = {:.8f}, total_loss = {:.8f}'.format(
                        epoch, max_epochs, n_iter, epoch_loss / epoch_items, total_loss / total_items), flush=True)
                    sw.add_scalar("Training/Total loss", total_loss / total_items, steps)
                    sw.add_scalar("Training/Epoch loss", epoch_loss / epoch_items, steps)

            # save model after each epoch
            save_model(self.net, optimizer, scheduler, epoch, f"{save_folder_path}/g_{steps:08d}.pth")

            # validation after each epoch
            validation_total_loss = self.test(ts_loader, criterion, norm=norm)
            print(
                'Epoch [{}/{}], Validation loss = {:.8f}'.format(
                    epoch, max_epochs, validation_total_loss), flush=True)
            sw.add_scalar("Validation/Loss", validation_total_loss, epoch)

            # save the best model based on validation loss
            if validation_total_loss < best_validation_total_loss and save_folder_path is not None:
                save_model(self.net, optimizer, scheduler, epoch, f"{save_folder_path}/best_model.pth")
                best_validation_total_loss = validation_total_loss

    def test(self, ts_loader, criterion, norm):
        self.net.eval()

        total_loss = 0
        total_items = 0
        with torch.no_grad():
            for batch in ts_loader:  # tr_loader loads after resampling
                # get the inputs and load to device
                clean_audio, noisy_audio, _noisy_mag, _noisy_pha = batch
                clean_audio = clean_audio.to(self.device)
                noisy_audio = noisy_audio.to(self.device)

                # TODO: without t60 I can process the signals through the primary path beforehead
                t60 = randomize_reverberation_time(self.reverberation_times)
                # Process signals through primary path
                clean_audio = process_signals_through_primary_path(clean_audio, self.simulator, t60)
                noisy_audio, audio_g = self.predict(noisy_audio, norm=norm)
                noisy_audio = process_signals_through_primary_path(noisy_audio,  self.simulator, t60)

                # Process the generated signal through secondary path
                audio_g = process_signals_through_secondary_path(audio_g,  self.simulator, t60, sef_factor="random")
                audio_g = audio_g + noisy_audio

                loss = criterion(clean_audio, audio_g)

                total_loss += loss.mean().item() * noisy_audio.shape[0]
                total_items += noisy_audio.shape[0]

        self.net.train()

        return total_loss / total_items


def get_model(
        reverberation_times, simulator,
        win_len, hop_len, sr, lr, exp_path):
    model = Model(reverberation_times, simulator, win_len, hop_len, sr)
    optimizer = Adam(model.net.parameters(), lr=lr, amsgrad=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_period, gamma=decay_factor)
    scheduler = None
    if os.path.isdir(exp_path):
        cp_file_path = scan_checkpoint(exp_path, 'g_')
        if cp_file_path is not None:
            model_state_dict, optimizer, scheduler, epoch = load_model(model, optimizer, scheduler, cp_file_path)
            model.net.load_state_dict(model_state_dict)
            return model, optimizer, scheduler, epoch

    return model, optimizer, scheduler, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='DeepANC_exp')
    parser.add_argument('--config', default='recipes/DeepANC/DeepANC.yaml')

    args = parser.parse_args()
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)

    # load config file
    cfg = load_config(args.config)

    device = torch.device('cuda:{:d}'.format(0))
    # Create simulator
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

    # Datasets
    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)

    # Create validset and validation_loader
    # TODO: split=True was foreced since the noise and the generated audio were not in the same length
    # (probably the generator knows to to output fixed length)
    validset = create_dataset(cfg, train=False, split=True, device=device)
    validation_loader = create_dataloader(validset, cfg, train=False)

    sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    # Define the model
    win_len = 0.020
    hop_len = 0.010
    sr = 16000
    model, optimizer, scheduler, start_epoch = get_model(
        reverberation_times, simulator, win_len, hop_len, sr, cfg['training_cfg']['learning_rate'],
        args.exp_path)

    model.train(
        train_loader, validation_loader, optimizer, scheduler, start_epoch=start_epoch,
        max_epochs=cfg['training_cfg']['training_epochs'], norm=cfg['training_cfg']['norm'],
        clip=cfg['training_cfg']['clip'], sw=sw, save_folder_path=args.exp_path)


if __name__ == "__main__":
    main()
