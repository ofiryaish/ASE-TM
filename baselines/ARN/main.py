
import os
import argparse
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from dataloaders.train_dataset_utilis import create_dataloader, create_dataset
from models.simulate_paths import (instance_simulator, process_signals_through_primary_path,
                                   process_signals_through_secondary_path, randomize_reverberation_time)
from utils.util import load_config

import baselines.ARN.model as model_py
from baselines.helpers import save_model
from baselines.ARN.ola import create_chuncks, merge_chuncks


def pad_num_to_len(num, length=10):
    num_str = f"{num:.{length-2}f}"
    if len(num_str) > length:
        if '.' in num_str:
            num_str = num_str[:length]
    return num_str


def predict(signals, model, device, window_size=256):
    with torch.autocast(
        dtype=torch.float32, device_type=device.type
    ):
        inputs, rest = create_chuncks(signals.unsqueeze(1), window_size)
        outputs = model(inputs.squeeze(1))
        anti_signals = merge_chuncks(outputs.unsqueeze(1), rest).squeeze(1)
        return anti_signals


def valid(model, validation_data, reverberation_times, simulator, sef_factor,
          criterion, device):
    model.eval()

    total_loss = 0
    total_items = 0
    with torch.no_grad():
        for batch in validation_data:
            # get the inputs and load to device
            clean_audio, noisy_audio, _noisy_mag, _noisy_pha = batch
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            # TODO: without t60 I can process the signals through the primary path beforehead
            t60 = randomize_reverberation_time(reverberation_times)
            # Process signals through primary path
            clean_audio = process_signals_through_primary_path(clean_audio, simulator, t60)
            audio_g = predict(noisy_audio, model, device)
            noisy_audio = process_signals_through_primary_path(noisy_audio, simulator, t60)

            # Process the generated signal through secondary path
            audio_g = process_signals_through_secondary_path(audio_g, simulator, t60, sef_factor=sef_factor)
            audio_g = audio_g + noisy_audio

            loss = criterion(clean_audio, audio_g)

            total_loss += loss.mean().item() * noisy_audio.shape[0]
            total_items += noisy_audio.shape[0]

        model.train()

    return total_loss / total_items


def train(model, validation_data, train_data, reverberation_times, simulator,
          sef_factor, criterion, optimizer, scheduler, scaler, device, sw, save_folder_path, cfg):
    model.train()

    # train model
    steps = 0
    total_loss = 0
    total_items = 0
    best_validation_total_loss = np.inf
    max_epochs = cfg["training_cfg"]["training_epochs"]
    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_items = 0
        for i, batch in enumerate(train_data):
            optimizer.zero_grad()

            # get the inputs and load to device
            clean_audio, noisy_audio, _noisy_mag, _noisy_pha = batch
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            # TODO: without t60 I can process the signals through the primary path beforehead
            t60 = randomize_reverberation_time(reverberation_times)
            # Process signals through primary path
            clean_audio = process_signals_through_primary_path(clean_audio, simulator, t60)
            audio_g = predict(noisy_audio, model, device)
            noisy_audio = process_signals_through_primary_path(noisy_audio, simulator, t60)

            # Process the generated signal through secondary path
            audio_g = process_signals_through_secondary_path(audio_g,  simulator, t60, sef_factor="random")
            audio_g = audio_g + noisy_audio

            loss = criterion(clean_audio, audio_g)

            total_items += noisy_audio.shape[0]
            epoch_items += noisy_audio.shape[0]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training_cfg"]["clip"])
            scaler.step(optimizer)
            scaler.update()

            # calculate loss
            running_loss = loss.item()
            epoch_loss += running_loss * noisy_audio.shape[0]
            total_loss += running_loss * noisy_audio.shape[0]

            steps += 1
            if i % 300 == 0:
                print('Epoch [{}/{}], Iter [{}], epoch_loss = {:.8f}, total_loss = {:.8f}'.format(
                    epoch, max_epochs, i, epoch_loss / epoch_items, total_loss / total_items), flush=True)
                sw.add_scalar("Training/Total loss", total_loss / total_items, steps)
                sw.add_scalar("Training/Epoch loss", epoch_loss / epoch_items, steps)

        scheduler.step()

        # save model after each epoch
        save_model(model, optimizer, scheduler, epoch, f"{save_folder_path}/g_{steps:08d}.pth")

        # validation after each epoch
        validation_total_loss = valid(
            model=model, validation_data=validation_data,
            reverberation_times=reverberation_times, simulator=simulator, sef_factor=sef_factor,
            criterion=criterion, device=device)
        print(
            'Epoch [{}/{}], Validation loss = {:.8f}'.format(
                epoch, max_epochs, validation_total_loss), flush=True)
        sw.add_scalar("Validation/Loss", validation_total_loss, epoch)

        # save the best model based on validation loss
        if validation_total_loss < best_validation_total_loss and save_folder_path is not None:
            save_model(model, optimizer, scheduler, epoch, f"{save_folder_path}/best_model.pth")
            best_validation_total_loss = validation_total_loss


def main():
    parser = argparse.ArgumentParser(description='ARN Model')
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='ARN_exp')
    parser.add_argument('--config', default='recipes/ARN/ARN.yaml')

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

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg["env_setting"]["seed"])
    torch.manual_seed(cfg["env_setting"]["seed"])
    torch.cuda.manual_seed(cfg["env_setting"]["seed"])  # must have GPU

    ###############################################################################
    # Build the model
    ###############################################################################
    window_size = 256
    N = 512
    sef_factor = "random"  # 0.5**0.5, "inf"
    model = model_py.SHARNN(window_size, N, 4*N, cfg["model_cfg"]["nlayers"], cfg["training_cfg"]["dropouth"])

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["training_cfg"]["learning_rate"],
                                 weight_decay=cfg["training_cfg"]["wdecay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Example: StepLR
    # scheduler = None
    scaler = torch.amp.GradScaler()  # torch.cuda.amp.GradScaler()
    model = model.to(device)
    criterion = torch.nn.MSELoss(reduction="mean")

    train(model=model, validation_data=validation_loader, train_data=train_loader,
          reverberation_times=reverberation_times, simulator=simulator, sef_factor=sef_factor, criterion=criterion,
          optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device,
          sw=sw, save_folder_path=args.exp_path, cfg=cfg)


if __name__ == "__main__":
    main()
