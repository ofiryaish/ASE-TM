import json
import random

from dataloaders.dataloader_vctk_reverb import VCTKDemandReverbDataset
from torch.utils.data import DistributedSampler, DataLoader
from models.simulate_paths import PreGeneratedSimulator


def create_dataset(cfg, train=True, split=True, test=False, const_rirs=False):
    """Create dataset based on configuration."""
    if train and test:
        raise ValueError("train and test cannot be True at the same time.")
    if test:
        clean_json = cfg['data_cfg']['test_clean_json']
    else:
        clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False

    simulator = PreGeneratedSimulator(
        sr=cfg['stft_cfg']['sampling_rate'], rir_files_path=cfg['rir_cfg']['rir_files_path'],
        rir_samples=-1, device="cpu", v=cfg['rir_cfg']['version'])

    if const_rirs:
        with open(clean_json) as f:
            data = json.load(f)
            n_rirs = len(data)
        seed = 1234
        random.seed(seed)
        const_rirs_files = random.sample(simulator.rir_files, n_rirs)
        print("The hash of the selected RIRs is: ", end="")
        print(hash(tuple(const_rirs_files)))
    else:
        const_rirs_files = None

    return VCTKDemandReverbDataset(
        clean_json=clean_json,
        simulator=simulator,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        pcs=pcs,
        normalize=cfg['training_cfg']['data_normalization'],
        const_rirs_files=const_rirs_files
    )


def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )
