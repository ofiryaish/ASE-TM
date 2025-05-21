from dataloaders.dataloader_vctk_clip import VCTKDemandClipDataset
from torch.utils.data import DistributedSampler, DataLoader


def create_dataset(cfg, train=True, split=True, test=False):
    """Create dataset based on configuration."""
    if train and test:
        raise ValueError("train and test cannot be True at the same time.")
    if test:
        clean_json = cfg['data_cfg']['test_clean_json']
    else:
        clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False

    try:
        const_clip_value = cfg['training_cfg']['const_clip_value']
    except KeyError:
        const_clip_value = None
    print(f"const_clip_value: {const_clip_value}")
    try:
        min_clip_value = cfg['training_cfg']['min_clip_value']
        max_clip_value = cfg['training_cfg']['max_clip_value']
    except KeyError:
        min_clip_value = None
        max_clip_value = None

    return VCTKDemandClipDataset(
        clean_json=clean_json,
        const_clip_value=const_clip_value,
        min_clip_value=min_clip_value,
        max_clip_value=max_clip_value,
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
        normalize=cfg['training_cfg']['data_normalization']
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
