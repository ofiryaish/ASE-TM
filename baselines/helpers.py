import torch


def save_model(model, optimizer, scheduler, epoch, model_path):
    data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(data, model_path)
    return


def load_model(optimizer, scheduler, cp_file_path):
    data = torch.load(cp_file_path)

    model_state_dict = data['model_state_dict']
    optimizer.load_state_dict(data['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(data['scheduler_state_dict'])
    epoch = data['epoch']
    return model_state_dict, optimizer, scheduler, epoch


def delay_signal(hop_len, frames_delay, signals):
    if frames_delay == 0:
        return signals
    pad_len = frames_delay * hop_len
    pad = torch.zeros((signals.shape[0], pad_len)).to(signals.device)
    signals = torch.cat([pad, signals[:, :-pad_len]], dim=-1)
    return signals
