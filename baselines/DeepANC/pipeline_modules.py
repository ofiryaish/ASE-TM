import torch
import torch.nn.functional as F


gpu_id = 0


def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    return device


device = get_device()


################
# 1. peak norm #
################
def extract_peak_norm_params(batch):
    max_val = torch.max(torch.abs(batch)).to(device)
    return max_val + 1e-7


def peak_norm(tensor, norm_params):
    max_val = norm_params
    return tensor / max_val


def peak_denorm(tensor, norm_params):
    max_val = norm_params
    return tensor * max_val


################
# 2. RMS norm #
################
def extract_rms_norm_params(batch):
    rms = torch.sqrt(torch.mean(batch**2, dim=1, keepdim=True)).to(device)
    return rms


def rms_norm(tensor, norm_params):
    rms = norm_params
    return tensor / (rms + 1e-7)


def rms_denorm(tensor, norm_params):
    rms = norm_params
    return tensor * (rms + 1e-7)


####################
# 3. Mean-std norm #
####################
def extract_mean_std_norm_params(batch):
    mean = torch.mean(batch, dim=1, keepdim=True).to(device)
    std = torch.std(batch, dim=1, keepdim=True).to(device)
    return mean, std


def mean_std_norm(tensor, norm_params):
    mean, std = norm_params
    return (tensor - mean) / (std + 1e-7)


def mean_std_denorm(tensor, norm_params):
    mean, std = norm_params
    return tensor * (std + 1e-7) + mean


###################
# 4. Min-max norm #
###################
def extract_min_max_norm_params(batch):
    min_val = torch.min(batch).to(device)
    max_val = torch.max(batch).to(device)
    return min_val, max_val


def min_max_norm(tensor, norm_params):
    min_val, max_val = norm_params
    return (tensor - min_val) / (max_val - min_val + 1e-7)


def min_max_denorm(tensor, norm_params):
    min_val, max_val = norm_params
    return tensor * (max_val - min_val + 1e-7) + min_val


def extract_norm_params(batch):
    return extract_rms_norm_params(batch)


def normalize(batch, norm_params):
    return rms_norm(batch, norm_params)


def denormalize(batch, norm_params):
    return rms_denorm(batch, norm_params)


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160, norm=True):
        # self.stft = STFT(win_size, hop_size).to(device)
        self.win_size = win_size
        self.hop_size = hop_size
        self.norm = norm

    def __call__(self, wav):
        complex_spec = torch.stft(
            input=wav, n_fft=self.win_size, hop_length=self.hop_size, return_complex=True, normalized=self.norm)
        mag_spec, phase_spec = complex_spec.real.permute(0, 2, 1), complex_spec.imag.permute(0, 2, 1)
        # mag_spec, phase_spec = torch.abs(complex_spec).permute(0, 2, 1), torch.angle(complex_spec).permute(0, 2, 1)
        # real_mix, imag_mix = self.stft.stft(wav)
        feat = torch.stack([mag_spec, phase_spec], dim=1)
        return feat


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160, norm=True):
        # self.stft = STFT(win_size, hop_size).to(device)
        self.win_size = win_size
        self.hop_size = hop_size
        self.norm = norm

    def __call__(self, est, shape):
        mag_spec, phase_spec = est[:, 0, :, :].permute(0, 2, 1), est[:, 1, :, :].permute(0, 2, 1)
        complex_spec = torch.view_as_complex(torch.stack([mag_spec, phase_spec], dim=-1))
        # complex_spec = torch.polar(mag_spec, phase_spec)
        sph_est = torch.istft(
            complex_spec, n_fft=self.win_size, hop_length=self.hop_size, return_complex=False, normalized=self.norm)

        # sph_est = self.stft.istft(est)

        sph_est = F.pad(sph_est, [0, shape[1]-sph_est.shape[1]])
        return sph_est
