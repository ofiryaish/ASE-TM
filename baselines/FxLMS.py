import numpy as np

import sys

import torch
import torch.nn.functional as F
import torchaudio

sys.path.append('/a/home/cc/students/cs/mishaly1/repos/data/code/speechbrain/_separation')
from data_utils.data_tools import sef
from tools.simulator import RIRGenSimulator, ANSTISIGNAL_ERROR, NOISE_ERROR
from tools.utils import get_wavfile_for_eval
from tools.helpers import get_device, nmse

device = 'cuda'
rir_samples=512
sr = 16000
simulator = RIRGenSimulator(sr=sr, reverbation_times=[0.2], device=device, rir_samples=rir_samples, hp_filter=False)
fftconvolve_valid = torchaudio.transforms.FFTConvolve(mode="valid")

class FxLMS_HER:
    def __init__(self, w_len, mu):
        self.w = torch.zeros(1, w_len,dtype=torch.float).to(device).requires_grad_(True)
        self.x_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.st_x_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.optimizer = torch.optim.SGD([self.w], lr=mu)

    def predict(self, s_t):
        """
        Update the adaptive filter weights based on the reference signal x and the error signal d.
        
        Parameters:
        x: Current sample of the reference noise signal
        d: Current sample of the error signal (difference between desired and actual signals)
        
        Returns:
        y: Current sample of the output signal (anti-noise)
        """
        # Update the reference signal buffer
        # self.x_buf = torch.roll(self.x_buf,1,1)
        # self.x_buf[0,0] = x

        self.st_x_buf = torch.roll(self.st_x_buf,1,1)
        self.st_x_buf[0,0] = s_t

        # Calculate the output signal (anti-noise)
        y = self.w @ self.st_x_buf.t()

        # power = self.x_buf @ self.x_buf.t() # FxNLMS different from FxLMS
        return y
    
    def step(self,e_value):
        # e_value = e_value *  torch.flip(self.st_x_buf,[1]) 
        loss = torch.mean(e_value**2)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()
        # self.w += 2* self.mu * e_value *  torch.flip(self.x_buf,[1])


class FxLMS_COP:
    def __init__(self, w_len, mu):
        self.w = torch.zeros(1, w_len,dtype=torch.float).to(device)
        self.x_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.st_x_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.mu = mu

    def predict(self, x, st_x):
        """
        Update the adaptive filter weights based on the reference signal x and the error signal d.
        
        Parameters:
        x: Current sample of the reference noise signal
        d: Current sample of the error signal (difference between desired and actual signals)
        
        Returns:
        y: Current sample of the output signal (anti-noise)
        """
        # Update the reference signal buffer
        self.x_buf = torch.roll(self.x_buf,1,1)
        self.x_buf[0,0] = x

        self.st_x_buf = torch.roll(self.st_x_buf,1,1)
        self.st_x_buf[0,0] = st_x
        # Calculate the output signal (anti-noise)
        y = self.w @ self.x_buf.t()
        # y = fftconvolve_valid(torch.flip(self.x_buf,[1]), self.w)
        # power = self.x_buf @ self.x_buf.t() # FxNLMS different from FxLMS
        return y
    
    def step(self,e_value):
        grad = 2 * self.mu * e_value * self.st_x_buf
        grad = torch.clamp(grad, -1e-4, 1e-4)
        self.w += grad
        return grad.mean().item()




def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std, mean, std

def denormalize(tensor, mean, std):
    return tensor * std + mean

def execute_(wavfile, mu,simulator=simulator,gama="inf", reverbation_time=0.2):
    fxlms = FxLMS_COP(w_len=rir_samples, mu=mu)

    # wavfile, mean, std = normalize(wavfile)

    ys = []
    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)

    st = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].squeeze(0).to(device)
    
    padded_signal = torch.nn.functional.pad(wavfile, (256,0), mode='constant', value=0)
    pt_signal = simulator.simulate(padded_signal, reverbation_time, NOISE_ERROR)[0]
    st_signal = simulator.simulate(padded_signal, reverbation_time, ANSTISIGNAL_ERROR)[0]

    signal = padded_signal[0].to(device)

    for i in range(signal.shape[0] - 256):
        y = fxlms.predict(signal[i+256],st_signal[i])
        ys.append(y.item())

        y_buf = torch.roll(y_buf,1,1)
        y_buf[0,0] = y

        st_y = fftconvolve_valid(torch.flip(sef(y_buf,gama),[1]), st).item()
        # st_y = torch.dot(sef(y_buf,gama).view(-1), st.view(-1))
        # s_ys = simulator.simulate(wavfile, reverbation_time, ANSTISIGNAL_ERROR)[0][-1]
        loss = pt_signal[i]-st_y
        # e[i-256] = diff.item()
        
        # first_i = max(i-256 - rir_samples, 0)
        # loss = 10*torch.log10(torch.sum(e[first_i:i-256]**2)/torch.max(torch.sum(pt_signal[first_i:i-256] ** 2), torch.tensor(1e-6).to(device)))

        fxlms.step(loss, st_signal[i])

        # if (i + 1) % 2000 == 0 and i > 3000:

        #     r = nmse(pt_signal[i-256-1000: i - 256], torch.tensor(st_y_buff[-1000:]).to(device))
        #     print(f"{i}. NMSE: {r}", flush=True)

    ys = torch.tensor(ys).to(device)
    # ys = denormalize(ys, mean, std)
    return ys


def execute_v2(wavfile, mu,simulator=simulator,gama="inf", reverbation_time=0.2):
    # fxlms = FxLMS_COP(w_len=rir_samples, mu=mu)
    fxlms = FxLMS_HER(w_len=rir_samples, mu=mu)
    # wavfile, mean, std = normalize(wavfile)

    ys = []
    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)

    st = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].squeeze(0).to(device)
    
    padded_signal = torch.nn.functional.pad(wavfile, (256,0), mode='constant', value=0)
    pt_signal = simulator.simulate(padded_signal, reverbation_time, NOISE_ERROR)[0]
    st_signal = simulator.simulate(padded_signal, reverbation_time, ANSTISIGNAL_ERROR)[0]

    signal = wavfile[0].to(device)

    for i in range(signal.shape[0] + 256):
        y = fxlms.predict(st_signal[i])

        # y_buf = torch.roll(y_buf,1,1)
        # y_buf[0,0] = y
        ys.append(y.item())

        # st_y = fftconvolve_valid(torch.flip(sef(y_buf,gama),[1]), st).item()
        # st_y = torch.dot(sef(y_buf,gama).view(-1), st.view(-1))
        # s_ys = simulator.simulate(wavfile, reverbation_time, ANSTISIGNAL_ERROR)[0][-1]
        loss = pt_signal[i]-sef(y,gama)
        # e[i-256] = diff.item()
        
        # first_i = max(i-256 - rir_samples, 0)
        # loss = 10*torch.log10(torch.sum(e[first_i:i-256]**2)/torch.max(torch.sum(pt_signal[first_i:i-256] ** 2), torch.tensor(1e-6).to(device)))

        fxlms.step(loss)

        # if (i + 1) % 2000 == 0 and i > 3000:

        #     r = nmse(pt_signal[i-256-1000: i - 256], torch.tensor(st_y_buff[-1000:]).to(device))
        #     print(f"{i}. NMSE: {r}", flush=True)

    ys = torch.tensor(ys)[256:].to(device)
    # ys = denormalize(ys, mean, std)
    return ys

def execute_v3(wavfile, mu,simulator=simulator,gama="inf", reverbation_time=0.2,thf=False):
    # fxlms = FxLMS_COP(w_len=rir_samples, mu=mu)
    fxlms = FxLMS_COP(w_len=rir_samples, mu=mu)
    # wavfile, mean, std = normalize(wavfile)

    ys = []
    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)

    st = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].squeeze(0).to(device)
    
    padded_signal = torch.nn.functional.pad(wavfile, (256,0), mode='constant', value=0)
    pt_signal = simulator.simulate(padded_signal, reverbation_time, NOISE_ERROR)[0]

    if thf:
        padded_signal = sef(padded_signal, gama)
    st_signal = simulator.simulate(padded_signal, reverbation_time, ANSTISIGNAL_ERROR)[0]

    signal = wavfile[0].to(device)
    loss_sum = 0
    for i in range(signal.shape[0]):
        # x = signal[i] if i < signal.shape[0] else 0
        y = fxlms.predict(signal[i], st_signal[i])

        y_buf = torch.roll(y_buf,1,1)
        y_buf[0,0] = y
        ys.append(y.item())

        st_y = fftconvolve_valid(torch.flip(sef(y_buf,gama),[1]), st).item()
        # st_y = torch.dot(torch.flip(sef(y_buf,gama),[1]).view(-1), st.view(-1))
        # s_ys = simulator.simulate(wavfile, reverbation_time, ANSTISIGNAL_ERROR)[0][-1]
        loss = pt_signal[i]-st_y
        # e[i-256] = diff.item()
        # loss = torch.clamp(loss, -0.1, 0.1)

        # first_i = max(i-256 - rir_samples, 0)
        # loss = 10*torch.log10(torch.sum(e[first_i:i-256]**2)/torch.max(torch.sum(pt_signal[first_i:i-256] ** 2), torch.tensor(1e-6).to(device)))

        loss_sum += fxlms.step(loss)

        if (i + 1) % 500 == 0 and i > 3000:
            print(f"{i + 1}. Loss: {loss_sum/(i+1)}", flush=True)
            loss_sum = 0
        #     r = nmse(pt_signal[i-256-1000: i - 256], torch.tensor(st_y_buff[-1000:]).to(device))
        #     print(f"{i}. NMSE: {r}", flush=True)

    ys = torch.tensor(ys).to(device)
    # ys = denormalize(ys, mean, std)
    return ys

def predict_signal(wavfile,mu=None, simulator=simulator, gama="inf",reverbation_time=0.2, thf=False):
    y = execute_v3(wavfile,mu=mu, simulator=simulator, gama=gama,reverbation_time=reverbation_time, thf=thf)
    # e = pt - y -> y = pt - e
    return y.unsqueeze(0)


# def eval_callback(wavfile,mu, simulator=simulator, gama="inf",reverbation_time=0.2):
#     e, pt = execute_v3(wavfile, mu=mu,simulator=simulator, gama=gama,reverbation_time=reverbation_time)
    
#     if gama != "inf":
#         # e = pt - y -> y = pt - e
#         y = pt - e
#         sef_y = sef(y, gama)
#         e = pt - sef_y
        
#     denominator = torch.sum(pt**2)
#     numerator = torch.sum(torch.tensor(e)**2)
#     nmse_value = 10 * torch.log10(numerator / denominator)
#     return (nmse_value,)


def format_ys(ys):
    if len(ys) < rir_samples:
        return torch.tensor([0]*(rir_samples-len(ys)) + ys).reshape(1, -1).to(device)
    return torch.tensor(ys).reshape(1, -1).to(device)


def eval_callback_v2(wavfile):
    fxlms = FxLMS_COP(w_len=rir_samples, mu=0.00001)

    sr = 16000
    reverbation_time = 0.2
    simulator = RIRGenSimulator(sr=sr, reverbation_times=[reverbation_time], device=device, rir_samples=rir_samples, hp_filter=False)

    e = []
    d = simulator.simulate(wavfile, reverbation_time, NOISE_ERROR)[0]
    signal = wavfile[0]
    ys = []
    diff = 0
    for i in range(signal.shape[0]):
        ys.append(fxlms.predict(signal[i]).item())
        anti_signal = format_ys(ys[-rir_samples:])
        y = simulator.simulate(anti_signal, reverbation_time, ANSTISIGNAL_ERROR, padding="valid")[0][0]
        diff = d[i]-y

        fxlms.step(diff)
        e.append(diff.item())

    denominator = torch.sum(pt**2)
    numerator = np.sum(np.array(e)**2)
    nmse_value = 10 * np.log10(numerator / denominator)
    return (nmse_value,)


def eval_callback_v3(wavfile):
    mu = 0.0001
    sr = 16000
    reverbation_time = 0.2
    
    simulator = RIRGenSimulator(sr=sr, reverbation_times=[reverbation_time], device=device, rir_samples=rir_samples, hp_filter=False)
    secondary_path = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].view(-1).cpu().numpy()
    d = simulator.simulate(wavfile, reverbation_time, NOISE_ERROR)[0]

    wavfile = wavfile[0].cpu().numpy()

    W = np.zeros(rir_samples)
    e = np.zeros_like(d)
    y_hat = np.zeros_like(d)  # Output signal based on W and x

    # Main FxLMS loop
    for n in range(len(wavfile)):
        # Generate the output signal using the current filter weights W
        if n < rir_samples:
            y_hat[n] = np.dot(W[:n+1], wavfile[:n+1][::-1])
        else:
            y_hat[n] = np.dot(W, wavfile[n-rir_samples+1:n+1][::-1])

        # Filter the reference signal through the secondary path
        y = np.convolve(y_hat, secondary_path, mode='same')

        # Calculate the error signal
        e[n] = d[n] - y[n]

        if n < rir_samples:
            padded_x = np.zeros(rir_samples)
            padded_x[rir_samples-(n+1):] = wavfile[:n+1] 
            W += 2 * mu * e[n] * padded_x[::-1]
        else:
            W += 2 * mu * e[n] * wavfile[n-rir_samples+1:n+1][::-1]

    denominator = torch.sum(d**2)
    numerator = np.sum((e)**2)
    nmse_value = 10 * np.log10(numerator / denominator)
    return (nmse_value,)

def main():

    filepath = "data/code/speechbrain/_separation/data_utils/tests/samples/Trolley.wav"
    wavfile = get_wavfile_for_eval(filepath)
    
    simulate_fxlms(wavfile, fxlms)




if __name__ == "__main__":
    main()
