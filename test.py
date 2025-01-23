import torch

checkpoint_path = 'exp/SEMamba_active_v8/g_00057000.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
print(checkpoint.keys())