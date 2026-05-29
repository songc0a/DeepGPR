
import torch

def design_fir_filter(cutoff: float, fs: float, numtaps: int) -> torch.Tensor:
    n = torch.arange(numtaps, dtype=torch.float32)
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    sinc = torch.sin(2 * torch.pi * (cutoff/fs) * (n - (numtaps-1)/2)) / (torch.pi * (n - (numtaps-1)/2))
    center = (numtaps-1) // 2
    sinc[center] = 2 * cutoff/fs
    h = window * sinc

    return h / h.sum()

def apply_filter(data: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    numtaps = int(1 * (fs / cutoff))
    fir_coeff = design_fir_filter(cutoff, fs, numtaps)
    fir_coeff = fir_coeff.to(data.device)

    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded_data = torch.nn.functional.pad(data_2d, (numtaps-1, 0), mode='reflect')
        filtered = torch.nn.functional.conv1d(
            padded_data, 
            fir_coeff.view(1, 1, -1), 
            padding=0
        )
        return filtered.view(-1)

    elif data.ndim == 3:
        step, iterations, nrx = data.shape
        reshaped_data = data.permute(0, 2, 1).reshape(-1, 1, iterations)
        padded_data = torch.nn.functional.pad(reshaped_data, (numtaps-1, 0), mode='reflect')
        filtered = torch.nn.functional.conv1d(
            padded_data,
            fir_coeff.view(1, 1, -1),
            padding=0
        )
        return filtered.view(step, nrx, iterations).permute(0, 2, 1)

    else:
        raise ValueError(f"Data dimension: {data.ndim}。expected 1D or 3D tensor。")
    
def hilbert_transform(data_in, p=1):
  ns, nt, nr = data_in.shape
  transforms = torch.fft.fftn(data_in,dim=1)
  #print(transforms.shape)
  transforms[:,1:nt//2,:]      *= 2.0
  transforms[:,nt//2 + 1: nt,:]  = 0+0j
  transforms[:,0,:] = 0;
  data_out = torch.abs(torch.fft.ifftn(transforms,dim=1))**p
  return data_out
