"""Forward timing (CUDA events, median of 10 after 3 warmups) for a 3D case with chosen CPML faces."""
import sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
import torch, DeepGPR
pml = [int(v) for v in sys.argv[2].split(",")]
modes = sys.argv[3].split(",")
n, nt = int(sys.argv[4]) if len(sys.argv) > 4 else 80, int(sys.argv[5]) if len(sys.argv) > 5 else 500
dev = torch.device("cuda")
eps = torch.full((n, n, n), 4.0, device=dev); sig = torch.full((n, n, n), 1e-3, device=dev)
dt = 1.0e-11
src = DeepGPR.ricker(4e8, nt, dt, 2.5e-9).reshape(1, nt, 1).to(dev)
sl = torch.tensor([[[n // 2, n // 2, n // 2]]], dtype=torch.int32, device=dev)
rl = torch.tensor([[[n // 2 - 8 + i, n // 2 + 10, n // 2] for i in range(16)]], dtype=torch.int32, device=dev)
out = []
for mode in modes:
    kw = {}
    if mode == "fdtd": kw["save_wavefield_history"] = False
    elif mode == "int8": kw["wavefield_compression"] = "int8"
    else: kw["wavefield_storage_dtype"] = {"fp32": torch.float32, "fp16": torch.float16}[mode]
    def run():
        e = eps.clone().requires_grad_(mode != "fdtd")
        r = DeepGPR.compute(eps_r=e, sigma=sig, device=dev, dx=(0.01, 0.01, 0.01), dt=dt, source_amplitudes=src,
                            source_location=sl, receiver_location=rl, pmlthick=pml, fdtd_order=4, mode=3, **kw)
        return r
    times = []
    for it in range(13):
        torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); r = run(); b.record(); torch.cuda.synchronize()
        del r
        if it >= 3: times.append(a.elapsed_time(b))
    times.sort(); out.append(f"{mode}={times[len(times)//2]:.2f}")
print(" ".join(out))
