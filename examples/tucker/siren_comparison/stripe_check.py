"""Why are the 2D Tucker artefacts vertical stripes? (crosshole: sources y=0, receivers y=ny-1)

Usage: ``python stripe_check.py [--protocol v2]`` (GPU; reads the protocol's
2D FWI results, writes ``results/full_<protocol>/stripe_check.json``).

A) anisotropy of the model update  Delta = final - initial:
   S_v = smooth_x(highpass_y(Delta))  (fast along y, slow along x: vertical stripes)
   S_h = smooth_y(highpass_x(Delta))  (horizontal stripes)
B) data sensitivity: loss of truth + A cos(2 pi y / L) vs truth + A cos(2 pi x / L)
C) loss of a Tucker-SIREN result with its vertical / horizontal stripe part removed
D) same anisotropy for the first dense gradient at the initial model
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bench_fwi  # noqa: E402
import DeepGPR  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", default="v2")
OUT = bench_fwi.result_dir(False, parser.parse_args().protocol)
RUNS = OUT / "fwi_runs" / "2d"
dev = torch.device("cuda")


def parts(delta):
    d = np.asarray(delta, dtype=np.float64)
    hp_y = d - gaussian_filter1d(d, 3, axis=1, mode="nearest")
    hp_x = d - gaussian_filter1d(d, 3, axis=0, mode="nearest")
    s_v = gaussian_filter1d(hp_y, 10, axis=0, mode="nearest")
    s_h = gaussian_filter1d(hp_x, 10, axis=1, mode="nearest")
    return s_v, s_h


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


case, observed, shots, _, _ = bench_fwi.load_case("2d", False, dev)
options, obs = shots[0], observed[0]
scale = float(obs.square().sum())


def loss(eps, sigma):
    with torch.no_grad():
        d = DeepGPR.compute(eps_r=torch.as_tensor(eps, dtype=torch.float32, device=dev),
                            sigma=torch.as_tensor(sigma, dtype=torch.float32, device=dev),
                            save_wavefield_history=False, **options)[-1]
    return float(0.5 * (d - obs).square().sum() / scale)


out = {}
init = torch.load(RUNS / "initial.pt")
eps0, sig0 = init["eps"].numpy(), init["sigma"].numpy()
true_eps, true_sig = case["eps"].cpu().numpy(), case["sigma"].cpu().numpy()

# A
print("A) rms of vertical-stripe part S_v / horizontal-stripe part S_h of (final - initial)")
rows = {"truth": (true_eps, true_sig)}
for p in sorted(RUNS.glob("*_final.pt")):
    data = torch.load(p)
    rows[p.name.split("|")[1] + "|" + p.name.split("|")[2].split("_")[0]] = (data["eps"].numpy(), data["sigma"].numpy())
out["A"] = {}
for name, (e, s) in rows.items():
    ve, he = parts(e - eps0)
    vs, hs = parts(np.log10(np.maximum(s, 1e-6)) - np.log10(np.maximum(sig0, 1e-6)))
    out["A"][name] = dict(eps_v=rms(ve), eps_h=rms(he), sig_v=rms(vs), sig_h=rms(hs))
    print(f"  {name:28s} eps S_v {rms(ve):.3f} S_h {rms(he):.3f} (v/h {rms(ve)/rms(he):5.2f})   "
          f"log10 sigma S_v {rms(vs):.3f} S_h {rms(hs):.3f} (v/h {rms(vs)/max(rms(hs),1e-9):5.2f})")

# B
print("B) loss of truth + 0.3 cos(2 pi y / L) (vertical stripes) vs + 0.3 cos(2 pi x / L) (horizontal)")
nx, ny = true_eps.shape
x = np.arange(nx)[:, None]
y = np.arange(ny)[None, :]
out["B"] = {}
for period in (4, 6, 8, 12, 20):
    lv = loss(np.maximum(true_eps + 0.3 * np.cos(2 * np.pi * y / period) + 0 * x, 1.0), true_sig)
    lh = loss(np.maximum(true_eps + 0.3 * np.cos(2 * np.pi * x / period) + 0 * y, 1.0), true_sig)
    out["B"][period] = dict(vertical=lv, horizontal=lh)
    print(f"  period {period:3d} cells: vertical {lv:.2e}  horizontal {lh:.2e}  ratio h/v {lh / lv:7.1f}")

# C
print("C) Tucker-SIREN finals: loss with stripe parts removed from eps")
out["C"] = {}
for label in ("tsiren_2x32_r8", "tsiren_2x32_r16", "tsiren_2x32_r24", "direct_tucker_r16"):
    data = torch.load(RUNS / f"2d|{label}|seed0_final.pt")
    e, s = data["eps"].numpy(), data["sigma"].numpy()
    ve, he = parts(e - eps0)
    l0, lv, lh = loss(e, s), loss(np.maximum(e - ve, 1.0), s), loss(np.maximum(e - he, 1.0), s)
    out["C"][label] = dict(final=l0, minus_vertical=lv, minus_horizontal=lh, rms_v=rms(ve), rms_h=rms(he))
    print(f"  {label:20s} loss {l0:.4f} | without S_v (rms {rms(ve):.3f}) {lv:.4f} | without S_h (rms {rms(he):.3f}) {lh:.4f}")

# D
eps_leaf = torch.tensor(eps0, device=dev, requires_grad=True)
d = DeepGPR.compute(eps_r=eps_leaf, sigma=torch.tensor(sig0, device=dev), **options)[-1]
(0.5 * (d - obs).square().sum() / scale).backward()
g = eps_leaf.grad.cpu().numpy()
gv, gh = parts(g)
out["D"] = dict(grad_v=rms(gv), grad_h=rms(gh), grad_rms=rms(g))
print(f"D) first dense eps gradient: rms {rms(g):.3e}, S_v {rms(gv):.3e}, S_h {rms(gh):.3e} (v/h {rms(gv)/rms(gh):.2f})")
json.dump(out, open(OUT / "stripe_check.json", "w"), indent=1)
