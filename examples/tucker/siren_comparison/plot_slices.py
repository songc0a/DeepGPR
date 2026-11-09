"""Depth slice and section of FWI results on the 4 x 4 x 2 m model next to the truth (CPU).

Usage: ``python plot_slices.py OUT.png "label=path_final.pt" ...`` (paths relative to this folder).
"""
import os, sys, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
sys.argv = ["x"]; sys.path.insert(0, HERE); os.chdir(HERE)
import bench_fwi as b
from metrics import model_errors
dev = torch.device("cpu")
PROTOCOL = os.environ.get("PROTOCOL", "v6")  # v7: the independent-sigma model
case, _, _, region, zone = b.load_case("3d", False, dev, PROTOCOL)
case["eps_start"], case["sigma_start"] = b.initial_models("3d", case, PROTOCOL)
air = case["air"]; nx = case["eps"].shape[0]
deep = (slice(None), slice(air + 25, None), slice(None))
panels = [("true", case["eps"], case["sigma"]), ("initial", case["eps_start"], case["sigma_start"])]
for label, path in items:
    d = torch.load(path); panels.append((label, d["eps"], d["sigma"]))
for label, e, s in panels[1:]:
    print("%-34s eps upper %.3f lower %.3f whole %.3f SSIM %.3f | log10 sigma upper %.3f whole %.3f SSIM %.3f" % (
        label, model_errors(e, case["eps"], zone)["rmse"], model_errors(e, case["eps"], deep)["rmse"], model_errors(e, case["eps"], region)["rmse"],
        model_errors(e, case["eps"], region)["ssim"], model_errors(s, case["sigma"], zone, log=True)["rmse"], model_errors(s, case["sigma"], region, log=True)["rmse"],
        model_errors(s, case["sigma"], region, log=True)["ssim"]))
true_log = torch.log10(case["sigma"][:, air:].clamp_min(1e-6))
sig_lo, sig_hi = float(true_log.min()), float(true_log.max())
n = len(panels)
fig, ax = plt.subplots(3, n, figsize=(3.7 * n, 8.6))
for j, (title, e, s) in enumerate(panels):
    ax[0, j].imshow(e[nx // 2].numpy(), cmap="viridis", vmin=1, vmax=8, extent=(0, 4, 2.2, 0), aspect="equal")
    ax[1, j].imshow(e[:, air + 12].numpy(), cmap="viridis", vmin=3, vmax=8, extent=(0, 4, 4, 0), aspect="equal")
    ax[2, j].imshow(torch.log10(s[nx // 2].clamp_min(1e-6)).numpy(), cmap="magma", vmin=sig_lo, vmax=sig_hi, extent=(0, 4, 2.2, 0), aspect="equal")
    ax[0, j].set_title(title, fontsize=9)
ax[0, 0].set_ylabel("eps, x = 2 m\ndepth (m)"); ax[1, 0].set_ylabel("eps, depth 0.5 m\nx (m)"); ax[2, 0].set_ylabel("log10 sigma, x = 2 m\ndepth (m)")
for a in ax[2]: a.set_xlabel("z (m)")
fig.suptitle("4 x 4 x 2 m model, 250 MHz, 36 shots, MS-SSIM", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=105)
