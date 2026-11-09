"""Round-7 sigma slices through the anomalies (CPU). Usage: sigma7.py OUT.png label=path ..."""
import os, sys, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
sys.argv = ["x"]; sys.path.insert(0, HERE); os.chdir(HERE)
import bench_fwi as b
dev = torch.device("cpu")
case, _, _, region, zone = b.load_case("3d", False, dev, "v7")
case["eps_start"], case["sigma_start"] = b.initial_models("3d", case, "v7")
air, dx = case["air"], 0.04
panels = [("true", case["sigma"]), ("initial", case["sigma_start"])] + [(l, torch.load(p)["sigma"]) for l, p in items]
lo, hi = -3.0, -2.1
views = [("section x = 1.2 m (sphere)", lambda s: s[30, :, :], (0, 4, 2.2, 0)),
         ("depth 1.0 m (sphere)", lambda s: s[:, air + 25, :], (0, 4, 4, 0)),
         ("depth 1.5 m (resistive box)", lambda s: s[:, air + 37, :], (0, 4, 4, 0)),
         ("depth 0.75 m (conductive layer)", lambda s: s[:, air + 18, :], (0, 4, 4, 0))]
fig, ax = plt.subplots(len(views), len(panels), figsize=(3.3 * len(panels), 3.0 * len(views)))
for j, (label, s) in enumerate(panels):
    ls = torch.log10(s.clamp_min(1e-6))
    for i, (vt, f, ext) in enumerate(views):
        ax[i, j].imshow(f(ls).numpy(), cmap="magma", vmin=lo, vmax=hi, extent=ext, aspect="equal")
        if j == 0: ax[i, j].set_ylabel(vt, fontsize=8)
    ax[0, j].set_title(label, fontsize=8)
fig.suptitle("log10 sigma (S/m), round 7 (sigma independent of eps); colour range -3.0 ... -2.1", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=100)
