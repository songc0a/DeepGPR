"""Module-B gradient accuracy on the 4 x 4 x 2 m model: one shot, MS-SSIM, at the start model."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]; sys.path.insert(0, HERE); os.chdir(HERE)
import bench_fwi as b
import benchmark_tucker as bt
import DeepGPR
dev = torch.device("cuda")
case, observed, shots, region, zone = b.load_case("3d", False, dev, "v6")
eps0, sig0 = b.initial_models("3d", case, "v6")
obj = b.DataObjective(dict(loss="mssim"), observed, shots, 1.0, dev)

def grad(sketch, shot=17):
    e = eps0.clone().requires_grad_(True); s = sig0.clone().requires_grad_(True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    kw = dict(wavefield_sketch=sketch) if sketch is not None else {}
    d = DeepGPR.compute(eps_r=e, sigma=s, **kw, **shots[shot])[-1]
    obj(d, observed[shot], shots[shot]).backward()
    torch.cuda.synchronize()
    return e.grad[:, case["air"]:].flatten(), s.grad[:, case["air"]:].flatten(), time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2**30

ge, gs, t0, m0 = grad(None); ge, gs, t0, m0 = grad(None)  # warm-up, then measure
print("exact: %.2f s, peak %.2f GiB" % (t0, m0), flush=True)
for rank, rhs, where, seg in ((24, "from_e", "host", 8), (24, "from_e", "host", 16), (24, "from_e", "host", 32), (32, "from_e", "host", 32)):
    cfg = bt.sketch_config({"sketch": dict(ranks=rank, oversampling=8, segment_steps=seg, rhs=rhs, recovery_device=where)})
    he, hs, t, m = grad(cfg)
    cos_e = float(torch.nn.functional.cosine_similarity(ge, he, dim=0)); err_e = float((he - ge).norm() / ge.norm())
    cos_s = float(torch.nn.functional.cosine_similarity(gs, hs, dim=0)); err_s = float((hs - gs).norm() / gs.norm())
    print("seg %2d rank %2d %-6s %-6s: eps grad cos %.6f rel err %.3e | sigma grad cos %.6f rel err %.3e | %.2f s, peak %.2f GiB" % (seg, rank, rhs, where, cos_e, err_e, cos_s, err_s, t, m), flush=True)
