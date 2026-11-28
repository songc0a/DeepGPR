"""The CPU backend must give bitwise identical results for any OpenMP thread count."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = r"""
import sys, torch
sys.path.insert(0, sys.argv[2])
import DeepGPR
torch.manual_seed(0)
results = {}
for shape, order, pml in (
    ((24, 26), 2, [2, 3, 3, 2]),
    ((24, 26), 8, [2, 3, 3, 2]),
    ((9, 10, 8), 4, [2, 3, 3, 2, 2, 1]),
    # Fewer shot/x slices than worker threads; y must remain parallel work.
    ((2, 26), 2, 0),
):
    three_d = len(shape) == 3
    eps_r = (2.0 + 4.0 * torch.rand(shape)).requires_grad_()
    sigma = (3e-3 * torch.rand(shape)).requires_grad_()
    nt = 70
    source = DeepGPR.wavelet.ricker(3e8, nt, 1.5e-11, 4e-9).reshape(1, nt, 1).repeat(3, 1, 1)
    source.requires_grad_()
    z = 3 if three_d else 0
    src = torch.tensor([[[3, 4, z], [3, 4, z], [5, 6, z]], [[4, 5, z], [6, 3, z], [2, 2, z]]])
    rec = torch.tensor([[[6, 7, z], [6, 7, z]], [[2, 8, z], [7, 2, z]]])
    src[..., 0].remainder_(shape[0])
    rec[..., 0].remainder_(shape[0])
    out = DeepGPR.compute(device="cpu", dx=0.02, dt=1.5e-11, source_amplitudes=source,
                          source_location=src, receiver_location=rec, eps_r=eps_r, sigma=sigma,
                          pmlthick=pml,
                          fdtd_order=order, mode=3 if three_d else 2,
                          source_direction=0 if three_d else 2,
                          receiver_component=1 if three_d else 2)
    (out[-1].square().sum() + sum(t.sum() for t in out[1])).backward()
    key = f"{shape}-{order}"
    results[key] = [out[-1].detach(), eps_r.grad, sigma.grad, source.grad]
torch.save(results, sys.argv[1])
"""


class CpuDeterminismTests(unittest.TestCase):
    def test_forward_and_gradients_are_bitwise_identical_across_thread_counts(self):
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import DeepGPR
        from DeepGPR.native import loader

        lib = DeepGPR.get_deepgpr_lib("cpu")
        if not loader.library_supports(lib, "deepgpr_deterministic_adjoint"):
            self.skipTest("CPU library predates the deterministic adjoint; rebuild it.")

        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for threads in (1, 2, 5, 8):
                path = Path(directory) / f"t{threads}.pt"
                env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
                subprocess.run([sys.executable, "-c", WORKER, str(path), str(REPO_ROOT / "src")],
                               check=True, env=env)
                outputs.append(torch.load(path))
        for other in outputs[1:]:
            for key, tensors in outputs[0].items():
                for index, (a, b) in enumerate(zip(tensors, other[key])):
                    with self.subTest(case=key, output=index):
                        self.assertTrue(torch.equal(a, b))
                        self.assertTrue(torch.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()
