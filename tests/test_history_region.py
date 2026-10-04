import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import verification_utils as vu

vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR
from DeepGPR.native.bridge import require_capability
from DeepGPR.solver.storage import history_box
from DeepGPR.utils.exceptions import NativeLibraryError

STORAGE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _case(device, mode, nt=40):
    """2D (two shots, asymmetric CPML) or 3D (asymmetric CPML on all faces)."""
    device = torch.device(device)
    if mode == 2:
        nx, ny = 17, 19
        x = torch.linspace(0.0, 1.0, nx)[:, None]
        y = torch.linspace(0.0, 1.0, ny)[None, :]
        eps_r = 4.0 + 0.3 * x + 0.2 * y
        sigma = 2.0e-4 + 1.0e-4 * x + 0.0 * y
        source_location = torch.tensor([[[7, 6, 0]], [[9, 8, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor(
            [[[6, 11, 0], [8, 13, 0]], [[5, 12, 0], [10, 14, 0]]], dtype=torch.int32
        )
        spacing, dt, pml = (0.020, 0.016, 0.012), 2.5e-11, [3, 5, 4, 2, 0, 0]
    else:
        shape = (12, 13, 11)
        x = torch.linspace(0.0, 1.0, shape[0])[:, None, None]
        eps_r = 4.0 + 0.3 * x + torch.zeros(shape)
        sigma = 2.0e-4 + 1.0e-4 * x + torch.zeros(shape)
        source_location = torch.tensor([[[6, 6, 5]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[5, 8, 6], [7, 4, 5]]], dtype=torch.int32)
        spacing, dt, pml = (0.020, 0.018, 0.016), 1.5e-11, [2, 1, 3, 2, 1, 2]
    source = DeepGPR.ricker(4.0e8, nt, dt, 1.5e-9).reshape(1, nt, 1)
    common = dict(
        device=device,
        dx=spacing,
        dt=dt,
        source_amplitudes=source.to(device),
        source_location=source_location.to(device),
        receiver_location=receiver_location.to(device),
        pmlthick=pml,
        fdtd_order=4,
        mode=mode,
    )
    return eps_r.to(device), sigma.to(device), common


def _run(device, mode, region, storage="fp32", rhs="auto", compression="none", offload=False):
    eps_base, sigma_base, common = _case(device, mode)
    eps_r = eps_base.clone().requires_grad_(True)
    sigma = sigma_base.clone().requires_grad_(True)
    result = DeepGPR.compute(
        eps_r=eps_r,
        sigma=sigma,
        wavefield_storage_dtype=STORAGE[storage],
        wavefield_rhs_history=rhs,
        wavefield_compression=compression,
        use_async_offload=offload,
        wavefield_history_region=region,
        **common,
    )
    node = result[-1].grad_fn
    history_bytes = sum(
        t.numel() * t.element_size()
        for t in (node.E_saved, node.R_saved, node.E_final)
        if t is not None
    )
    data = result[-1]
    (data - 0.9 * data.detach()).square().sum().backward()
    return {
        "data": data.detach().cpu(),
        "history": result[0].detach().cpu(),
        "eps": eps_r.grad.detach().cpu(),
        "sigma": sigma.grad.detach().cpu(),
        "history_bytes": history_bytes,
        "pml": common["pmlthick"],
    }


def _crop(history, mode, pml):
    """Crop an extended E_saved to the physical cells."""
    x0, xm, y0, ym, z0, zm = pml
    nx, ny, nz = history.shape[-3:]
    return history[..., x0 : nx - xm, y0 : ny - ym, z0 : nz - zm]


def _relative(candidate, reference):
    return float((candidate - reference).norm() / reference.norm().clamp_min(1.0e-30))


class HistoryRegionContractTests(unittest.TestCase):
    def test_history_box_matches_tile_aligned_full_grid(self):
        pml = [3, 5, 4, 2, 0, 0]
        self.assertEqual(history_box(30, 26, 1, pml, "extended"), ((0, 0, 0), (30, 26, 1)))
        self.assertEqual(history_box(30, 26, 1, pml, "physical"), ((3, 4, 0), (22, 20, 1)))
        # INT8 8x8 tiles: origin rounded down, end rounded up to the full-grid
        # tiling and clamped to the grid.
        self.assertEqual(
            history_box(30, 26, 1, pml, "physical", (8, 8, 1)), ((0, 0, 0), (30, 24, 1))
        )
        self.assertEqual(
            history_box(100, 100, 100, [10] * 6, "physical", (4, 4, 4)),
            ((8, 8, 8), (84, 84, 84)),
        )

    def test_invalid_region_and_stale_library(self):
        eps_r, sigma, common = _case("cpu", 2)
        with self.assertRaisesRegex(ValueError, "wavefield_history_region"):
            DeepGPR.compute(eps_r=eps_r, sigma=sigma, wavefield_history_region="interior", **common)

        class StaleLibrary:
            pass

        with self.assertRaises(NativeLibraryError):
            require_capability(StaleLibrary(), "deepgpr_supports_physical_history", "stale")

    def test_memory_estimate_scales_with_physical_cells(self):
        common = dict(
            device=torch.device("cpu"), nx=30, ny=26, nz=1, nt=50, nstep=2, nsr=1, nrx=3,
            source_waveforms=1, pml=[3, 5, 4, 2, 0, 0], mode=2, sampling_interval=1,
            storage_dtype=torch.float32, use_async_offload=False,
            er_requires_grad=True, se_requires_grad=True,
        )
        extended = DeepGPR.estimate_compute_memory(**common)
        physical = DeepGPR.estimate_compute_memory(history_region="physical", **common)
        self.assertEqual(physical["history_shape"], (22, 20, 1))
        self.assertEqual(
            physical["saved_gradient_wavefields"] * 30 * 26,
            extended["saved_gradient_wavefields"] * 22 * 20,
        )

    def test_saved_file_records_the_physical_region(self):
        eps_r, sigma, common = _case("cpu", 2)
        with tempfile.TemporaryDirectory() as directory:
            result = DeepGPR.compute(
                eps_r=eps_r, sigma=sigma, wavefield_history_region="physical",
                save_forward_wavefield_path=directory, **common,
            )
            files = list(Path(directory).glob("forward_wavefield_*.pt"))
            self.assertEqual(len(files), 1)
            payload = torch.load(files[0])
        self.assertEqual(payload["history_region"], "physical")
        self.assertEqual(payload["history_origin"], (3, 4, 0))
        self.assertEqual(payload["pmlthick"], [3, 5, 4, 2, 0, 0])
        self.assertEqual(tuple(payload["uncompressed_shape"]), tuple(result[0].shape))
        self.assertTrue(torch.equal(payload["wavefield"], result[0]))


class CpuHistoryRegionTests(unittest.TestCase):
    def test_physical_history_gives_bitwise_identical_gradients(self):
        cases = (("fp32", "auto"), ("fp32", "stored"), ("fp16", "auto"), ("bf16", "reconstructed"))
        for mode in (2, 3):
            for storage, rhs in cases:
                extended = _run("cpu", mode, "extended", storage, rhs)
                physical = _run("cpu", mode, "physical", storage, rhs)
                with self.subTest(mode=mode, storage=storage, rhs=rhs):
                    for key in ("data", "eps", "sigma"):
                        self.assertTrue(torch.equal(extended[key], physical[key]), key)
                    cropped = _crop(extended["history"], mode, extended["pml"])
                    self.assertTrue(torch.equal(cropped, physical["history"]))
                    ratio = physical["history"].numel() / extended["history"].numel()
                    self.assertEqual(
                        physical["history_bytes"] * extended["history"].numel(),
                        extended["history_bytes"] * physical["history"].numel(),
                    )
                    self.assertLess(ratio, 0.75)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaHistoryRegionTests(unittest.TestCase):
    def test_physical_history_matches_extended(self):
        cases = (
            ("fp32", "auto", False),
            ("fp32", "stored", True),
            ("fp16", "auto", False),
            ("bf16", "reconstructed", True),
        )
        for mode in (2, 3):
            for storage, rhs, offload in cases:
                extended = _run("cuda", mode, "extended", storage, rhs, offload=offload)
                physical = _run("cuda", mode, "physical", storage, rhs, offload=offload)
                with self.subTest(mode=mode, storage=storage, rhs=rhs, offload=offload):
                    self.assertTrue(torch.equal(extended["data"], physical["data"]))
                    cropped = _crop(extended["history"], mode, extended["pml"])
                    self.assertTrue(torch.equal(cropped, physical["history"]))
                    # Remaining differences come from the atomic adjoint scatter.
                    for key in ("eps", "sigma"):
                        self.assertLess(_relative(physical[key], extended[key]), 2.0e-6, key)

    def test_int8_tiles_and_gradients_follow_the_full_grid_tiling(self):
        for mode in (2, 3):
            for rhs in ("stored", "reconstructed"):
                extended = _run("cuda", mode, "extended", rhs=rhs, compression="int8")
                physical = _run("cuda", mode, "physical", rhs=rhs, compression="int8")
                _, _, common = _case("cpu", mode)
                nx, ny, nz = (25, 25, 1) if mode == 2 else (15, 18, 14)
                block = (8, 8, 1) if mode == 2 else (4, 4, 4)
                origin, extent = history_box(nx, ny, nz, common["pmlthick"], "physical", block)
                nstep, nt_saved = extended["data"].shape[:2]
                extended_shape = DeepGPR.solver.storage.saved_history_shape(
                    mode, nt_saved, nstep, nx, ny, nz
                )
                physical_shape = DeepGPR.solver.storage.saved_history_shape(
                    mode, nt_saved, nstep, *extent
                )
                full = DeepGPR.decompress_wavefield_history(
                    extended["history"], extended_shape, block
                )
                part = DeepGPR.decompress_wavefield_history(
                    physical["history"], physical_shape, block
                )
                (x0, y0, z0), (hx, hy, hz) = origin, extent
                with self.subTest(mode=mode, rhs=rhs):
                    self.assertTrue(
                        torch.equal(full[..., x0 : x0 + hx, y0 : y0 + hy, z0 : z0 + hz], part)
                    )
                    self.assertLess(physical["history_bytes"], extended["history_bytes"])
                    for key in ("eps", "sigma"):
                        self.assertLess(_relative(physical[key], extended[key]), 2.0e-6, key)


if __name__ == "__main__":
    with warnings.catch_warnings():
        unittest.main(verbosity=2)
