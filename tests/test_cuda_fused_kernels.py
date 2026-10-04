"""CUDA kernel fusion: CPML corrections inside the field-update kernels, the E
history write inside ``update_h_gpu`` and the CPML transposes inside the
scatter adjoint. Forward results must stay bitwise identical to the separate
kernels they replace."""

import sys
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

STORAGE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _case(mode, nt):
    """2D (two shots) or 3D problem with asymmetric CPML on every face."""
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
    device = torch.device("cuda")
    # Early peak so that the CPML cells carry non-zero fields within a few steps.
    source = DeepGPR.ricker(4.0e8, nt, dt, 5.0e-10).reshape(1, nt, 1)
    common = dict(
        device=device,
        dx=spacing,
        dt=dt,
        source_location=source_location.to(device),
        receiver_location=receiver_location.to(device),
        pmlthick=pml,
        fdtd_order=4,
        mode=mode,
    )
    return eps_r.to(device), sigma.to(device), source.to(device), common


def _history(mode, nt, **options):
    """Return (E_saved, E fields, receiver data) of a gradient-enabled run."""
    eps_r, sigma, source, common = _case(mode, nt)
    eps_r = eps_r.clone().requires_grad_(True)
    result = DeepGPR.compute(eps_r=eps_r, sigma=sigma, source_amplitudes=source, **common, **options)
    node = result[-1].grad_fn
    return node.E_saved.detach().cpu(), [t.detach().cpu() for t in result[1]], result[-1].detach().cpu()


def _fields_after(mode, steps):
    """E fields after ``steps`` time steps (forward only, no history)."""
    eps_r, sigma, source, common = _case(mode, steps)
    result = DeepGPR.compute(
        eps_r=eps_r, sigma=sigma, source_amplitudes=source, save_wavefield_history=False, **common
    )
    return [t.detach().cpu() for t in result[1]]


def _history_frame(fields, mode, origin, extent):
    """Crop E fields (one padding cell per axis) to a history box, E_saved layout."""
    (x0, y0, z0), (hx, hy, hz) = origin, extent
    crops = [f[:, x0 : x0 + hx, y0 : y0 + hy, z0 : z0 + hz] for f in fields]
    return torch.stack(crops) if mode == 3 else crops[2]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaFusedHistoryTests(unittest.TestCase):
    def test_saved_frames_are_the_fields_after_each_step(self):
        nt = 30
        for mode in (2, 3):
            _, _, _, common = _case(mode, nt)
            reference = {steps: _fields_after(mode, steps) for steps in range(1, nt)}
            for region in ("extended", "physical"):
                for offload in (False, True):
                    for interval in (1, 3):
                        with warnings.catch_warnings():
                            # interval 3 exceeds the spectral bound of this short source
                            warnings.simplefilter("ignore", RuntimeWarning)
                            history, _, _ = _history(
                                mode, nt, wavefield_history_region=region,
                                use_async_offload=offload, model_gradient_sampling_interval=interval,
                            )
                        nx, ny, nz = (f - 1 for f in reference[1][0].shape[1:])
                        origin, extent = DeepGPR.solver.storage.history_box(
                            nx, ny, nz, common["pmlthick"], region
                        )
                        frames = history if mode == 2 else history.transpose(0, 1)
                        with self.subTest(mode=mode, region=region, offload=offload, interval=interval):
                            self.assertEqual(frames.shape[0], (nt + interval - 1) // interval)
                            self.assertEqual(int(torch.count_nonzero(frames[0])), 0)
                            if region == "extended":
                                # the low-x CPML slab is non-zero, so its updates are covered
                                x0 = common["pmlthick"][0]
                                self.assertGreater(int(torch.count_nonzero(frames[-1][..., :x0, :, :])), 0)
                            for index in range(1, frames.shape[0]):
                                expected = _history_frame(reference[index * interval], mode, origin, extent)
                                self.assertTrue(torch.equal(frames[index], expected), index)

    def test_low_precision_frames_match_the_separate_writer(self):
        # native_scalar histories are written by update_h_gpu; native_vec2
        # keeps the separate snapshot kernels. Both round to nearest even.
        for mode in (2, 3):
            for storage in ("fp16", "bf16"):
                for offload in (False, True):
                    runs = [
                        _history(
                            mode, 24, wavefield_storage_dtype=STORAGE[storage],
                            wavefield_conversion_backend=backend, use_async_offload=offload,
                        )
                        for backend in ("native_scalar", "native_vec2")
                    ]
                    with self.subTest(mode=mode, storage=storage, offload=offload):
                        self.assertTrue(torch.equal(runs[0][0], runs[1][0]))
                        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))

    def test_forward_does_not_depend_on_the_history_writer(self):
        for mode in (2, 3):
            options = (
                dict(wavefield_storage_dtype=torch.float32),
                dict(wavefield_storage_dtype=torch.float16, use_async_offload=True),
                dict(wavefield_compression="int8"),
                dict(wavefield_history_region="physical", wavefield_rhs_history="stored"),
            )
            runs = [_history(mode, 24, **option) for option in options]
            reference = _fields_after(mode, 24)
            for option, (_, fields, data) in zip(options, runs):
                with self.subTest(mode=mode, option=option):
                    self.assertTrue(torch.equal(data, runs[0][2]))
                    for field, expected in zip(fields, reference):
                        self.assertTrue(torch.equal(field, expected))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaFusedCpmlStateTests(unittest.TestCase):
    def test_segmented_run_reproduces_the_full_run_bitwise(self):
        nt, split = 30, 13
        for mode in (2, 3):
            eps_r, sigma, source, common = _case(mode, nt)
            full = DeepGPR.compute(
                eps_r=eps_r, sigma=sigma, source_amplitudes=source, save_wavefield_history=False, **common
            )
            first = DeepGPR.compute(
                eps_r=eps_r, sigma=sigma, source_amplitudes=source[:, :split],
                save_wavefield_history=False, **common
            )
            second = DeepGPR.compute(
                eps_r=eps_r, sigma=sigma, source_amplitudes=source[:, split:],
                E=tuple(t.clone() for t in first[1]), H=tuple(t.clone() for t in first[2]),
                PML=tuple(t.clone() for t in first[3]), save_wavefield_history=False, **common
            )
            with self.subTest(mode=mode):
                data = torch.cat((first[-1], second[-1]), dim=1)
                self.assertTrue(torch.equal(data, full[-1]))
                for part, whole in zip((*second[1], *second[2], *second[3]), (*full[1], *full[2], *full[3])):
                    self.assertTrue(torch.equal(part, whole))


if __name__ == "__main__":
    with warnings.catch_warnings():
        unittest.main(verbosity=2)
