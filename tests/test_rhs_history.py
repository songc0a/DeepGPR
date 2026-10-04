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
from DeepGPR.native.bridge import require_capability
from DeepGPR.solver.storage import resolve_rhs_reconstruction
from DeepGPR.utils.exceptions import NativeLibraryError

STORAGE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _case(device, mode, nt=60):
    """Small heterogeneous 2D (mode 2, two shots) or 3D (mode 3) problem."""
    device = torch.device(device)
    if mode == 2:
        nx, ny = 18, 21
        x = torch.linspace(0.0, 1.0, nx)[:, None]
        y = torch.linspace(0.0, 1.0, ny)[None, :]
        eps_r = 4.0 + 0.3 * x + 0.2 * y
        sigma = 2.0e-4 + 1.0e-4 * x + 0.0 * y
        source_location = torch.tensor([[[8, 7, 0]], [[10, 9, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor(
            [[[7, 12, 0], [9, 14, 0]], [[6, 13, 0], [11, 15, 0]]], dtype=torch.int32
        )
        spacing, dt, pml = (0.020, 0.016, 0.012), 2.5e-11, 3
    else:
        shape = (12, 13, 11)
        x = torch.linspace(0.0, 1.0, shape[0])[:, None, None]
        eps_r = 4.0 + 0.3 * x + torch.zeros(shape)
        sigma = 2.0e-4 + 1.0e-4 * x + torch.zeros(shape)
        source_location = torch.tensor([[[6, 6, 5]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[5, 8, 6], [7, 4, 5]]], dtype=torch.int32)
        spacing, dt, pml = (0.020, 0.018, 0.016), 1.5e-11, [2, 1, 2, 3, 1, 2]
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


def _run(device, mode, rhs, storage="fp32", compression="none", async_offload=False, mutate=None):
    eps_base, sigma_base, common = _case(device, mode)
    eps_r = eps_base.clone().requires_grad_(True)
    sigma = sigma_base.clone().requires_grad_(True)
    source = common.pop("source_amplitudes").clone().requires_grad_(True)
    result = DeepGPR.compute(
        eps_r=eps_r,
        sigma=sigma,
        source_amplitudes=source,
        wavefield_storage_dtype=STORAGE[storage],
        wavefield_compression=compression,
        use_async_offload=async_offload,
        wavefield_rhs_history=rhs,
        **common,
    )
    node = result[-1].grad_fn
    histories = {
        name: getattr(node, name)
        for name in ("E_saved", "R_saved", "E_final")
        if getattr(node, name) is not None
    }
    history_bytes = sum(t.numel() * t.element_size() for t in histories.values())
    data = result[-1]
    if mutate is not None:
        mutate(result)
    (data - 0.9 * data.detach()).square().sum().backward()
    return {
        "data": data.detach().cpu(),
        "history": result[0].detach().cpu(),
        "eps": eps_r.grad.detach().cpu(),
        "sigma": sigma.grad.detach().cpu(),
        "source": source.grad.detach().cpu(),
        "names": set(histories),
        "history_bytes": history_bytes,
    }


def _relative(candidate, reference):
    return float((candidate - reference).norm() / reference.norm().clamp_min(1.0e-30))


class RhsHistoryPolicyTests(unittest.TestCase):
    def test_resolution_rules(self):
        common = dict(stores_model_history=True, library_supported=True)
        self.assertTrue(resolve_rhs_reconstruction("auto", sampling_interval=1, **common))
        self.assertFalse(resolve_rhs_reconstruction("auto", sampling_interval=2, **common))
        self.assertFalse(
            resolve_rhs_reconstruction("auto", sampling_interval=1, lossless_storage=False, **common)
        )
        self.assertTrue(
            resolve_rhs_reconstruction(
                "reconstructed", sampling_interval=1, lossless_storage=False, **common
            )
        )
        self.assertFalse(resolve_rhs_reconstruction("stored", sampling_interval=1, **common))
        # A library without the probe silently keeps E+R under "auto" ...
        self.assertFalse(
            resolve_rhs_reconstruction(
                "auto", sampling_interval=1, stores_model_history=True, library_supported=False
            )
        )
        # ... and rejects an explicit request.
        with self.assertRaises(NativeLibraryError):
            resolve_rhs_reconstruction(
                "reconstructed", sampling_interval=1, stores_model_history=True,
                library_supported=False,
            )
        with self.assertRaisesRegex(ValueError, "model_gradient_sampling_interval=1"):
            resolve_rhs_reconstruction("reconstructed", sampling_interval=2, **common)
        # Nothing to rebuild without a material gradient.
        self.assertFalse(
            resolve_rhs_reconstruction(
                "reconstructed", sampling_interval=1, stores_model_history=False,
                library_supported=False,
            )
        )

    def test_stale_library_is_rejected_before_native_call(self):
        class StaleLibrary:
            pass

        with self.assertRaises(NativeLibraryError):
            require_capability(StaleLibrary(), "deepgpr_supports_rhs_reconstruction", "stale")

    def test_invalid_option_is_rejected(self):
        eps_r, sigma, common = _case("cpu", 2)
        with self.assertRaisesRegex(ValueError, "wavefield_rhs_history"):
            DeepGPR.compute(eps_r=eps_r, sigma=sigma, wavefield_rhs_history="none", **common)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "requires model_gradient_sampling_interval=1"):
                DeepGPR.compute(
                    eps_r=eps_r.clone().requires_grad_(True),
                    sigma=sigma,
                    wavefield_rhs_history="reconstructed",
                    model_gradient_sampling_interval=2,
                    **common,
                )

    def test_auto_selects_e_only_for_float32_and_keeps_low_precision_pairs(self):
        self.assertEqual(_run("cpu", 2, "auto")["names"], {"E_saved", "E_final"})
        self.assertEqual(_run("cpu", 2, "auto", "fp16")["names"], {"E_saved", "R_saved"})
        self.assertEqual(_run("cpu", 2, "stored")["names"], {"E_saved", "R_saved"})
        self.assertEqual(_run("cpu", 2, "reconstructed", "bf16")["names"], {"E_saved", "E_final"})

    def test_memory_estimate_counts_one_history_and_the_final_frame(self):
        common = dict(
            device=torch.device("cpu"), nx=20, ny=24, nz=1, nt=50, nstep=2, nsr=1, nrx=3,
            source_waveforms=1, pml=[3, 3, 3, 3, 0, 0], mode=2, sampling_interval=1,
            use_async_offload=False, er_requires_grad=True, se_requires_grad=False,
        )
        one_history = 50 * 2 * 20 * 24 * 4
        e_only = DeepGPR.estimate_compute_memory(storage_dtype=torch.float32, **common)
        pair = DeepGPR.estimate_compute_memory(
            storage_dtype=torch.float32, reconstruct_rhs=False, **common
        )
        half = DeepGPR.estimate_compute_memory(storage_dtype=torch.float16, **common)
        self.assertTrue(e_only["reconstruct_rhs"])
        self.assertEqual(e_only["final_e_frame"], 2 * 20 * 24 * 4)
        self.assertEqual(e_only["saved_gradient_wavefields"], one_history + 2 * 20 * 24 * 4)
        self.assertEqual(pair["saved_gradient_wavefields"], 2 * one_history)
        self.assertFalse(half["reconstruct_rhs"])
        self.assertGreater(half["low_precision_exact_snapshot"], 0)


class CpuRhsReconstructionTests(unittest.TestCase):
    def test_float32_gradients_are_bitwise_identical(self):
        for mode in (2, 3):
            stored = _run("cpu", mode, "stored")
            rebuilt = _run("cpu", mode, "reconstructed")
            with self.subTest(mode=mode):
                for key in ("data", "history", "eps", "sigma", "source"):
                    self.assertTrue(torch.equal(stored[key], rebuilt[key]), key)
                nt_saved = 60  # _case: nt = 60, sampling interval 1
                frame_bytes = stored["history_bytes"] // 2 // nt_saved
                self.assertEqual(
                    rebuilt["history_bytes"], stored["history_bytes"] // 2 + frame_bytes
                )

    def test_stored_rhs_equals_rebuilt_rhs_bitwise(self):
        eps_r, sigma, common = _case("cpu", 2)
        eps_r = eps_r.clone().requires_grad_(True)
        result = DeepGPR.compute(
            eps_r=eps_r, sigma=sigma, wavefield_rhs_history="stored", **common
        )
        node = result[-1].grad_fn
        e_saved, r_saved = node.E_saved, node.R_saved
        final = result[1][2][:, :-1, :-1, :-1]
        later = torch.cat((e_saved[1:], final.unsqueeze(0)), dim=0)
        # Rebuild ca/cb exactly as build_update_coeffs_cpu does.
        eps_pad = node.saved_tensors[15][:-1, :-1, :-1]
        sigma_pad = node.saved_tensors[16][:-1, :-1, :-1]
        e_term = torch.tensor(DeepGPR.config.constants.VACUUM_PERMITTIVITY, dtype=torch.float32)
        e_term = e_term * eps_pad / torch.tensor(common["dt"], dtype=torch.float32)
        s_term = 0.5 * sigma_pad
        ca = (e_term - s_term) / (e_term + s_term)
        cb = 1.0 / (e_term + s_term)
        rebuilt = (later - ca * e_saved) / cb
        self.assertTrue(torch.equal(rebuilt, r_saved))

    def test_final_frame_is_independent_of_returned_states(self):
        def advance_returned_states(result):
            with torch.no_grad():
                for tensor in (*result[1], *result[2]):
                    tensor.mul_(-3.0).add_(1.0)

        reference = _run("cpu", 2, "reconstructed")
        mutated = _run("cpu", 2, "reconstructed", mutate=advance_returned_states)
        for key in ("eps", "sigma", "source"):
            self.assertTrue(torch.equal(reference[key], mutated[key]), key)

    def test_low_precision_reconstruction_stays_close(self):
        reference = _run("cpu", 2, "stored")
        for storage in ("fp16", "bf16"):
            candidate = _run("cpu", 2, "reconstructed", storage)
            with self.subTest(storage=storage):
                self.assertLess(_relative(candidate["eps"], reference["eps"]), 1.0e-2)
                self.assertLess(_relative(candidate["sigma"], reference["sigma"]), 1.0e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaRhsReconstructionTests(unittest.TestCase):
    def test_float32_gradients_match_stored_rhs(self):
        for mode in (2, 3):
            for async_offload in (False, True):
                stored = _run("cuda", mode, "stored", async_offload=async_offload)
                rebuilt = _run("cuda", mode, "reconstructed", async_offload=async_offload)
                with self.subTest(mode=mode, async_offload=async_offload):
                    self.assertTrue(torch.equal(stored["data"], rebuilt["data"]))
                    self.assertTrue(torch.equal(stored["history"], rebuilt["history"]))
                    # Remaining differences come from the atomic adjoint scatter.
                    for key in ("eps", "sigma", "source"):
                        self.assertLess(_relative(rebuilt[key], stored[key]), 2.0e-6, key)
                    self.assertLess(rebuilt["history_bytes"], 0.6 * stored["history_bytes"])

    def test_low_precision_and_int8_reconstruction(self):
        reference = _run("cuda", 3, "stored")
        cases = (("fp16", "none"), ("bf16", "none"), ("fp32", "int8"))
        for storage, compression in cases:
            for async_offload in (False, True) if compression == "none" else (False,):
                candidate = _run(
                    "cuda", 3, "reconstructed", storage, compression, async_offload
                )
                with self.subTest(storage=storage, compression=compression, offload=async_offload):
                    self.assertTrue(torch.isfinite(candidate["eps"]).all())
                    self.assertLess(_relative(candidate["eps"], reference["eps"]), 5.0e-2)
                    self.assertLess(_relative(candidate["sigma"], reference["sigma"]), 5.0e-2)

    def test_cpu_and_cuda_reconstruction_agree(self):
        cpu = _run("cpu", 2, "reconstructed")
        cuda = _run("cuda", 2, "reconstructed")
        for key in ("data", "eps", "sigma"):
            torch.testing.assert_close(cuda[key], cpu[key], rtol=5.0e-4, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
