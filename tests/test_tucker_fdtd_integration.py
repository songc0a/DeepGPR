"""Test C: Tucker model + Tucker-sketched history inside DeepGPR's FDTD/FWI.

Cases are reduced versions of the repository's 2D FWI example
(``examples/2DFWImodel.npy``) and 3D FWI example (OverThrust section); see
``tests/tucker_cases.py``.
"""

import gc
import math
import sys
import unittest
import warnings
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import verification_utils as vu

vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR  # noqa: E402
import tucker_cases as tc  # noqa: E402
from DeepGPR.solver.sketched import WavefieldSketchStats  # noqa: E402
from DeepGPR.tucker import (  # noqa: E402
    TuckerMaterialModel,
    TuckerModelConfig,
    TuckerSketchConfig,
    cosine_similarity,
    relative_error,
    tucker_project,
    tucker_reconstruct,
)

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def setUpModule():
    # Test discovery imports every test module first and each one re-imports
    # DeepGPR. Bind this module's names to the instance that is current when
    # its tests run (the one tucker_cases calls), so the isinstance checks in
    # DeepGPR.compute see the same TuckerSketchConfig class.
    current = sys.modules["DeepGPR"]
    globals().update(
        DeepGPR=current,
        WavefieldSketchStats=current.solver.sketched.WavefieldSketchStats,
        **{
            name: getattr(current.tucker, name)
            for name in (
                "TuckerMaterialModel",
                "TuckerModelConfig",
                "TuckerSketchConfig",
                "cosine_similarity",
                "relative_error",
                "tucker_project",
                "tucker_reconstruct",
            )
        },
    )


def _bitwise_equal(a, b):
    return a.shape == b.shape and bool(torch.equal(a, b))


class _CaseMixin:
    @classmethod
    def setUpClass(cls):
        cls.case2d = tc.case_2d(DEVICE)
        cls.obs2d = tc.observed(cls.case2d)
        cls.ref2d = tc.l2_gradients(cls.case2d, cls.obs2d)
        cls.ref2d.pop("e_saved")
        cls.case3d = tc.case_3d(DEVICE)
        cls.obs3d = tc.observed(cls.case3d)
        cls.ref3d = tc.l2_gradients(cls.case3d, cls.obs3d)
        cls.ref3d.pop("e_saved")
        repeat = tc.l2_gradients(cls.case3d, cls.obs3d)
        # Run-to-run floor of the 3D CUDA adjoint (atomic scatter).
        cls.floor3d = {
            key: relative_error(repeat[f"grad_{key}"], cls.ref3d[f"grad_{key}"]) for key in ("eps", "sigma")
        }
        del repeat
        gc.collect()

    def _gradient_errors(self, result, reference):
        return {
            key: (
                relative_error(result[f"grad_{key}"], reference[f"grad_{key}"]),
                cosine_similarity(result[f"grad_{key}"], reference[f"grad_{key}"]),
            )
            for key in ("eps", "sigma")
        }


class DefaultPathTests(_CaseMixin, unittest.TestCase):
    def test_disabled_sketch_is_the_original_path(self):
        for case, obs, ref in ((self.case2d, self.obs2d, self.ref2d), (self.case3d, self.obs3d, self.ref3d)):
            for extra in (dict(wavefield_sketch=None), dict(wavefield_sketch=TuckerSketchConfig(enabled=False))):
                with self.subTest(case=case.name, extra=extra):
                    out = tc.l2_gradients(case, obs, **extra)
                    self.assertGreater(out["e_saved"].numel(), 0)
                    self.assertTrue(_bitwise_equal(out["data"], ref["data"]))
                    if case is self.case2d or not CUDA:
                        self.assertTrue(_bitwise_equal(out["grad_eps"], ref["grad_eps"]))
                        self.assertTrue(_bitwise_equal(out["grad_sigma"], ref["grad_sigma"]))
                    else:
                        errors = self._gradient_errors(out, ref)
                        self.assertLess(errors["eps"][0], 10 * self.floor3d["eps"] + 1e-6)
                        self.assertLess(errors["sigma"][0], 10 * self.floor3d["sigma"] + 1e-5)

    def test_lossless_sketch_reproduces_the_original_gradient(self):
        # Ranks >= frame size store raw frames through the segmented path:
        # this isolates segmentation and time-index alignment.
        for interval in (1, 3):
            for case, obs in ((self.case2d, self.obs2d), (self.case3d, self.obs3d)):
                with self.subTest(case=case.name, interval=interval):
                    ref = tc.l2_gradients(case, obs, model_gradient_sampling_interval=interval)
                    for rhs in ("stored", "from_e") if interval == 1 else ("stored",):
                        stats = WavefieldSketchStats()
                        out = tc.l2_gradients(
                            case,
                            obs,
                            model_gradient_sampling_interval=interval,
                            wavefield_sketch=TuckerSketchConfig(ranks=10_000, segment_steps=7, rhs=rhs),
                            wavefield_sketch_stats=stats,
                        )
                        self.assertEqual(out["e_saved"].numel(), 0)
                        self.assertGreater(stats.segments, 10)
                        self.assertTrue(_bitwise_equal(out["data"], ref["data"]))
                        if case is self.case2d or not CUDA:
                            self.assertTrue(_bitwise_equal(out["grad_eps"], ref["grad_eps"]), rhs)
                            self.assertTrue(_bitwise_equal(out["grad_sigma"], ref["grad_sigma"]), rhs)
                        else:
                            errors = self._gradient_errors(out, ref)
                            self.assertLess(errors["eps"][0], 1e-5)
                            self.assertLess(errors["sigma"][0], 1e-4)


class SketchAccuracyTests(_CaseMixin, unittest.TestCase):
    # Thresholds are ~3x the values measured on an RTX 4090 (float32, no TF32).
    # Built at run time (see setUpModule).
    @staticmethod
    def cases_2d():
        return [
        # (config, max eps rel. error, max sigma rel. error, min cosine)
        (TuckerSketchConfig(ranks=10, oversampling=6, measure_error=True), 3e-2, 3e-2, 0.9998),
        (TuckerSketchConfig(ranks=16, oversampling=6, measure_error=True), 6e-3, 3e-3, 0.99999),
        (TuckerSketchConfig(ranks=16, oversampling=6, rhs="from_e"), 4e-3, 3e-3, 0.99999),
        (TuckerSketchConfig(ranks=16, oversampling=6, time_block=8, time_rank=4), 1.5e-2, 6e-3, 0.9999),
    ]

    @staticmethod
    def cases_3d():
        return [
        (TuckerSketchConfig(ranks=6, oversampling=6, measure_error=True), 3e-3, 7e-3, 0.99998),
        (TuckerSketchConfig(ranks=10, oversampling=6, measure_error=True), 1e-3, 2e-3, 0.999998),
        (TuckerSketchConfig(ranks=10, oversampling=6, rhs="from_e"), 1e-3, 2e-3, 0.999998),
        (TuckerSketchConfig(ranks=10, oversampling=6, time_block=8, time_rank=4, segment_steps=5), 2e-3, 2e-3, 0.99999),
    ]

    def _check(self, case, obs, ref, configs):
        for config, eps_max, sigma_max, cos_min in configs:
            with self.subTest(case=case.name, config=config):
                stats = WavefieldSketchStats()
                out = tc.l2_gradients(case, obs, wavefield_sketch=config, wavefield_sketch_stats=stats)
                self.assertTrue(_bitwise_equal(out["data"], ref["data"]))  # forward unchanged
                self.assertEqual(out["loss"], ref["loss"])
                errors = self._gradient_errors(out, ref)
                self.assertLess(errors["eps"][0], eps_max)
                self.assertLess(errors["sigma"][0], sigma_max)
                self.assertGreater(errors["eps"][1], cos_min)
                self.assertGreater(errors["sigma"][1], cos_min)
                self.assertGreater(stats.size.storage_ratio, 1.5)
                self.assertLess(stats.size.stored_bytes, stats.size.raw_bytes)
                if config.measure_error:
                    for name, error in stats.errors.items():
                        self.assertTrue(math.isfinite(error.relative_l2), name)
                        self.assertLess(error.relative_l2, 0.05, name)
                        self.assertEqual(error.frames, stats.frames * case.kwargs["source_location"].shape[0])

    def test_2d_example_gradient_accuracy(self):
        self._check(self.case2d, self.obs2d, self.ref2d, self.cases_2d())

    def test_3d_example_gradient_accuracy(self):
        self._check(self.case3d, self.obs3d, self.ref3d, self.cases_3d())

    def test_error_decreases_with_rank(self):
        for case, obs, ref, ranks in ((self.case2d, self.obs2d, self.ref2d, (6, 10, 16)), (self.case3d, self.obs3d, self.ref3d, (4, 6, 10))):
            errors = []
            for rank in ranks:
                out = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=rank, oversampling=6))
                errors.append(self._gradient_errors(out, ref)["eps"][0])
            self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])), (case.name, errors))


class FourModeTests(_CaseMixin, unittest.TestCase):
    """Original / model Tucker only / sketch only / both."""

    MODEL_RANKS = {"2d_fwi_example": ((12, 8), (8, 6)), "3d_overthrust_example": ((3, 8, 10), (2, 6, 8))}
    @staticmethod
    def sketch_for(name):
        ranks = {"2d_fwi_example": 16, "3d_overthrust_example": 10}[name]
        return TuckerSketchConfig(ranks=ranks, oversampling=6)

    def _model(self, case, tucker=True):
        # Low-rank approximations undershoot eps = 1 in the air layer, and
        # voxel Adam steps can too: both fields use smooth lower bounds
        # (eps > 1, sigma > 0), the only difference being Tucker on/off.
        eps_ranks, sigma_ranks = self.MODEL_RANKS[case.name]
        return TuckerMaterialModel.from_models(
            case.eps_init,
            case.sigma_init,
            TuckerModelConfig(enabled=tucker, ranks=eps_ranks, mapping="softplus", bounds=(1.0, None)),
            TuckerModelConfig(enabled=tucker, ranks=sigma_ranks, mapping="softplus", bounds=(0.0, None), scale=1e-3),
        )

    def _loss(self, case, obs, eps, sigma, sketch):
        data = case.compute(eps, sigma, wavefield_sketch=sketch)[-1]
        return 0.5 * (data - obs).square().sum() / obs.square().sum()

    def test_tucker_gradients_through_exact_and_sketched_adjoint(self):
        for case, obs in ((self.case2d, self.obs2d), (self.case3d, self.obs3d)):
            grads = {}
            for label, config in (("exact", None), ("sketch", self.sketch_for(case.name))):
                model = self._model(case)
                tucker = model.epsilon.tucker
                unconstrained = tucker()
                unconstrained.retain_grad()
                eps = model.epsilon.mapping(unconstrained)
                self.assertGreaterEqual(float(eps.min()), 1.0)
                self._loss(case, obs, eps, model.sigma(), config).backward()
                # Chain rule through the FDTD adjoint: dL/dG = dL/dT ×_n U_n^T.
                with torch.no_grad():
                    expected_core = tucker_project(unconstrained.grad, tucker.factors())
                self.assertLess(relative_error(tucker.core_param.grad, expected_core), 2e-5)
                grads[label] = [p.grad.detach().clone() for p in model.parameters()]
                self.assertTrue(all(g is not None and float(g.norm()) > 0 for g in grads[label]))
            flat_exact = torch.cat([g.reshape(-1) for g in grads["exact"]])
            flat_sketch = torch.cat([g.reshape(-1) for g in grads["sketch"]])
            with self.subTest(case=case.name):
                self.assertGreater(cosine_similarity(flat_sketch, flat_exact), 0.9999)
                self.assertLess(relative_error(flat_sketch, flat_exact), 2e-2)

    def test_external_factor_tensors_through_fdtd(self):
        # Factors computed by external differentiable code (no network inside
        # DeepGPR); gradients reach the upstream leaf tensors through FDTD.
        case, obs = self.case2d, self.obs2d
        nx, ny = case.eps_init.shape
        base = case.eps_init.detach()
        u, s, v = torch.linalg.svd(base)
        rank = 10
        theta_x = (u[:, :rank] * s[:rank].sqrt()).clone().requires_grad_(True)
        theta_y = (v[:rank].T * s[:rank].sqrt()).clone().requires_grad_(True)
        core = torch.eye(rank, device=DEVICE, requires_grad=True)
        for sketch in (None, TuckerSketchConfig(ranks=16, oversampling=6)):
            for tensor in (theta_x, theta_y, core):
                tensor.grad = None
            ux = 1.0 * theta_x  # stand-ins for network outputs
            uy = theta_y.clone()
            eps = tucker_reconstruct(core, [ux, uy])
            self._loss(case, obs, eps, case.sigma_init, sketch).backward()
            with self.subTest(sketch=sketch):
                for tensor in (theta_x, theta_y, core):
                    self.assertIsNotNone(tensor.grad)
                    self.assertGreater(float(tensor.grad.norm()), 0.0)

    def test_four_modes_reduce_the_loss(self):
        for case, obs in ((self.case2d, self.obs2d), (self.case3d, self.obs3d)):
            for use_model in (False, True):
                for use_sketch in (False, True):
                    with self.subTest(case=case.name, model=use_model, sketch=use_sketch):
                        model = self._model(case, tucker=use_model)
                        # Adam moves every factor entry by ~lr, and each entry
                        # touches a whole fibre of the model: Tucker needs a
                        # ~10-20x smaller learning rate than voxels (measured).
                        lr = 0.001 if use_model else 0.02
                        optimizer = torch.optim.Adam(model.parameter_groups(lr, lr))
                        losses = []
                        for _ in range(4):
                            optimizer.zero_grad()
                            eps, sigma = model()
                            loss = self._loss(case, obs, eps, sigma, self.sketch_for(case.name) if use_sketch else None)
                            loss.backward()
                            optimizer.step()
                            losses.append(float(loss))
                        self.assertLess(losses[-1], losses[0])


class CombinationAndValidationTests(_CaseMixin, unittest.TestCase):
    def _checkpointed(self, case, obs, sketch, segment_steps=90):
        eps = case.eps_init.detach().clone().requires_grad_(True)
        sigma = case.sigma_init.detach().clone().requires_grad_(True)
        source = case.kwargs["source_amplitudes"]
        common = {k: v for k, v in case.kwargs.items() if k != "source_amplitudes"}
        state = ()
        pieces = []
        for start in range(0, source.shape[1], segment_steps):
            chunk = source[:, start : start + segment_steps].contiguous()

            def advance(er, se, src, *boundary):
                work = tuple(t.clone() for t in boundary)
                result = DeepGPR.compute(
                    eps_r=er, sigma=se, source_amplitudes=src,
                    E=work[:3] or None, H=work[3:6] or None, PML=work[6:] or None,
                    wavefield_sketch=sketch, **common,
                )
                return (*result[1], *result[2], *result[3], result[-1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = checkpoint(advance, eps, sigma, chunk, *state, preserve_rng_state=False)
            state = outputs[:30]
            pieces.append(outputs[30])
        data = torch.cat(pieces, dim=1)
        loss = 0.5 * (data - obs).square().sum() / obs.square().sum()
        loss.backward()
        return {"data": data.detach(), "grad_eps": eps.grad, "grad_sigma": sigma.grad}

    def test_checkpoint_combination(self):
        case, obs, ref = self.case2d, self.obs2d, self.ref2d
        checkpointed = self._checkpointed(case, obs, None)
        lossless = self._checkpointed(case, obs, TuckerSketchConfig(ranks=10_000))
        self.assertTrue(_bitwise_equal(lossless["data"], checkpointed["data"]))
        self.assertTrue(_bitwise_equal(lossless["grad_eps"], checkpointed["grad_eps"]))
        sketch = TuckerSketchConfig(ranks=16, oversampling=6)
        both = self._checkpointed(case, obs, sketch)
        alone = tc.l2_gradients(case, obs, wavefield_sketch=sketch)
        for key in ("grad_eps", "grad_sigma"):
            # Boundary states stay exact; only the history inside each
            # recomputed segment is compressed, frame by frame, as without
            # checkpointing.
            self.assertLess(relative_error(both[key], alone[key]), 1e-5)
            self.assertLess(relative_error(both[key], ref[key]), 6e-3)

    def test_source_and_state_gradients_are_exact(self):
        case, obs = self.case2d, self.obs2d
        results = []
        for sketch in (None, TuckerSketchConfig(ranks=4, oversampling=2, segment_steps=11)):
            source = case.kwargs["source_amplitudes"].detach().clone().requires_grad_(True)
            electric, magnetic, phi = DeepGPR.checkpoint_initial_field(
                device=DEVICE, dx=case.kwargs["dx"], dt=case.kwargs["dt"], source_amplitudes=source.detach(),
                source_location=case.kwargs["source_location"], receiver_location=case.kwargs["receiver_location"],
                er=case.eps_init, se=case.sigma_init, pmlthick=case.kwargs["pmlthick"],
            )
            hx = magnetic[0].clone().requires_grad_(True)
            options = {k: v for k, v in case.kwargs.items() if k != "source_amplitudes"}
            out = DeepGPR.compute(
                eps_r=case.eps_init, sigma=case.sigma_init, source_amplitudes=source,
                H=(hx, magnetic[1], magnetic[2]), wavefield_sketch=sketch, **options,
            )
            (0.5 * (out[-1] - obs).square().sum()).backward()
            results.append((source.grad, hx.grad))
        self.assertLess(relative_error(results[1][0], results[0][0]), 1e-6)
        self.assertLess(relative_error(results[1][1], results[0][1]), 1e-6)

    def test_multiple_sources_float64_sketch_and_physical_region(self):
        case, obs = self.case2d, self.obs2d
        nshots = case.kwargs["source_location"].shape[0]
        # Two simultaneous sources per shot with independent waveforms.
        locations = case.kwargs["source_location"].repeat(1, 2, 1).clone()
        locations[:, 1, 0] = (locations[:, 1, 0] + 15) % case.eps_init.shape[0]
        waveform = case.kwargs["source_amplitudes"]
        waveforms = torch.cat([waveform, -0.5 * waveform.roll(20, dims=1)], dim=0)
        multi = dict(source_location=locations, source_amplitudes=waveforms)
        ref = tc.l2_gradients(case, obs, **multi)
        lossless = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=10_000, segment_steps=9), **multi)
        self.assertTrue(_bitwise_equal(lossless["data"], ref["data"]))
        self.assertTrue(_bitwise_equal(lossless["grad_eps"], ref["grad_eps"]))
        float32 = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=16, oversampling=6), **multi)
        float64 = tc.l2_gradients(
            case, obs, wavefield_sketch=TuckerSketchConfig(ranks=16, oversampling=6, compute_dtype=torch.float64), **multi
        )
        # At rank 16 the truncation error dominates, not the arithmetic
        # precision; float64 also draws different Gaussian maps.
        e32 = relative_error(float32["grad_eps"], ref["grad_eps"])
        e64 = relative_error(float64["grad_eps"], ref["grad_eps"])
        self.assertLess(e32, 1.5e-2)
        self.assertLess(e64, 1.5e-2)
        self.assertEqual(nshots, ref["data"].shape[0])
        # Physical-region history (allowed for material gradients without PML).
        no_pml = dict(pmlthick=0, wavefield_history_region="physical")
        ref = tc.l2_gradients(case, obs, **no_pml)
        out = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=10_000, segment_steps=13), **no_pml)
        self.assertTrue(_bitwise_equal(out["grad_eps"], ref["grad_eps"]))
        self.assertTrue(_bitwise_equal(out["grad_sigma"], ref["grad_sigma"]))

    def test_no_grad_forward_and_double_backward(self):
        case = self.case2d
        with torch.no_grad():
            stats = WavefieldSketchStats()
            out = case.compute(case.eps_init, case.sigma_init, wavefield_sketch=TuckerSketchConfig(), wavefield_sketch_stats=stats)
        reference = case.compute(case.eps_init, case.sigma_init, save_wavefield_history=False)
        self.assertTrue(_bitwise_equal(out[-1], reference[-1]))
        self.assertEqual(out[0].numel(), 0)
        self.assertEqual(stats.segments, 1)
        self.assertEqual(stats.frames, 0)
        eps = case.eps_init.clone().requires_grad_(True)
        data = case.compute(eps, case.sigma_init, wavefield_sketch=TuckerSketchConfig(ranks=8))[-1]
        data.square().sum().backward(retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, "already released"):
            data.square().sum().backward()

    def test_invalid_combinations_are_rejected(self):
        case = self.case2d
        eps = case.eps_init.clone().requires_grad_(True)
        sketch = TuckerSketchConfig(ranks=8)
        if CUDA:
            with self.assertRaisesRegex(ValueError, "wavefield_compression"):
                case.compute(eps, case.sigma_init, wavefield_sketch=sketch, wavefield_compression="int8")
            with self.assertRaisesRegex(ValueError, "use_async_offload"):
                case.compute(eps, case.sigma_init, wavefield_sketch=sketch, use_async_offload=True)
        with self.assertRaisesRegex(ValueError, "float32"):
            case.compute(eps, case.sigma_init, wavefield_sketch=sketch, wavefield_storage_dtype="fp16")
        with self.assertRaisesRegex(ValueError, "R\\^n explicitly"):
            case.compute(eps, case.sigma_init, wavefield_sketch=sketch, wavefield_rhs_history="reconstructed")
        with self.assertRaisesRegex(ValueError, "save_forward_wavefield_path"):
            case.compute(eps, case.sigma_init, wavefield_sketch=sketch, save_forward_wavefield_path="/tmp/x")
        with self.assertRaisesRegex(ValueError, "sampling_interval=1"):
            case.compute(
                eps, case.sigma_init, wavefield_sketch=TuckerSketchConfig(rhs="from_e"),
                model_gradient_sampling_interval=2,
            )
        with self.assertRaises(TypeError):
            case.compute(eps, case.sigma_init, wavefield_sketch={"ranks": 8})
        with self.assertRaisesRegex(ValueError, "requires an enabled"):
            case.compute(eps, case.sigma_init, wavefield_sketch_stats=WavefieldSketchStats())

    @unittest.skipUnless(CUDA, "CUDA is not available")
    def test_measured_peak_memory_is_lower_than_the_default_history(self):
        # Peak allocated CUDA memory of forward + backward, measured, against
        # DeepGPR's default float32 history (E only, R rebuilt).
        case, obs = self.case3d, self.obs3d

        def peak(**extra):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = tc.l2_gradients(case, obs, **extra)
            del out
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base

        default = peak()
        stats = WavefieldSketchStats()
        sketched = peak(wavefield_sketch=TuckerSketchConfig(ranks=6, oversampling=6, segment_steps=8), wavefield_sketch_stats=stats)
        self.assertLess(sketched, 0.5 * default)
        # The measured peak covers at least the compressed history and the segment buffer.
        self.assertGreater(sketched, stats.size.stored_bytes + stats.segment_buffer_bytes)


class CpuBackendTests(unittest.TestCase):
    """The CPU/OpenMP native backend through the segmented path (deterministic)."""

    def test_cpu_lossless_bitwise_and_lossy_accuracy(self):
        cpu = torch.device("cpu")
        cases = (tc.case_2d(cpu, nt=200, stride=4), tc.case_3d(cpu, nt=120, extrude=8, stride=16))
        for case in cases:
            with self.subTest(case=case.name):
                obs = tc.observed(case)
                ref = tc.l2_gradients(case, obs)
                lossless = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=10_000, segment_steps=11))
                self.assertTrue(_bitwise_equal(lossless["data"], ref["data"]))
                self.assertTrue(_bitwise_equal(lossless["grad_eps"], ref["grad_eps"]))
                self.assertTrue(_bitwise_equal(lossless["grad_sigma"], ref["grad_sigma"]))
                lossy = tc.l2_gradients(case, obs, wavefield_sketch=TuckerSketchConfig(ranks=6, oversampling=4))
                self.assertGreater(cosine_similarity(lossy["grad_eps"], ref["grad_eps"]), 0.999)
                self.assertGreater(cosine_similarity(lossy["grad_sigma"], ref["grad_sigma"]), 0.999)


if __name__ == "__main__":
    unittest.main()
