"""Test A: Tucker model parameterization (DeepGPR.tucker.model)."""

import math
import sys
import unittest
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
import DeepGPR  # noqa: E402
from DeepGPR.tucker import (  # noqa: E402
    MaterialMapping,
    TuckerMaterialField,
    TuckerMaterialModel,
    TuckerModelConfig,
    TuckerTensor,
    hosvd,
    randomized_hosvd,
    relative_error,
    tucker_parameter_count,
    tucker_project,
    tucker_reconstruct,
)

CUDA = torch.cuda.is_available()


def _reference_reconstruct(core, factors):
    """Independent dense reference: explicit einsum over all modes."""
    letters = "abcd"[: core.dim()]
    outputs = "wxyz"[: core.dim()]
    operands = ",".join(f"{o}{l}" for o, l in zip(outputs, letters))
    return torch.einsum(f"{letters},{operands}->{outputs}", core, *factors)


def _random_tucker(shape, ranks, dtype=torch.float64, seed=0, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    core = torch.randn(ranks, generator=generator, dtype=dtype).to(device)
    factors = [torch.randn((s, r), generator=generator, dtype=dtype).to(device) for s, r in zip(shape, ranks)]
    return core, factors


class TuckerReconstructTests(unittest.TestCase):
    def test_2d_matches_dense_reference(self):
        core, factors = _random_tucker((23, 17), (4, 6))
        result = tucker_reconstruct(core, factors)
        self.assertEqual(result.shape, (23, 17))
        expected = factors[0] @ core @ factors[1].T
        self.assertLess(relative_error(result, expected), 1e-14)

    def test_3d_matches_dense_reference_for_many_rank_configurations(self):
        shape = (13, 11, 9)
        for ranks in [(1, 1, 1), (3, 5, 2), (13, 11, 9), (2, 11, 4), (7, 1, 9)]:
            with self.subTest(ranks=ranks):
                core, factors = _random_tucker(shape, ranks, seed=sum(ranks))
                result = tucker_reconstruct(core, factors)
                self.assertEqual(result.shape, shape)
                self.assertLess(relative_error(result, _reference_reconstruct(core, factors)), 1e-14)

    def test_axis_order_is_detected(self):
        # A transposed factor order must not reproduce the tensor.
        core, factors = _random_tucker((10, 10, 10), (3, 3, 3))
        result = tucker_reconstruct(core, factors)
        swapped = tucker_reconstruct(core, [factors[1], factors[0], factors[2]])
        self.assertGreater(relative_error(swapped, result), 0.1)

    def test_none_factor_is_identity(self):
        core, factors = _random_tucker((6, 5, 4), (3, 5, 2))
        expected = tucker_reconstruct(core, [factors[0], torch.eye(5, dtype=torch.float64), factors[2]])
        self.assertLess(relative_error(tucker_reconstruct(core, [factors[0], None, factors[2]]), expected), 1e-15)

    def test_dtypes_preserved(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-14)):
            with self.subTest(dtype=dtype):
                core, factors = _random_tucker((8, 7, 6), (2, 3, 4), dtype=dtype)
                result = tucker_reconstruct(core, factors)
                self.assertEqual(result.dtype, dtype)
                reference = _reference_reconstruct(core.double(), [f.double() for f in factors])
                self.assertLess(relative_error(result, reference), tolerance)

    def test_invalid_inputs_raise(self):
        core, factors = _random_tucker((8, 7, 6), (2, 3, 4))
        with self.assertRaisesRegex(ValueError, "one factor per core mode"):
            tucker_reconstruct(core, factors[:2])
        with self.assertRaisesRegex(ValueError, "columns"):
            tucker_reconstruct(core, [factors[0], factors[2], factors[1]])
        with self.assertRaisesRegex(TypeError, "dtype"):
            tucker_reconstruct(core, [factors[0].float(), factors[1], factors[2]])
        with self.assertRaises(TypeError):
            tucker_reconstruct(core, factors[0])

    @unittest.skipUnless(CUDA, "CUDA is not available")
    def test_cpu_gpu_consistency(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for dtype, tolerance in ((torch.float64, 1e-13), (torch.float32, 2e-6)):
                core, factors = _random_tucker((31, 29, 27), (5, 6, 7), dtype=dtype)
                cpu = tucker_reconstruct(core, factors)
                gpu = tucker_reconstruct(core.cuda(), [f.cuda() for f in factors])
                self.assertEqual(gpu.device.type, "cuda")
                self.assertLess(relative_error(gpu.cpu(), cpu), tolerance)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_gradcheck_core_and_factors(self):
        core, factors = _random_tucker((5, 4, 3), (2, 3, 2))
        inputs = [core.requires_grad_(True)] + [f.requires_grad_(True) for f in factors]
        self.assertTrue(
            torch.autograd.gradcheck(lambda c, a, b, d: tucker_reconstruct(c, [a, b, d]), inputs)
        )

    def test_analytic_gradients(self):
        # dL/dG = W ×_n U_n^T and dL/dU_1 = W_(1) (G ×_2 U_2 ×_3 U_3)_(1)^T for L = <W, X>.
        core, factors = _random_tucker((9, 8, 7), (3, 4, 2))
        core.requires_grad_(True)
        for f in factors:
            f.requires_grad_(True)
        weight = torch.randn(9, 8, 7, dtype=torch.float64)
        (tucker_reconstruct(core, factors) * weight).sum().backward()
        with torch.no_grad():
            expected_core = tucker_project(weight, factors)
            partial = tucker_reconstruct(core, [None, factors[1], factors[2]])
            expected_u1 = weight.reshape(9, -1) @ partial.reshape(3, -1).T
        self.assertLess(relative_error(core.grad, expected_core), 1e-13)
        self.assertLess(relative_error(factors[0].grad, expected_u1), 1e-13)


class DecompositionTests(unittest.TestCase):
    def test_hosvd_exact_for_low_rank_tensor(self):
        core, factors = _random_tucker((15, 12, 10), (3, 4, 2))
        tensor = tucker_reconstruct(core, factors)
        c, f = hosvd(tensor, (3, 4, 2))
        self.assertLess(relative_error(tucker_reconstruct(c, f), tensor), 1e-12)
        for factor in f:
            identity = torch.eye(factor.shape[1], dtype=factor.dtype)
            self.assertLess(float((factor.T @ factor - identity).abs().max()), 1e-12)

    def test_randomized_hosvd_exact_for_low_rank_tensor(self):
        core, factors = _random_tucker((15, 12, 10), (3, 4, 2))
        tensor = tucker_reconstruct(core, factors)
        c, f = randomized_hosvd(tensor, (3, 4, 2), oversampling=3, seed=7)
        self.assertLess(relative_error(tucker_reconstruct(c, f), tensor), 1e-12)

    def test_hosvd_error_decreases_with_rank(self):
        x = torch.linspace(-1, 1, 30, dtype=torch.float64)
        # Not separable: full multilinear rank with a decaying spectrum.
        tensor = torch.exp(
            -((x[:, None, None] - x[None, :, None]) ** 2 + (x[None, :, None] - x[None, None, :]) ** 2) / 0.1
        )
        errors = [relative_error(tucker_reconstruct(*hosvd(tensor, r)), tensor) for r in (1, 2, 4, 8)]
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])), errors)

    def test_parameter_count(self):
        count = tucker_parameter_count((120, 120, 100), (12, 12, 12))
        self.assertEqual(count.full_grid, 1_440_000)
        self.assertEqual(count.core, 1728)
        self.assertEqual(count.factors, 12 * 340)
        self.assertEqual(count.total, 1728 + 4080)
        self.assertAlmostEqual(count.compression_ratio, 1_440_000 / 5808)
        count2d = tucker_parameter_count((50, 40), (5, 3))
        self.assertEqual((count2d.core, count2d.factors, count2d.total), (15, 250 + 120, 385))
        module = TuckerTensor.random((120, 120, 100), (8, 16, 24))
        self.assertEqual(module.parameter_count().total, sum(p.numel() for p in module.parameters()))


class TuckerTensorTests(unittest.TestCase):
    def test_normalizations_represent_the_initial_tensor(self):
        core, factors = _random_tucker((12, 10, 8), (3, 4, 2))
        tensor = tucker_reconstruct(core, factors)
        for normalization in ("none", "rms", "orthonormal"):
            with self.subTest(normalization=normalization):
                module = TuckerTensor(core, factors, normalization)
                self.assertLess(relative_error(module(), tensor), 1e-13)

    def test_rms_normalization_removes_the_scale_gauge(self):
        core, factors = _random_tucker((12, 10, 8), (3, 4, 2))
        module = TuckerTensor(core, factors, "rms")
        before = module().detach()
        with torch.no_grad():
            module.factor_params[1].mul_(torch.tensor([10.0, 0.1, 3.0, 7.0], dtype=torch.float64))
        self.assertLess(relative_error(module(), before), 1e-13)
        for factor in module.factors():
            rms = factor.square().mean(dim=0).sqrt()
            self.assertLess(float((rms - 1).abs().max()), 1e-12)

    def test_constant_model_has_material_scale_core(self):
        module = TuckerTensor.from_tensor(torch.full((20, 16, 12), 5.0, dtype=torch.float64), 1)
        self.assertAlmostEqual(abs(float(module.core().reshape(-1)[0])), 5.0, places=10)

    def test_orthogonalize_preserves_tensor(self):
        core, factors = _random_tucker((12, 10, 8), (3, 4, 2))
        for normalization in ("none", "rms", "orthonormal"):
            module = TuckerTensor(core, factors, normalization)
            before = module().detach().clone()
            module.orthogonalize_()
            self.assertLess(relative_error(module(), before), 1e-12)
            if normalization != "orthonormal":
                self.assertLess(float(module.gauge_penalty()), 1e-20)

    def test_parameters_are_contiguous_for_lbfgs(self):
        x = torch.linspace(0, 1, 18)
        target = (4 + torch.exp(-((x[:, None, None] - x[None, :, None]) ** 2 + x[None, None, :] ** 2) / 0.1)).contiguous()
        module = TuckerTensor.from_tensor(target.permute(2, 0, 1).contiguous().permute(1, 2, 0), (3, 4, 2))
        self.assertTrue(all(p.is_contiguous() for p in module.parameters()))
        optimizer = torch.optim.LBFGS(module.parameters(), max_iter=5, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = (module() - target).square().mean()
            loss.backward()
            return loss

        optimizer.step(closure)  # LBFGS uses grad.view(-1)
        self.assertTrue(all(p.grad.is_contiguous() for p in module.parameters()))

    def test_gauge_penalty_zero_for_orthogonal_unit_rms(self):
        module = TuckerTensor.random((12, 10, 8), (3, 4, 2), dtype=torch.float64, factor_normalization="orthonormal")
        effective = [f * math.sqrt(f.shape[0]) for f in module.factors()]
        rms_module = TuckerTensor(module.core().detach(), effective, "none")
        self.assertLess(float(rms_module.gauge_penalty()), 1e-20)

    def test_export_load_roundtrip_and_device_dtype(self):
        module = TuckerTensor.random((9, 8, 7), (2, 3, 4), seed=3)
        exported = module.export()
        rebuilt = TuckerTensor.from_export(exported)
        self.assertLess(relative_error(rebuilt(), module()), 1e-6)
        state = module.state_dict()
        clone = TuckerTensor.random((9, 8, 7), (2, 3, 4), seed=99)
        clone.load_state_dict(state)
        self.assertEqual(float((clone() - module()).abs().max()), 0.0)
        double = module.double()
        self.assertEqual(double().dtype, torch.float64)
        if CUDA:
            self.assertEqual(module.cuda()().device.type, "cuda")

    def test_rms_parameterization_converges_over_a_range_of_learning_rates(self):
        # Constant background + rank-3 anomaly has multilinear rank <= 4, so a
        # rank-4 model can fit it exactly. With the gauge-fixed (rms) factors
        # one Adam learning rate serves core and factors alike.
        target_core, target_factors = _random_tucker((30, 28, 24), (3, 3, 3), dtype=torch.float32, seed=4)
        anomaly = tucker_reconstruct(target_core, target_factors)
        target = 5.0 + 0.5 * anomaly / anomaly.std()
        for lr in (0.003, 0.03):
            with self.subTest(lr=lr):
                torch.manual_seed(0)
                start = torch.full_like(target, 5.0) + 0.01 * torch.randn_like(target)
                module = TuckerTensor.from_tensor(start, (4, 4, 4), factor_normalization="rms")
                optimizer = torch.optim.Adam(module.parameters(), lr=lr)
                initial = float((module() - target).square().mean())
                for _ in range(500):
                    optimizer.zero_grad()
                    loss = (module() - target).square().mean()
                    loss.backward()
                    optimizer.step()
                self.assertLess(float(loss), 1e-4 * initial)
                self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


class MappingAndDualModelTests(unittest.TestCase):
    def test_mappings_invert_and_keep_gradients_at_bounds(self):
        values = torch.tensor([1.0, 2.5, 7.999, 8.0, 1.0], dtype=torch.float64)
        sigmoid = MaterialMapping("sigmoid", lower=1.0, upper=8.0)
        recovered = sigmoid(sigmoid.inverse(values))
        self.assertLess(float((recovered - values.clamp(1 + 7e-6, 8 - 7e-6)).abs().max()), 1e-6)
        t = sigmoid.inverse(values).requires_grad_(True)
        sigmoid(t).sum().backward()
        self.assertTrue(bool((t.grad > 0).all()))
        for kind in ("softplus", "exp"):
            mapping = MaterialMapping(kind, lower=0.0, scale=1e-3)
            sigma = torch.tensor([1e-4, 1e-3, 0.05], dtype=torch.float64)
            self.assertLess(relative_error(mapping(mapping.inverse(sigma)), sigma), 1e-10)
        with self.assertRaisesRegex(ValueError, "Hard clipping"):
            MaterialMapping("identity", lower=0.0)
        with self.assertRaisesRegex(ValueError, "lower < upper"):
            MaterialMapping("sigmoid", lower=2.0, upper=1.0)

    def test_dual_model_independent_parameters_and_ranks(self):
        shape = (18, 16, 14)
        eps0 = torch.full(shape, 4.0)
        sigma0 = torch.full(shape, 2e-3)
        model = TuckerMaterialModel.from_models(
            eps0,
            sigma0,
            TuckerModelConfig(ranks=(4, 5, 3), mapping="sigmoid", bounds=(1.0, 20.0)),
            TuckerModelConfig(ranks=(2, 3, 2), mapping="softplus", bounds=(0.0, None), scale=1e-3),
        )
        eps, sigma = model()
        self.assertLess(relative_error(eps, eps0), 1e-5)
        self.assertLess(relative_error(sigma, sigma0), 1e-5)
        self.assertEqual(model.epsilon.tucker.ranks, (4, 5, 3))
        self.assertEqual(model.sigma.tucker.ranks, (2, 3, 2))
        eps_ptrs = {p.data_ptr() for p in model.epsilon.parameters()}
        sigma_ptrs = {p.data_ptr() for p in model.sigma.parameters()}
        self.assertFalse(eps_ptrs & sigma_ptrs)
        # A loss on epsilon only leaves the sigma parameters without gradient.
        eps.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.epsilon.parameters()))
        self.assertTrue(all(p.grad is None for p in model.sigma.parameters()))
        report = model.parameter_report()
        self.assertEqual(report["epsilon_r"]["total"], 60 + 18 * 4 + 16 * 5 + 14 * 3)

    def test_dual_model_optimises_both_fields_independently(self):
        torch.manual_seed(1)
        shape = (16, 14, 12)
        grid = torch.meshgrid(*(torch.linspace(0, 1, n) for n in shape), indexing="ij")
        eps_true = 4.0 + 2.0 * torch.exp(-((grid[0] - 0.4) ** 2 + (grid[1] - 0.5) ** 2) / 0.05)
        sigma_true = 1e-3 + 4e-3 * grid[2]
        model = TuckerMaterialModel.from_models(
            torch.full(shape, 4.0),
            torch.full(shape, 2e-3),
            TuckerModelConfig(ranks=4, mapping="sigmoid", bounds=(1.0, 10.0)),
            TuckerModelConfig(ranks=2, mapping="softplus", bounds=(0.0, None), scale=1e-3),
        )
        optimizer = torch.optim.Adam(model.parameter_groups(0.05, 0.05))
        start_eps = start_sigma = None
        for _ in range(300):
            optimizer.zero_grad()
            eps, sigma = model()
            loss_eps = (eps - eps_true).square().mean()
            loss_sigma = ((sigma - sigma_true) / 1e-3).square().mean()
            if start_eps is None:
                start_eps, start_sigma = float(loss_eps), float(loss_sigma)
            (loss_eps + loss_sigma).backward()
            optimizer.step()
        self.assertLess(float(loss_eps), 0.05 * start_eps)
        self.assertLess(float(loss_sigma), 0.05 * start_sigma)
        self.assertGreaterEqual(float(model.epsilon().min()), 1.0)
        self.assertGreater(float(model.sigma().min()), 0.0)

    def test_disabled_tucker_gives_dense_field(self):
        model = TuckerMaterialModel.from_models(
            torch.full((6, 5), 4.0), torch.full((6, 5), 1e-3), TuckerModelConfig(enabled=False), None
        )
        self.assertEqual(model.parameter_report()["epsilon_r"]["tucker"], False)
        self.assertEqual(model.epsilon.value.shape, (6, 5))
        eps, sigma = model()
        self.assertEqual(float((eps - 4.0).abs().max()), 0.0)

    def test_init_methods(self):
        core, factors = _random_tucker((14, 12, 10), (2, 3, 2), dtype=torch.float32)
        material = 5.0 + 0.1 * tucker_reconstruct(core, factors)
        for init in ("hosvd", "randomized"):
            field = TuckerMaterialField.from_model(material, TuckerModelConfig(ranks=(3, 4, 3), init=init, seed=2))
            self.assertLess(relative_error(field(), material), 1e-5)
        random_field = TuckerMaterialField.from_model(material, TuckerModelConfig(ranks=3, init="random", seed=5))
        self.assertEqual(random_field().shape, material.shape)
        user = TuckerMaterialField.from_factors(core, factors, MaterialMapping("identity", offset=5.0, scale=0.1))
        self.assertLess(relative_error(user(), material), 1e-6)


class ExternalFactorGradientTests(unittest.TestCase):
    def test_gradients_reach_upstream_parameters(self):
        # Factors generated by an external differentiable map (a stand-in for
        # a user's network) receive the full chain-rule gradient.
        torch.manual_seed(2)
        shape, ranks = (12, 10, 8), (3, 2, 4)
        coordinates = [torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape]
        features = [torch.stack([c ** k for k in range(5)], dim=1) for c in coordinates]
        weights = [torch.randn(5, r, dtype=torch.float64, requires_grad=True) for r in ranks]
        core = torch.randn(ranks, dtype=torch.float64, requires_grad=True)
        factors = [torch.tanh(f @ w) for f, w in zip(features, weights)]
        target = torch.randn(shape, dtype=torch.float64)
        loss = 0.5 * (tucker_reconstruct(core, factors) - target).square().sum()
        loss.backward()
        self.assertTrue(all(w.grad is not None and float(w.grad.norm()) > 0 for w in weights))
        # Finite-difference check of one upstream weight entry.
        def objective(delta):
            shifted = [w.detach().clone() for w in weights]
            shifted[1][2, 1] += delta
            fs = [torch.tanh(f @ w) for f, w in zip(features, shifted)]
            return 0.5 * float((tucker_reconstruct(core.detach(), fs) - target).square().sum())
        h = 1e-6
        numeric = (objective(h) - objective(-h)) / (2 * h)
        self.assertAlmostEqual(float(weights[1].grad[2, 1]), numeric, delta=1e-6 * max(1.0, abs(numeric)))


if __name__ == "__main__":
    unittest.main()
