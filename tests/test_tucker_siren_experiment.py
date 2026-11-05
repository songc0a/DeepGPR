"""Checks of the external SIREN / Tucker-SIREN experiment code.

The networks live in ``examples/tucker/siren_comparison`` (not in DeepGPR).
These tests run DeepGPR's CPU backend on a reduced 2D FWI case so they do not
need a GPU: gradients from the FDTD adjoint must reach the SIREN weights, the
Tucker cores and the factor networks; the optimizer must update every
trainable tensor; chunked (checkpointed) SIREN evaluation must give the same
gradient as the full evaluation.
"""

import sys
import unittest
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import verification_utils as vu

vu.configure_local_import()
EXPERIMENT = TESTS_DIR.parent / "examples" / "tucker" / "siren_comparison"
sys.path.insert(0, str(EXPERIMENT))

import tucker_cases as tc  # noqa: E402
from DeepGPR.tucker import cosine_similarity, relative_error, tucker_project  # noqa: E402
from networks import (  # noqa: E402
    DenseField,
    DirectTuckerField,
    PropertyMapping,
    SirenField,
    TuckerSirenField,
    prefit_tucker_siren,
)

CPU = torch.device("cpu")


def _fdtd_loss(case, observed, eps, sigma):
    data = case.compute(eps, sigma)[-1]
    return 0.5 * (data - observed).square().sum() / observed.square().sum()


class SirenExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = tc.case_2d(CPU, nt=160, stride=4)
        cls.observed = tc.observed(cls.case)
        cls.mapping = PropertyMapping(sigma_max=float(cls.case.sigma_true.max()) * 1.25)
        cls.shape = tuple(cls.case.eps_init.shape)

    def _check_all_parameters_get_gradients_and_move(self, field, lr=1e-3):
        optimizer = torch.optim.Adam(field.parameters(), lr=lr)
        before = [p.detach().clone() for p in field.parameters()]
        optimizer.zero_grad()
        eps, sigma = field()
        self.assertGreaterEqual(float(eps.min()), 1.0)
        _fdtd_loss(self.case, self.observed, eps, sigma).backward()
        for name, p in field.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(float(p.grad.abs().max()), 0.0, name)
        optimizer.step()
        for (name, p), old in zip(field.named_parameters(), before):
            self.assertGreater(float((p.detach() - old).abs().max()), 0.0, name)

    def test_gradients_reach_every_parameterization(self):
        torch.manual_seed(0)
        t_eps = self.mapping.eps_inverse(self.case.eps_init)
        t_sig = self.mapping.sigma_inverse(self.case.sigma_init)
        fields = {
            "siren": SirenField(self.shape, self.mapping, hidden=32, layers=3),
            "tucker_siren": TuckerSirenField(self.shape, self.mapping, ranks=6, hidden=16, layers=2),
            "direct_tucker": DirectTuckerField(t_eps, t_sig, self.mapping, ranks=6),
            "dense": DenseField(t_eps, t_sig, self.mapping),
        }
        with torch.no_grad():  # nonzero cores so the factor networks get gradients
            for prop in (fields["tucker_siren"].eps_field, fields["tucker_siren"].sigma_field):
                prop.core.normal_(0, 0.1)
        for name, field in fields.items():
            with self.subTest(field=name):
                self._check_all_parameters_get_gradients_and_move(field)

    def test_chain_rule_through_tucker_reconstruct_and_fdtd(self):
        torch.manual_seed(1)
        field = TuckerSirenField(self.shape, self.mapping, ranks=5, hidden=16, layers=2)
        prefit_tucker_siren(field.eps_field, self.mapping.eps_inverse(self.case.eps_init), steps=50, joint_steps=10)
        prefit_tucker_siren(field.sigma_field, self.mapping.sigma_inverse(self.case.sigma_init), steps=50, joint_steps=10)
        factors = field.eps_field.factors()
        unconstrained = DeepGPR_tucker_reconstruct(field.eps_field.core, factors)
        unconstrained.retain_grad()
        eps = self.mapping.eps(unconstrained)
        sigma = field.mapping.sigma(field.sigma_field())
        _fdtd_loss(self.case, self.observed, eps, sigma).backward()
        expected = tucker_project(unconstrained.grad, [f.detach() for f in factors])
        # float32 reductions with cancellation (unnormalized factors): a
        # wrong chain rule would be off by O(1), rounding is ~1e-3.
        self.assertLess(relative_error(field.eps_field.core.grad, expected), 2e-3)
        self.assertGreater(cosine_similarity(field.eps_field.core.grad, expected), 0.99999)

    def test_chunked_siren_gradient_equals_full(self):
        torch.manual_seed(2)
        full = SirenField(self.shape, self.mapping, hidden=24, layers=3, chunk=0)
        chunked = SirenField(self.shape, self.mapping, hidden=24, layers=3, chunk=777)
        chunked.load_state_dict(full.state_dict())
        weights = torch.randn(self.shape)
        for field in (full, chunked):
            eps, sigma = field()
            ((eps * weights).sum() + (sigma * weights).sum() * 1e3).backward()
        for (name, a), b in zip(full.named_parameters(), chunked.parameters()):
            self.assertLess(relative_error(b.grad, a.grad), 1e-5, name)


def DeepGPR_tucker_reconstruct(core, factors):
    from DeepGPR.tucker import tucker_reconstruct

    return tucker_reconstruct(core, factors)


if __name__ == "__main__":
    unittest.main()
