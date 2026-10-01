"""FIR design, analytic-signal envelope and TV regularisation (0.1.0 fixes)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import DeepGPR  # noqa: E402


def numpy_fir(cutoff, fs, numtaps):
    n = np.arange(numtaps, dtype=np.float64)
    t = n - (numtaps - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(t == 0, 2 * cutoff / fs, np.sin(2 * np.pi * cutoff / fs * t) / (np.pi * t))
    h *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1))
    return h / h.sum()


def numpy_envelope(data, power, remove_dc=True):
    nt = data.shape[1]
    if remove_dc:
        data = data - data.mean(axis=1, keepdims=True)
    weights = np.zeros(nt)
    weights[0] = 1.0
    if nt % 2 == 0:
        weights[1 : nt // 2] = 2.0
        weights[nt // 2] = 1.0
    else:
        weights[1 : (nt + 1) // 2] = 2.0
    analytic = np.fft.ifft(np.fft.fft(data, axis=1) * weights[None, :, None], axis=1)
    return np.abs(analytic) ** power


def numpy_isotropic_tv(model, epsilon):
    squared = np.zeros_like(model, dtype=np.float64)
    for axis in range(model.ndim):
        pad = [(0, 0)] * model.ndim
        pad[axis] = (0, 1)
        squared += np.pad(np.diff(model, axis=axis), pad) ** 2
    return np.sqrt(squared + epsilon).sum()


class FirFilterTests(unittest.TestCase):
    def test_matches_windowed_sinc_for_odd_and_even_lengths(self):
        for numtaps in (2, 7, 8, 14, 15, 64):
            with self.subTest(numtaps=numtaps):
                taps = DeepGPR.design_fir_filter(6e8, 1e10, numtaps, dtype=torch.float64)
                np.testing.assert_allclose(taps.numpy(), numpy_fir(6e8, 1e10, numtaps), rtol=1e-12)
                self.assertAlmostEqual(float(taps.sum()), 1.0, places=12)

    def test_legacy_reproduces_the_even_length_centre_override(self):
        even = DeepGPR.design_fir_filter(6e8, 1e10, 8, dtype=torch.float64, legacy=True)
        fixed = DeepGPR.design_fir_filter(6e8, 1e10, 8, dtype=torch.float64)
        self.assertFalse(torch.equal(even, fixed))
        odd_legacy = DeepGPR.design_fir_filter(6e8, 1e10, 9, dtype=torch.float64, legacy=True)
        odd_fixed = DeepGPR.design_fir_filter(6e8, 1e10, 9, dtype=torch.float64)
        self.assertTrue(torch.equal(odd_legacy, odd_fixed))

    def test_apply_filter_validates_cutoff_and_length(self):
        with self.assertRaisesRegex(ValueError, "too high"):
            DeepGPR.apply_filter(torch.randn(100), 1e10, 9e9)
        with self.assertRaisesRegex(ValueError, "padding samples"):
            DeepGPR.apply_filter(torch.randn(10), 1e10, 1e8)


class EnvelopeTests(unittest.TestCase):
    def test_matches_analytic_signal_for_odd_and_even_lengths(self):
        generator = torch.Generator().manual_seed(1)
        for nt in (1, 2, 9, 10, 255, 256):
            data = torch.randn((2, nt, 3), generator=generator, dtype=torch.float64)
            for remove_dc in (True, False):
                with self.subTest(nt=nt, remove_dc=remove_dc):
                    result = DeepGPR.hilbert_transform(data, p=1.5, remove_dc=remove_dc)
                    expected = numpy_envelope(data.numpy(), 1.5, remove_dc)
                    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10, atol=1e-12)

    def test_envelope_of_a_pure_cosine_is_flat(self):
        nt = 257  # odd: the previously missing bin carries the carrier here
        t = torch.arange(nt, dtype=torch.float64)
        cosine = torch.cos(2 * torch.pi * 128 * t / nt).reshape(1, nt, 1)
        envelope = DeepGPR.hilbert_transform(cosine)
        torch.testing.assert_close(envelope, torch.ones_like(envelope), rtol=0, atol=1e-10)
        legacy = DeepGPR.hilbert_transform(cosine, legacy=True)
        self.assertGreater(float((legacy - 1).abs().max()), 0.4)

    def test_even_lengths_are_unchanged_from_legacy(self):
        data = torch.randn((1, 64, 2), dtype=torch.float64)
        torch.testing.assert_close(
            DeepGPR.hilbert_transform(data), DeepGPR.hilbert_transform(data, legacy=True),
            rtol=0, atol=0,
        )


class TotalVariationTests(unittest.TestCase):
    def test_isotropic_matches_per_cell_gradient_magnitude(self):
        generator = torch.Generator().manual_seed(2)
        for shape in ((17, 23), (17, 23, 1), (7, 8, 9)):
            model = torch.randn(shape, generator=generator, dtype=torch.float64)
            with self.subTest(shape=shape):
                value = DeepGPR.TVRegularization(1, 0, method="isotropic")(model)
                expected = numpy_isotropic_tv(model.squeeze(-1).numpy() if shape[-1] == 1
                                              else model.numpy(), 1e-8)
                # The penalty is accumulated in float32 (as in 0.0.20).
                self.assertAlmostEqual(float(value), expected, delta=1e-6 * expected)

    def test_isotropic_is_rotation_invariant_unlike_anisotropic(self):
        x, y = torch.meshgrid(torch.arange(40.0), torch.arange(40.0), indexing="ij")
        axis_ramp = x.double()
        diagonal_ramp = ((x + y) / 2 ** 0.5).double()
        iso = DeepGPR.TVRegularization(1, 0, method="isotropic", epsilon=0.0)
        aniso = DeepGPR.TVRegularization(1, 0, method="anisotropic")
        # Interior cells have unit gradient magnitude in both cases.
        self.assertAlmostEqual(float(iso(axis_ramp)) / 39, float(iso(diagonal_ramp)) / 39,
                               delta=0.6)
        self.assertGreater(float(aniso(diagonal_ramp)) / float(aniso(axis_ramp)), 1.35)

    def test_isotropic_gradient_is_finite_on_constant_models(self):
        model = torch.ones((10, 10), dtype=torch.float64, requires_grad=True)
        DeepGPR.TVRegularization(1, 0, method="isotropic")(model).backward()
        self.assertTrue(torch.isfinite(model.grad).all())

    def test_method_names(self):
        model = torch.randn((6, 7), dtype=torch.float64)
        self.assertEqual(
            float(DeepGPR.TVRegularization(1, 0, method="iso")(model)),
            float(DeepGPR.TVRegularization(1, 0, method="isotropic")(model)),
        )
        legacy = DeepGPR.TVRegularization(1, 0, method="legacy_isotropic")(model)
        dx = (model[1:] - model[:-1]).abs().sum()
        dy = (model[:, 1:] - model[:, :-1]).abs().sum()
        expected = (dx**2 + dy**2 + 1e-8).sqrt()
        torch.testing.assert_close(legacy, expected.to(legacy.dtype))
        with self.assertRaisesRegex(ValueError, "Unknown TV method"):
            DeepGPR.TVRegularization(method="l2")
        with self.assertRaisesRegex(ValueError, "at least one"):
            DeepGPR.TVRegularization()(None, None)


class ConstantTests(unittest.TestCase):
    def test_codata_2018_constants(self):
        self.assertEqual(DeepGPR.e0, 8.8541878128e-12)
        self.assertEqual(DeepGPR.m0, 1.25663706212e-06)
        self.assertAlmostEqual(DeepGPR.c ** 2 * DeepGPR.e0 * DeepGPR.m0, 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
