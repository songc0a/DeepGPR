import logging
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
from DeepGPR.config.constants import SOURCE_SPECTRUM_CUTOFF_FRACTION


# Relative L2 deviation of the 'auto' (S = 8) permittivity gradient from S = 1
# in the case below, measured at 4.1e-4 (CPU, deterministic). The tolerance
# keeps a factor of about five; S = 16 (twice the bound) measured 8.7e-2.
AUTO_GRADIENT_TOLERANCE = 2.0e-3


def _two_layer_case():
    dx, dt, nt, frequency = 0.05, 5.0e-11, 400, 2.0e8
    nx, ny = 48, 36
    x = torch.arange(nx)[:, None].float()
    y = torch.arange(ny)[None, :].float()
    eps_true = 4.0 + 2.0 * (y > 18) + 0.0 * x
    eps_start = 4.0 + 2.0 * torch.sigmoid((y - 18) / 3.0) + 0.0 * x
    source = DeepGPR.ricker(frequency, nt, dt, 1.0 / frequency).reshape(1, nt, 1)
    source_location = torch.tensor([[[12, 0, 0]], [[36, 0, 0]]], dtype=torch.int32)
    receiver_location = torch.zeros((2, nx // 2, 3), dtype=torch.int32)
    receiver_location[:, :, 0] = torch.arange(0, nx, 2)
    receiver_location[:, :, 1] = 4
    common = dict(
        device="cpu",
        dx=dx,
        dt=dt,
        source_amplitudes=source,
        source_location=source_location,
        receiver_location=receiver_location,
        sigma=torch.full((nx, ny), 2.0e-3),
        pmlthick=8,
    )
    return eps_true, eps_start, common


class SourceSpectrumBoundTests(unittest.TestCase):
    def test_ricker_maximum_frequency_is_about_three_peak_frequencies(self):
        dt = 5.0e-11
        for frequency in (1.0e8, 2.0e8, 4.0e8):
            wavelet = DeepGPR.ricker(frequency, 2000, dt, 1.0 / frequency)
            ratio = DeepGPR.source_max_frequency(wavelet.reshape(1, -1, 1), dt) / frequency
            with self.subTest(frequency=frequency):
                self.assertAlmostEqual(ratio, 3.0, delta=0.03)

    def test_cutoff_fraction_matches_ricker_spectrum_at_three_peaks(self):
        ratio = 3.0
        relative_amplitude = ratio**2 * torch.exp(torch.tensor(1.0 - ratio**2)).item()
        self.assertAlmostEqual(relative_amplitude, SOURCE_SPECTRUM_CUTOFF_FRACTION, delta=1.0e-4)

    def test_recommended_interval_for_the_2d_fwi_example(self):
        wavelet = DeepGPR.ricker(2.0e8, 2000, 5.0e-11, 5.0e-9).reshape(1, 2000, 1)
        self.assertEqual(DeepGPR.recommended_sampling_interval(wavelet, 5.0e-11), 8)
        self.assertEqual(DeepGPR.recommended_sampling_interval(wavelet[0, :, 0], 5.0e-11), 8)

    def test_several_waveforms_use_the_most_conservative_interval(self):
        dt = 5.0e-11
        low = DeepGPR.ricker(1.0e8, 2000, dt, 1.0e-8)
        high = DeepGPR.ricker(4.0e8, 2000, dt, 1.0e-8)
        pair = torch.stack((low, high)).unsqueeze(-1)
        self.assertEqual(DeepGPR.recommended_sampling_interval(low, dt), 16)
        self.assertEqual(DeepGPR.recommended_sampling_interval(high, dt), 4)
        self.assertEqual(DeepGPR.recommended_sampling_interval(pair, dt), 4)

    def test_zero_source_and_invalid_arguments(self):
        self.assertEqual(DeepGPR.recommended_sampling_interval(torch.zeros(1, 50, 1), 1.0e-11), 1)
        with self.assertRaises(ValueError):
            DeepGPR.recommended_sampling_interval(torch.zeros(1, 50, 2), 1.0e-11)
        with self.assertRaises(ValueError):
            DeepGPR.recommended_sampling_interval(torch.ones(1, 50, 1), 0.0)
        with self.assertRaises(ValueError):
            DeepGPR.source_max_frequency(torch.ones(50), 1.0e-11, cutoff_fraction=1.5)


class AutomaticSamplingIntervalTests(unittest.TestCase):
    def _gradient(self, interval, record_warnings=None):
        eps_true, eps_start, common = _two_layer_case()
        with torch.no_grad():
            observed = DeepGPR.compute(eps_r=eps_true, **common)[-1]
        eps_r = eps_start.clone().requires_grad_(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = DeepGPR.compute(
                eps_r=eps_r, model_gradient_sampling_interval=interval, **common
            )
        if record_warnings is not None:
            record_warnings.extend(caught)
        (result[-1] - observed).square().mean().backward()
        return eps_r.grad, result[0]

    def test_auto_interval_gradient_stays_close_to_exact_gradient(self):
        exact, _ = self._gradient(1)
        caught = []
        automatic, history = self._gradient("auto", caught)
        self.assertEqual(history.shape[0], (400 + 7) // 8)
        self.assertFalse([w for w in caught if issubclass(w.category, RuntimeWarning)])
        relative_error = float((automatic - exact).norm() / exact.norm())
        self.assertLess(relative_error, AUTO_GRADIENT_TOLERANCE)
        explicit, _ = self._gradient(8)
        torch.testing.assert_close(automatic, explicit, rtol=0.0, atol=0.0)

    def test_only_intervals_above_the_bound_warn(self):
        inside = []
        logger = logging.getLogger("DeepGPR")
        with self.assertLogs(logger, level="INFO") as logs:
            self._gradient(4, inside)
        self.assertFalse([w for w in inside if issubclass(w.category, RuntimeWarning)])
        self.assertTrue(any("recommended bound 8" in line for line in logs.output))

        outside = []
        self._gradient(16, outside)
        messages = [str(w.message) for w in outside if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(messages), 1)
        self.assertIn("exceeds the recommended bound 8", messages[0])

    def test_invalid_interval_values_are_rejected(self):
        _, eps_start, common = _two_layer_case()
        for value in ("fast", 0, -2, True, 2.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer or 'auto'"):
                    DeepGPR.compute(
                        eps_r=eps_start, model_gradient_sampling_interval=value, **common
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
