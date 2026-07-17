from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


TESTS_DIR = Path(__file__).resolve().parent
tests_path = str(TESTS_DIR)
if tests_path not in sys.path:
    sys.path.insert(0, tests_path)

import verification_utils as vu

REPO_ROOT = vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR


class WaveletTests(unittest.TestCase):
    def setUp(self):
        self.frequency = 5.0e8
        self.length = 401
        self.dt = 1.0e-11
        self.center_index = self.length // 2
        self.peak_time = self.center_index * self.dt

    def generate(self, name, **kwargs):
        function = getattr(DeepGPR.wavelet, name)
        return function(
            self.frequency,
            self.length,
            self.dt,
            self.peak_time,
            dtype=torch.float64,
            device="cpu",
            **kwargs,
        )

    def test_wavelet_module_and_legacy_exports(self):
        self.assertIs(DeepGPR.ricker, DeepGPR.wavelet.ricker)
        for name in (
            "ricker",
            "gaussian",
            "gaussian_derivative",
            "morlet",
            "sine_burst",
        ):
            self.assertTrue(callable(getattr(DeepGPR.wavelet, name)))

    def test_common_wavelets_are_finite_with_expected_peaks(self):
        wavelets = {
            "ricker": self.generate("ricker"),
            "gaussian": self.generate("gaussian"),
            "gaussian_derivative": self.generate("gaussian_derivative"),
            "morlet": self.generate("morlet"),
            "sine_burst": self.generate("sine_burst"),
        }
        for name, values in wavelets.items():
            with self.subTest(name=name):
                self.assertEqual(values.shape, (self.length,))
                self.assertEqual(values.dtype, torch.float64)
                self.assertEqual(values.device.type, "cpu")
                self.assertTrue(torch.isfinite(values).all().item())

        for name in ("ricker", "gaussian", "morlet", "sine_burst"):
            self.assertAlmostEqual(
                float(wavelets[name][self.center_index]), 1.0, places=12
            )
        self.assertAlmostEqual(
            float(wavelets["gaussian_derivative"][self.center_index]), 0.0, places=12
        )
        self.assertGreater(
            float(wavelets["gaussian_derivative"].abs().max()), 0.999
        )
        self.assertLessEqual(
            float(wavelets["gaussian_derivative"].abs().max()), 1.0
        )

    def test_ricker_matches_the_reference_formula(self):
        actual = DeepGPR.wavelet.ricker(
            self.frequency,
            self.length,
            self.dt,
            self.peak_time,
        )
        self.assertEqual(actual.dtype, torch.get_default_dtype())
        time = (
            torch.arange(self.length, dtype=torch.get_default_dtype()) * self.dt
            - self.peak_time
        )
        phase_squared = (torch.pi * self.frequency * time).square()
        expected = (1.0 - 2.0 * phase_squared) * torch.exp(-phase_squared)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_wavelet_symmetry_and_burst_support(self):
        for name in ("ricker", "gaussian", "morlet", "sine_burst"):
            values = self.generate(name)
            torch.testing.assert_close(values, values.flip(0), rtol=0.0, atol=1.0e-12)

        derivative = self.generate("gaussian_derivative")
        torch.testing.assert_close(
            derivative, -derivative.flip(0), rtol=0.0, atol=1.0e-12
        )

        burst = self.generate("sine_burst", cycles=3.0)
        half_duration = 0.5 * 3.0 / self.frequency
        time = torch.arange(self.length, dtype=torch.float64) * self.dt - self.peak_time
        outside = burst[time.abs() > half_duration]
        self.assertTrue(torch.equal(outside, torch.zeros_like(outside)))

    def test_invalid_wavelet_parameters_are_rejected(self):
        common = (self.frequency, self.length, self.dt, self.peak_time)
        invalid_calls = (
            lambda: DeepGPR.wavelet.ricker(0.0, *common[1:]),
            lambda: DeepGPR.wavelet.ricker(common[0], True, *common[2:]),
            lambda: DeepGPR.wavelet.ricker(common[0], 0, *common[2:]),
            lambda: DeepGPR.wavelet.ricker(common[0], common[1], 0.0, common[3]),
            lambda: DeepGPR.wavelet.ricker(*common[:3], float("nan")),
            lambda: DeepGPR.wavelet.ricker(*common, dtype=torch.int32),
            lambda: DeepGPR.wavelet.morlet(*common, cycles=0.0),
            lambda: DeepGPR.wavelet.sine_burst(*common, cycles=-1.0),
        )
        for callable_object in invalid_calls:
            with self.subTest(callable_object=callable_object):
                with self.assertRaises((TypeError, ValueError)):
                    callable_object()


if __name__ == "__main__":
    unittest.main()
