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
import DeepGPR


def _relative_l2(candidate, reference):
    return float((candidate - reference).norm() / reference.norm().clamp_min(1.0e-30))


def _cosine(candidate, reference):
    return float(
        torch.dot(candidate.flatten(), reference.flatten())
        / (candidate.norm() * reference.norm()).clamp_min(1.0e-30)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class Int8WavefieldCompressionTests(unittest.TestCase):
    def _run_2d(self, compression):
        device = torch.device("cuda")
        nx, ny, nt = 18, 21, 96
        x = torch.linspace(0.0, 1.0, nx, device=device)[:, None]
        y = torch.linspace(0.0, 1.0, ny, device=device)[None, :]
        eps_r = (4.0 + 0.25 * x + 0.15 * y).requires_grad_(True)
        sigma = (2.0e-4 + 0.4e-4 * x + 0.2e-4 * y).requires_grad_(True)
        source = DeepGPR.wavelet.ricker(
            3.5e8, nt, 2.5e-11, 2.0e-9, device=device
        ).reshape(1, nt, 1)
        source_location = torch.tensor([[[8, 7, 0]]], dtype=torch.int32, device=device)
        receiver_location = torch.tensor(
            [[[7, 12, 0], [9, 14, 0]]], dtype=torch.int32, device=device
        )
        kwargs = dict(
            device=device,
            dx=(0.020, 0.016, 0.012),
            dt=2.5e-11,
            source_amplitudes=source,
            source_location=source_location,
            receiver_location=receiver_location,
            eps_r=eps_r,
            sigma=sigma,
            pmlthick=3,
            fdtd_order=4,
            mode=2,
            model_gradient_sampling_interval=1,
            wavefield_storage_dtype=torch.float32,
            wavefield_compression=compression,
            wavefield_compression_block_size=(8, 8) if compression == "int8" else None,
        )
        result = DeepGPR.compute(**kwargs)
        data = result[-1]
        data_scale = data.detach().abs().max().clamp_min(1.0e-12)
        target = 0.87 * data.detach()
        loss = 0.5 * ((data - target) / data_scale).square().sum()
        loss.backward()
        return {
            "history": result[0].detach(),
            "data": data.detach(),
            "eps": eps_r.grad.detach(),
            "sigma": sigma.grad.detach(),
            "kwargs": kwargs,
            "loss": float(loss.detach()),
            "target": target,
            "data_scale": data_scale,
        }

    def test_2d_wavefield_and_gradient_accuracy(self):
        reference = self._run_2d("none")
        compressed = self._run_2d("int8")
        reconstructed = DeepGPR.decompress_wavefield_history(
            compressed["history"], reference["history"].shape, (8, 8, 1)
        )
        absolute_error = float((reconstructed - reference["history"]).abs().max())
        wavefield_error = _relative_l2(reconstructed, reference["history"])

        self.assertLess(wavefield_error, 5.0e-2)
        self.assertTrue(math.isfinite(absolute_error))
        for name in ("eps", "sigma"):
            relative_error = _relative_l2(compressed[name], reference[name])
            cosine = _cosine(compressed[name], reference[name])
            with self.subTest(parameter=name):
                self.assertLess(relative_error, 1.2e-1)
                self.assertGreater(cosine, 0.995)

    def test_2d_compressed_gradient_directional_consistency(self):
        compressed = self._run_2d("int8")
        kwargs = dict(compressed["kwargs"])
        eps_base = kwargs.pop("eps_r").detach()
        sigma_base = kwargs.pop("sigma").detach()
        torch.manual_seed(4821)
        direction_eps = torch.randn_like(eps_base)
        direction_eps *= 0.05 / direction_eps.abs().max().clamp_min(1.0e-30)
        direction_sigma = torch.randn_like(sigma_base)
        direction_sigma *= 1.0e-5 / direction_sigma.abs().max().clamp_min(1.0e-30)
        adjoint = float(
            (compressed["eps"] * direction_eps).sum()
            + (compressed["sigma"] * direction_sigma).sum()
        )

        kwargs["wavefield_compression"] = "none"
        kwargs["wavefield_compression_block_size"] = None
        target = compressed["target"]
        data_scale = compressed["data_scale"]

        def objective(eps_r, sigma):
            result = DeepGPR.compute(eps_r=eps_r, sigma=sigma, **kwargs)
            data = result[-1]
            return 0.5 * ((data - target) / data_scale).square().sum()

        # The receiver data is independent of history compression. This central
        # difference therefore checks the lossy checkpoint's adjoint direction.
        best_error = math.inf
        for step in (1.0, 0.5, 0.25):
            with torch.no_grad():
                plus = float(
                    objective(
                        eps_base + step * direction_eps,
                        sigma_base + step * direction_sigma,
                    )
                )
                minus = float(
                    objective(
                        eps_base - step * direction_eps,
                        sigma_base - step * direction_sigma,
                    )
                )
            finite_difference = (plus - minus) / (2.0 * step)
            error = abs(adjoint - finite_difference) / max(
                abs(adjoint), abs(finite_difference), 1.0e-30
            )
            best_error = min(best_error, error)
        self.assertLess(best_error, 1.5e-1)

    def test_3d_partial_tiles_are_device_safe(self):
        device = torch.device("cuda")
        shape = (9, 10, 7)
        nt = 8
        eps_r = torch.full(shape, 4.0, device=device, requires_grad=True)
        sigma = torch.full(shape, 2.0e-4, device=device, requires_grad=True)
        source = torch.zeros((1, nt, 1), device=device)
        source[0, 1, 0] = 1.0
        location = torch.tensor([[[4, 5, 3]]], dtype=torch.int32, device=device)
        result = DeepGPR.compute(
            device=device,
            dx=(0.020, 0.018, 0.016),
            dt=1.0e-11,
            source_amplitudes=source,
            source_location=location,
            receiver_location=location,
            eps_r=eps_r,
            sigma=sigma,
            pmlthick=1,
            mode=3,
            wavefield_compression="int8",
            wavefield_compression_block_size=(4, 4, 4),
        )
        result[-1].square().sum().backward()
        torch.cuda.synchronize(device)
        self.assertEqual(result[0].dtype, torch.int8)
        self.assertTrue(torch.isfinite(eps_r.grad).all().item())
        self.assertTrue(torch.isfinite(sigma.grad).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
