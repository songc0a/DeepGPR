import ctypes
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
    candidate = candidate.float()
    reference = reference.float()
    return float(
        (candidate - reference).norm()
        / reference.norm().clamp_min(1.0e-30)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaConversionBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.library = DeepGPR.get_deepgpr_lib(cls.device)
        capability = getattr(
            cls.library, "deepgpr_supports_conversion_backends", None
        )
        if capability is None or int(capability()) != 1:
            raise unittest.SkipTest(
                "CUDA library does not expose conversion-backend tests"
            )

    def _conversion_inputs(self):
        # Zeros, normals, FP16 limits/subnormals, BF16 rounding boundaries,
        # infinities, NaNs, and a deterministic random distribution.
        values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            -0.1,
            2.0 ** -24,
            2.0 ** -25,
            2.0 ** -14,
            65504.0,
            -65504.0,
            70000.0,
            -70000.0,
            1.0 + 2.0 ** -11,
            1.0 + 2.0 ** -11 - 2.0 ** -23,
            1.0 + 2.0 ** -11 + 2.0 ** -23,
            1.0 + 2.0 ** -8,
            1.0 + 2.0 ** -8 - 2.0 ** -23,
            1.0 + 2.0 ** -8 + 2.0 ** -23,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
        generator = torch.Generator(device="cpu").manual_seed(81273)
        random_values = torch.randn(4096, generator=generator) * 1000.0
        tiny_values = torch.randn(1024, generator=generator) * 2.0 ** -20
        return torch.cat(
            (torch.tensor(values, dtype=torch.float32), random_values, tiny_values)
        ).to(self.device)

    def _convert(self, values, storage_kind, backend):
        encoded = torch.empty(values.numel(), dtype=torch.int16, device=self.device)
        decoded = torch.empty_like(values)
        self.library.deepgpr_test_wavefield_conversion(
            ctypes.cast(values.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(encoded.data_ptr()),
            ctypes.cast(decoded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            values.numel(),
            storage_kind,
            backend,
        )
        torch.cuda.synchronize(self.device)
        return encoded.cpu().to(torch.int32).bitwise_and(0xFFFF), decoded.cpu()

    def _assert_conversion_parity(self, storage_kind):
        values = self._conversion_inputs()
        legacy_bits, legacy_decoded = self._convert(values, storage_kind, 0)
        native_bits, native_decoded = self._convert(values, storage_kind, 1)
        vec2_bits, vec2_decoded = self._convert(values, storage_kind, 2)
        input_cpu = values.cpu()
        nan_mask = torch.isnan(input_cpu)

        self.assertTrue(
            torch.equal(legacy_bits[~nan_mask], native_bits[~nan_mask]),
            "finite/inf encodings must be bit-exact",
        )
        self.assertTrue(
            torch.equal(legacy_bits[~nan_mask], vec2_bits[~nan_mask]),
            "finite/inf vec2 encodings must be bit-exact",
        )
        self.assertTrue(torch.isnan(legacy_decoded[nan_mask]).all().item())
        self.assertTrue(torch.isnan(native_decoded[nan_mask]).all().item())
        self.assertTrue(torch.isnan(vec2_decoded[nan_mask]).all().item())
        self.assertTrue(
            torch.equal(legacy_decoded[~nan_mask], native_decoded[~nan_mask]),
            "non-NaN decoded values must be bit-exact",
        )
        self.assertTrue(
            torch.equal(legacy_decoded[~nan_mask], vec2_decoded[~nan_mask]),
            "non-NaN vec2 decoded values must be bit-exact",
        )

        # The sign bit of both zero encodings is part of the contract.
        self.assertEqual(int(legacy_bits[0]), 0x0000)
        self.assertEqual(int(legacy_bits[1]), 0x8000)

    def test_fp16_legacy_native_special_values_are_bit_exact(self):
        self._assert_conversion_parity(storage_kind=1)

    def test_bf16_legacy_native_special_values_are_bit_exact(self):
        self._assert_conversion_parity(storage_kind=2)

    def _run_deepgpr(self, storage_dtype, backend):
        nx, ny, nt = 24, 28, 120
        x = torch.linspace(0.0, 1.0, nx, device=self.device)[:, None]
        y = torch.linspace(0.0, 1.0, ny, device=self.device)[None, :]
        eps_r = (4.0 + 0.35 * x + 0.15 * y).requires_grad_(True)
        sigma = (2.0e-4 + 0.3e-4 * x + 0.2e-4 * y).requires_grad_(True)
        source = DeepGPR.wavelet.ricker(
            3.5e8, nt, 2.5e-11, 2.0e-9, device=self.device
        ).reshape(1, nt, 1)
        source_location = torch.tensor(
            [[[10, 8, 0]]], dtype=torch.int32, device=self.device
        )
        receiver_location = torch.tensor(
            [[[8, 14, 0], [12, 18, 0]]],
            dtype=torch.int32,
            device=self.device,
        )
        result = DeepGPR.compute(
            device=self.device,
            dx=(0.020, 0.016, 0.012),
            dt=2.5e-11,
            source_amplitudes=source,
            source_location=source_location,
            receiver_location=receiver_location,
            eps_r=eps_r,
            sigma=sigma,
            pmlthick=4,
            fdtd_order=4,
            mode=2,
            model_gradient_sampling_interval=1,
            wavefield_storage_dtype=storage_dtype,
            wavefield_conversion_backend=backend,
        )
        data = result[-1]
        data_scale = data.detach().abs().max().clamp_min(1.0e-12)
        loss = 0.5 * (data / data_scale).square().sum()
        loss.backward()
        torch.cuda.synchronize(self.device)
        return {
            "history": result[0].detach().cpu(),
            "data": data.detach().cpu(),
            "eps": eps_r.grad.detach().cpu(),
            "sigma": sigma.grad.detach().cpu(),
        }

    def test_full_fp16_legacy_native_parity(self):
        legacy = self._run_deepgpr(torch.float16, "legacy")
        native = self._run_deepgpr(torch.float16, "native_scalar")
        vector = self._run_deepgpr(torch.float16, "native_vec2")
        for backend, candidate in (("native_scalar", native), ("native_vec2", vector)):
            for name in legacy:
                with self.subTest(backend=backend, quantity=name):
                    self.assertLess(_relative_l2(candidate[name], legacy[name]), 5.0e-7)

    def test_full_bf16_legacy_native_parity(self):
        legacy = self._run_deepgpr(torch.bfloat16, "legacy")
        native = self._run_deepgpr(torch.bfloat16, "native_scalar")
        vector = self._run_deepgpr(torch.bfloat16, "native_vec2")
        for backend, candidate in (("native_scalar", native), ("native_vec2", vector)):
            for name in legacy:
                with self.subTest(backend=backend, quantity=name):
                    self.assertLess(_relative_l2(candidate[name], legacy[name]), 5.0e-7)

    def test_backend_contract_rejects_irrelevant_selection(self):
        with self.assertRaisesRegex(ValueError, "only selectable"):
            DeepGPR.compute(
                device=self.device,
                dx=0.02,
                dt=1.0e-11,
                source_amplitudes=torch.zeros((1, 2, 1), device=self.device),
                source_location=torch.tensor(
                    [[[2, 2, 0]]], dtype=torch.int32, device=self.device
                ),
                receiver_location=torch.tensor(
                    [[[2, 2, 0]]], dtype=torch.int32, device=self.device
                ),
                eps_r=torch.full((6, 6), 4.0, device=self.device),
                sigma=torch.full((6, 6), 2.0e-4, device=self.device),
                pmlthick=1,
                wavefield_storage_dtype=torch.float32,
                wavefield_conversion_backend="native_scalar",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
