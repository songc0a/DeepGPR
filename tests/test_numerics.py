import math
import unittest

import torch

from DeepGPR.common import check_cfl, initialization
from DeepGPR.compute2 import _normalize_wavefield_storage_dtype


class NumericsValidationTests(unittest.TestCase):
    def test_high_order_cfl_rejects_second_order_time_step(self):
        er = torch.ones((12, 14, 1), dtype=torch.float32)
        mr = torch.ones_like(er)
        with self.assertRaisesRegex(ValueError, "CFL"):
            check_cfl(0.02, 4.5e-11, 12, 14, 1, er=er, mr=mr, fdtd_order=8)

    def test_high_order_cfl_includes_material_velocity(self):
        er = torch.full((12, 14, 1), 4.0, dtype=torch.float32)
        mr = torch.ones_like(er)
        check_cfl(0.02, 4.5e-11, 12, 14, 1, er=er, mr=mr, fdtd_order=8)

    def test_initialization_casts_native_inputs_to_float32(self):
        er = torch.full((8, 10), 4.0, dtype=torch.float64)
        se = torch.zeros_like(er)
        mr = torch.ones_like(er)
        source = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).reshape(1, 20, 1)
        source_location = torch.tensor([[[3, 3, 0]]], dtype=torch.int64)
        receiver_location = torch.tensor([[[3, 5, 0]]], dtype=torch.int64)

        result = initialization(
            torch.device("cpu"), er, se, mr, source,
            source_location, receiver_location, 0.02, 3.0e-11, 2, 2,
        )
        prepared_er, prepared_se = result[0], result[1]
        prepared_mr, prepared_source = result[11], result[15]
        self.assertEqual(prepared_er.dtype, torch.float32)
        self.assertEqual(prepared_se.dtype, torch.float32)
        self.assertEqual(prepared_mr.dtype, torch.float32)
        self.assertEqual(prepared_source.dtype, torch.float32)
        self.assertTrue(math.isfinite(float(prepared_source.max())))

    def test_nonpositive_relative_permeability_is_rejected(self):
        er = torch.full((8, 10), 4.0)
        se = torch.zeros_like(er)
        mr = torch.zeros_like(er)
        source = torch.zeros((1, 20, 1))
        location = torch.tensor([[[3, 3, 0]]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "mr"):
            initialization(
                torch.device("cpu"), er, se, mr, source,
                location, location, 0.02, 3.0e-11, 2, 2,
            )

    def test_wavefield_storage_aliases(self):
        self.assertIs(_normalize_wavefield_storage_dtype("fp32"), torch.float32)
        self.assertIs(_normalize_wavefield_storage_dtype("fp16"), torch.float16)
        self.assertIs(_normalize_wavefield_storage_dtype("bf16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            _normalize_wavefield_storage_dtype(torch.float64)


if __name__ == "__main__":
    unittest.main()
