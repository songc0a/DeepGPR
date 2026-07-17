import contextlib
import io
import math
import unittest

import torch

from pathlib import Path
import sys

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

from DeepGPR.common import (
    _normalize_grid_spacing,
    build_pml_phi,
    buildpmlcoeffs,
    check_cfl,
    initialization,
)
from DeepGPR.compute2 import (
    _estimate_compute_memory,
    _normalize_wavefield_storage_dtype,
    _pml_phi_elements,
)


class NumericsValidationTests(unittest.TestCase):
    def test_grid_spacing_accepts_scalar_sequence_and_tensor(self):
        self.assertEqual(_normalize_grid_spacing(0.02), (0.02, 0.02, 0.02))
        self.assertEqual(
            _normalize_grid_spacing([0.02, 0.015, 0.01]),
            (0.02, 0.015, 0.01),
        )
        tensor_spacing = _normalize_grid_spacing(
            torch.tensor([0.02, 0.015, 0.01], dtype=torch.float64)
        )
        self.assertEqual(tensor_spacing, (0.02, 0.015, 0.01))

        with self.assertRaisesRegex(ValueError, "exactly three"):
            _normalize_grid_spacing([0.02, 0.015])
        with self.assertRaisesRegex(ValueError, "positive"):
            _normalize_grid_spacing([0.02, 0.0, 0.01])

    def test_high_order_cfl_rejects_second_order_time_step(self):
        er = torch.ones((12, 14, 1), dtype=torch.float32)
        mr = torch.ones_like(er)
        with self.assertRaisesRegex(ValueError, "CFL"):
            check_cfl(0.02, 4.5e-11, 12, 14, 1, er=er, mr=mr, fdtd_order=8)

    def test_high_order_cfl_includes_material_velocity(self):
        er = torch.full((12, 14, 1), 4.0, dtype=torch.float32)
        mr = torch.ones_like(er)
        check_cfl(0.02, 4.5e-11, 12, 14, 1, er=er, mr=mr, fdtd_order=8)

    def test_anisotropic_cfl_uses_each_active_axis_spacing(self):
        er = torch.full((12, 14, 1), 4.0, dtype=torch.float32)
        mr = torch.ones_like(er)
        check_cfl(0.02, 4.5e-11, 12, 14, 1, er=er, mr=mr, fdtd_order=2)
        with self.assertRaisesRegex(ValueError, "CFL"):
            check_cfl(
                [0.02, 0.005, 0.02],
                4.5e-11,
                12,
                14,
                1,
                er=er,
                mr=mr,
                fdtd_order=2,
            )

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

    def test_compute_parameter_preview_reports_memory(self):
        nx, ny, nt = 12, 16, 80
        er = torch.full((nx, ny), 4.0, requires_grad=True)
        se = torch.full((nx, ny), 2.0e-4, requires_grad=True)
        source = DeepGPR.wavelet.ricker(3.0e8, nt, 3.0e-11, 3.0e-9).reshape(1, nt, 1)
        source_location = torch.tensor([[[4, 6, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[4, 10, 0]]], dtype=torch.int32)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            DeepGPR.compute(
                device="cpu",
                dx=0.02,
                dt=3.0e-11,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=er,
                se=se,
                pmlthick=3,
                model_gradient_sampling_interval=2,
                wavefield_storage_dtype=torch.float16,
                fdtd_order=4,
                mode=2,
                print_parameters=True,
            )

        preview = output.getvalue()
        print(preview, end="")
        self.assertIn("=== DeepGPR compute preview ===", preview)
        self.assertIn("dx / dy / dz", preview)
        self.assertIn("FDTD order / gradient mode: 4 / 2", preview)
        self.assertIn("saved Eall and Rall wavefields", preview)
        self.assertIn("adjoint fields and CPML", preview)
        self.assertIn("estimated peak CPU memory", preview)
        self.assertIn("recommended CPU capacity with 20% margin", preview)
        self.assertIn("=== End DeepGPR compute preview ===", preview)

    def test_compute_parameter_preview_requires_bool(self):
        with self.assertRaisesRegex(TypeError, "print_parameters"):
            DeepGPR.compute(device="cpu", print_parameters=1)

    def test_cpml_memory_estimate_matches_allocated_tensors(self):
        nx, ny, nz, nstep = 10, 12, 14, 3
        pml = [2, 3, 4, 2, 3, 1]
        er = torch.full((nx, ny, nz), 4.0)
        mr = torch.ones((nx + 1, ny + 1, nz + 1))
        descriptors = buildpmlcoeffs(
            er,
            mr,
            2.0e-11,
            0.02,
            nx,
            ny,
            nz,
            torch.tensor(pml, dtype=torch.int32),
            torch.device("cpu"),
            torch.float32,
        )[:6]
        tensors = build_pml_phi(
            *descriptors, nstep, None, torch.device("cpu")
        )

        self.assertEqual(
            sum(tensor.numel() for tensor in tensors),
            _pml_phi_elements(nx, ny, nz, nstep, pml),
        )

    def test_anisotropic_cpml_coefficients_use_axis_spacing(self):
        nx, ny, nz = 10, 12, 14
        er = torch.full((nx, ny, nz), 4.0)
        mr = torch.ones((nx + 1, ny + 1, nz + 1))
        coefficients = buildpmlcoeffs(
            er,
            mr,
            2.0e-11,
            [0.02, 0.015, 0.01],
            nx,
            ny,
            nz,
            torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32),
            torch.device("cpu"),
            torch.float32,
        )
        x_electric, y_electric, z_electric = (
            coefficients[6],
            coefficients[10],
            coefficients[14],
        )

        self.assertFalse(torch.equal(x_electric, y_electric))
        self.assertFalse(torch.equal(y_electric, z_electric))

    def test_cuda_async_memory_estimate_splits_host_and_device_payload(self):
        common = dict(
            device=torch.device("cuda"),
            nx=30,
            ny=40,
            nz=1,
            nt=1000,
            nstep=4,
            nsr=1,
            nrx=50,
            source_waveforms=1,
            pml=[5, 5, 5, 5, 0, 0],
            mode=2,
            sampling_interval=1,
            storage_dtype=torch.float32,
            er_requires_grad=True,
            se_requires_grad=True,
        )
        resident = _estimate_compute_memory(
            **common, use_async_offload=False
        )
        offloaded = _estimate_compute_memory(
            **common, use_async_offload=True
        )

        self.assertTrue(offloaded["effective_async_offload"])
        self.assertEqual(
            offloaded["estimated_host_peak"],
            offloaded["saved_gradient_wavefields"],
        )
        self.assertGreater(offloaded["cuda_transfer_buffers"], 0)
        self.assertLess(
            offloaded["estimated_device_peak"],
            resident["estimated_device_peak"],
        )

    def test_scalar_and_equal_axis_spacing_produce_identical_results(self):
        nx, ny, nt = 14, 18, 100
        er = torch.full((nx, ny), 4.0)
        se = torch.zeros_like(er)
        source = DeepGPR.wavelet.ricker(3.0e8, nt, 3.0e-11, 3.0e-9).reshape(1, nt, 1)
        source_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[6, 11, 0]]], dtype=torch.int32)
        arguments = dict(
            device="cpu",
            dt=3.0e-11,
            source_amplitudes=source,
            source_location=source_location,
            receiver_location=receiver_location,
            er=er,
            se=se,
            pmlthick=3,
            fdtd_order=2,
            mode=2,
        )

        scalar_receiver = DeepGPR.compute(dx=0.02, **arguments)[-1]
        vector_receiver = DeepGPR.compute(
            dx=[0.02, 0.02, 0.02], **arguments
        )[-1]
        tensor_receiver = DeepGPR.compute(
            dx=torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64),
            **arguments,
        )[-1]

        torch.testing.assert_close(scalar_receiver, vector_receiver, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scalar_receiver, tensor_receiver, rtol=0.0, atol=0.0)

    def test_anisotropic_forward_and_backward_are_finite(self):
        nx, ny, nt = 14, 18, 120
        er = torch.full((nx, ny), 4.0, requires_grad=True)
        se = torch.zeros_like(er)
        source = DeepGPR.wavelet.ricker(3.0e8, nt, 3.0e-11, 3.0e-9).reshape(1, nt, 1)
        source_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[6, 11, 0]]], dtype=torch.int32)

        receiver = DeepGPR.compute(
            device="cpu",
            dx=[0.02, 0.015, 0.01],
            dt=3.0e-11,
            source_amplitudes=source,
            source_location=source_location,
            receiver_location=receiver_location,
            er=er,
            se=se,
            pmlthick=3,
            fdtd_order=2,
            mode=2,
        )[-1]
        receiver.square().sum().backward()

        self.assertTrue(torch.isfinite(receiver).all().item())
        self.assertIsNotNone(er.grad)
        self.assertTrue(torch.isfinite(er.grad).all().item())
        self.assertGreater(int(torch.count_nonzero(er.grad[4:-4, 4:-4]).item()), 0)

    def test_anisotropic_base_update_uses_directional_spacing(self):
        nx, ny = 8, 10
        er = torch.full((nx, ny), 4.0)
        se = torch.zeros_like(er)
        source = torch.zeros((1, 1, 1))
        location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)

        def updated_ez(dy):
            shape = (1, nx + 1, ny + 1, 2)
            electric = tuple(torch.zeros(shape) for _ in range(3))
            hx = torch.zeros(shape)
            hx[:] = torch.arange(ny + 1, dtype=torch.float32).reshape(1, 1, ny + 1, 1)
            magnetic = (hx, torch.zeros(shape), torch.zeros(shape))
            result = DeepGPR.compute(
                device="cpu",
                dx=[0.02, dy, 0.02],
                dt=3.0e-11,
                source_amplitudes=source,
                source_location=location,
                receiver_location=location,
                er=er,
                se=se,
                E=electric,
                H=magnetic,
                pmlthick=0,
                fdtd_order=2,
                mode=2,
            )
            return result[1][2][0, 4, 5, 0]

        coarse = updated_ez(0.02)
        fine = updated_ez(0.01)
        torch.testing.assert_close(fine, 2.0 * coarse, rtol=1.0e-6, atol=0.0)

    def test_anisotropic_3d_base_update_uses_z_spacing(self):
        nx, ny, nz = 6, 7, 8
        er = torch.full((nx, ny, nz), 4.0)
        se = torch.zeros_like(er)
        source = torch.zeros((1, 1, 1))
        location = torch.tensor([[[3, 3, 4]]], dtype=torch.int32)

        def updated_ex(dz):
            shape = (1, nx + 1, ny + 1, nz + 1)
            electric = tuple(torch.zeros(shape) for _ in range(3))
            hy = torch.zeros(shape)
            hy[:] = torch.arange(nz + 1, dtype=torch.float32).reshape(1, 1, 1, nz + 1)
            magnetic = (torch.zeros(shape), hy, torch.zeros(shape))
            result = DeepGPR.compute(
                device="cpu",
                dx=[0.02, 0.02, dz],
                dt=2.0e-11,
                source_amplitudes=source,
                source_location=location,
                receiver_location=location,
                er=er,
                se=se,
                E=electric,
                H=magnetic,
                pmlthick=0,
                source_direction=0,
                reciever_direction=0,
                fdtd_order=2,
                mode=3,
            )
            return result[1][0][0, 3, 3, 4]

        coarse = updated_ex(0.02)
        fine = updated_ex(0.01)
        torch.testing.assert_close(fine, 2.0 * coarse, rtol=1.0e-6, atol=0.0)

    def test_anisotropic_source_scaling_uses_cell_dimensions(self):
        nx, ny = 8, 10
        er = torch.full((nx, ny), 4.0)
        se = torch.zeros_like(er)
        source = torch.ones((1, 1, 1))
        location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)

        def source_sample(dy):
            return DeepGPR.compute(
                device="cpu",
                dx=[0.02, dy, 0.01],
                dt=2.0e-11,
                source_amplitudes=source,
                source_location=location,
                receiver_location=location,
                er=er,
                se=se,
                pmlthick=0,
                source_direction=2,
                reciever_direction=2,
                fdtd_order=2,
                mode=2,
            )[-1][0, 0, 0]

        coarse = source_sample(0.02)
        fine = source_sample(0.01)
        torch.testing.assert_close(fine, 2.0 * coarse, rtol=1.0e-6, atol=0.0)

    def test_anisotropic_material_gradient_matches_finite_difference(self):
        nx, ny, nt = 14, 18, 160
        spacing = [0.02, 0.015, 0.01]
        dt = 3.0e-11
        se = torch.zeros((nx, ny))
        source = DeepGPR.wavelet.ricker(3.0e8, nt, dt, 3.0e-9).reshape(1, nt, 1)
        source_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[6, 11, 0]]], dtype=torch.int32)

        def objective(model):
            receiver = DeepGPR.compute(
                device="cpu",
                dx=spacing,
                dt=dt,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=model,
                se=se,
                pmlthick=3,
                model_gradient_sampling_interval=1,
                wavefield_storage_dtype=torch.float32,
                fdtd_order=2,
                mode=2,
            )[-1]
            return 0.5 * receiver.square().sum()

        model = torch.full((nx, ny), 4.0, requires_grad=True)
        objective(model).backward()
        cell = (6, 9)
        direction = torch.zeros_like(model)
        direction[cell] = 1.0
        perturbation = 1.0e-2
        with torch.no_grad():
            loss_plus = objective(model.detach() + perturbation * direction)
            loss_minus = objective(model.detach() - perturbation * direction)
        finite_difference = (loss_plus - loss_minus) / (2.0 * perturbation)
        adjoint = model.grad[cell]
        relative_error = abs(float(adjoint - finite_difference)) / max(
            abs(float(adjoint)), abs(float(finite_difference)), 1.0e-20
        )

        self.assertLess(relative_error, 0.02)

    def test_initialization_warns_for_acquisition_inside_pml(self):
        er = torch.full((12, 14), 4.0)
        se = torch.zeros_like(er)
        source = torch.zeros((1, 20, 1))
        source_location = torch.tensor([[[2, 7, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)

        with self.assertWarnsRegex(RuntimeWarning, "source.*inside CPML"):
            initialization(
                torch.device("cpu"), er, se, None, source,
                source_location, receiver_location, 0.02, 3.0e-11, 2, 2,
            )

    def test_long_3d_cpml_backward_remains_finite(self):
        nx, ny, nz = 20, 24, 24
        nt = 1500
        er = torch.full((nx, ny, nz), 4.0, dtype=torch.float32, requires_grad=True)
        se = torch.zeros_like(er)
        source = DeepGPR.wavelet.ricker(5.0e8, nt, 2.0e-11, 2.0e-9).reshape(1, nt, 1)
        source_location = torch.tensor([[[10, 7, 12]]], dtype=torch.int32)
        receiver_location = torch.tensor(
            [[[8, 7, 10], [12, 7, 14]]], dtype=torch.int32
        )

        result = DeepGPR.compute(
            device="cpu",
            dx=0.02,
            dt=2.0e-11,
            source_amplitudes=source,
            source_location=source_location,
            receiver_location=receiver_location,
            er=er,
            se=se,
            pmlthick=5,
            model_gradient_sampling_interval=10,
            wavefield_storage_dtype=torch.float32,
            fdtd_order=2,
            mode=3,
        )
        receiver = result[-1]
        receiver.square().mean().backward()

        self.assertTrue(torch.isfinite(receiver).all().item())
        self.assertIsNotNone(er.grad)
        self.assertTrue(torch.isfinite(er.grad).all().item())
        self.assertEqual(int(torch.count_nonzero(er.grad[:6]).item()), 0)


if __name__ == "__main__":
    unittest.main()
