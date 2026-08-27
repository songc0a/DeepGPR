import contextlib
import inspect
import io
import math
import re
import tempfile
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
    _int8_history_layout,
    _normalize_compression_block_size,
    _normalize_wavefield_compression,
    _normalize_wavefield_storage_dtype,
    _pml_phi_elements,
    decompress_wavefield_history,
)


class NumericsValidationTests(unittest.TestCase):
    def test_int8_layout_handles_partial_boundary_blocks(self):
        shape = (5, 2, 13, 18, 1)
        layout = _int8_history_layout(shape, (8, 8, 1))
        self.assertEqual(layout["blocks_xyz"], (2, 3, 1))
        self.assertEqual(layout["q_count"], math.prod(shape))
        self.assertEqual(layout["scale_count"], 5 * 2 * 2 * 3)
        self.assertEqual(layout["q_bytes_aligned"] % 4, 0)

    def test_int8_diagnostic_decoder_uses_per_block_scales(self):
        shape = (1, 1, 3, 5, 1)
        block_size = (2, 4, 1)
        layout = _int8_history_layout(shape, block_size)
        packed = torch.zeros(layout["packed_bytes"], dtype=torch.int8)
        q = torch.arange(1, 16, dtype=torch.int8)
        packed[: q.numel()] = q
        scales = packed[layout["q_bytes_aligned"] :].view(torch.float32)
        scales.copy_(torch.tensor([0.5, 1.0, 2.0, 4.0]))
        decoded = decompress_wavefield_history(packed, shape, block_size)
        expected_scales = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5, 1.0],
             [0.5, 0.5, 0.5, 0.5, 1.0],
             [2.0, 2.0, 2.0, 2.0, 4.0]]
        ).reshape(shape)
        torch.testing.assert_close(decoded, q.float().reshape(shape) * expected_scales)

    def test_int8_configuration_validation(self):
        self.assertEqual(_normalize_wavefield_compression("INT8"), "int8")
        self.assertEqual(_normalize_compression_block_size(None, 2), (8, 8, 1))
        self.assertEqual(_normalize_compression_block_size((4, 4, 4), 3), (4, 4, 4))
        with self.assertRaisesRegex(ValueError, "power of two"):
            _normalize_compression_block_size((3, 5), 2)
        with self.assertRaisesRegex(ValueError, "must contain 3"):
            _normalize_compression_block_size((8, 8), 3)
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            DeepGPR.compute(device="cpu", wavefield_compression="int8")
        with self.assertRaisesRegex(NotImplementedError, "block-level CUDA decoder"):
            DeepGPR.compute(device="cpu", wavefield_compression="zfp")

    def test_int8_memory_estimate_includes_fp32_scales(self):
        common = dict(
            device=torch.device("cuda"),
            nx=13,
            ny=18,
            nz=1,
            nt=5,
            nstep=2,
            nsr=1,
            nrx=3,
            source_waveforms=1,
            pml=[0] * 6,
            mode=2,
            sampling_interval=1,
            storage_dtype=torch.float32,
            use_async_offload=False,
            er_requires_grad=True,
            se_requires_grad=True,
        )
        fp32 = _estimate_compute_memory(**common)
        int8 = _estimate_compute_memory(
            **common,
            wavefield_compression="int8",
            compression_block_size=(8, 8, 1),
        )
        expected_one_history = _int8_history_layout(
            (5, 2, 13, 18, 1), (8, 8, 1)
        )["packed_bytes"]
        self.assertEqual(int8["saved_gradient_wavefields"], 2 * expected_one_history)
        self.assertLess(int8["saved_gradient_wavefields"], fp32["saved_gradient_wavefields"])

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

    def test_initialization_rejects_empty_native_launch_dimensions(self):
        source = torch.zeros((1, 4, 1))
        location = torch.tensor([[[1, 1, 0]]], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "model dimensions"):
            initialization(
                "cpu", torch.empty((0, 4)), torch.empty((0, 4)), None,
                source, location, location, 0.02, 3.0e-11, 0, 2,
            )

        empty_cases = (
            (
                torch.empty((0, 1, 3), dtype=torch.int32),
                torch.empty((0, 1, 3), dtype=torch.int32),
                "shot",
            ),
            (
                torch.empty((1, 0, 3), dtype=torch.int32),
                location,
                "source",
            ),
            (
                location,
                torch.empty((1, 0, 3), dtype=torch.int32),
                "receiver",
            ),
        )
        er = torch.full((4, 4), 4.0)
        se = torch.zeros_like(er)
        for source_location, receiver_location, message in empty_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                initialization(
                    "cpu", er, se, None, source,
                    source_location, receiver_location, 0.02, 3.0e-11, 0, 2,
                )

    def test_initialization_validates_coordinates_before_int32_conversion(self):
        er = torch.full((4, 4), 4.0)
        se = torch.zeros_like(er)
        source = torch.zeros((1, 4, 1))
        location = torch.tensor([[[1, 1, 0]]], dtype=torch.int64)

        fractional = location.to(torch.float64)
        fractional[0, 0, 1] = 1.5
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            initialization(
                "cpu", er, se, None, source,
                fractional, location, 0.02, 3.0e-11, 0, 2,
            )

        wrapped = location.clone()
        wrapped[0, 0, 0] = 2**32 + 1
        with self.assertRaisesRegex(ValueError, "out of range"):
            initialization(
                "cpu", er, se, None, source,
                wrapped, location, 0.02, 3.0e-11, 0, 2,
            )

        with self.assertRaisesRegex(ValueError, "finite integer"):
            initialization(
                "cpu", er, se, None, source,
                location, location, 0.02, 3.0e-11, [0, 0, 0.5, 0], 2,
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
        self.assertIn("saved E_saved and R_saved wavefields", preview)
        self.assertIn("adjoint fields and CPML", preview)
        self.assertIn("forward wavefield save directory: disabled", preview)
        self.assertIn("estimated peak CPU memory", preview)
        self.assertIn("recommended CPU capacity with 20% margin", preview)
        self.assertIn("=== End DeepGPR compute preview ===", preview)

    def test_compute_parameter_preview_requires_bool(self):
        with self.assertRaisesRegex(TypeError, "print_parameters"):
            DeepGPR.compute(device="cpu", print_parameters=1)

    def test_compute_saves_forward_wavefield_to_requested_directory(self):
        nx, ny, nt = 10, 12, 40
        er = torch.full((nx, ny), 4.0, requires_grad=True)
        se = torch.zeros((nx, ny), requires_grad=True)
        source = torch.zeros((1, nt, 1))
        source[0, 5, 0] = 1.0
        source_location = torch.tensor([[[3, 3, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[3, 6, 0]]], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = DeepGPR.compute(
                device="cpu",
                dx=0.02,
                dt=3.0e-11,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=er,
                se=se,
                pmlthick=2,
                save_forward_wavefield_path=temporary_directory,
            )
            saved_files = list(Path(temporary_directory).glob("*.pt"))
            self.assertEqual(len(saved_files), 1)
            self.assertIsNotNone(
                re.fullmatch(r"forward_wavefield_\d{2}-\d{2}\.pt", saved_files[0].name)
            )
            load_kwargs = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = True
            saved_wavefield = torch.load(saved_files[0], **load_kwargs)
            self.assertTrue(torch.equal(saved_wavefield, result[0].detach().cpu()))
            result[-1].square().mean().backward()
            self.assertTrue(torch.isfinite(er.grad).all())
            self.assertTrue(torch.isfinite(se.grad).all())

    def test_compute_rejects_invalid_forward_wavefield_path(self):
        with self.assertRaisesRegex(TypeError, "save_forward_wavefield_path"):
            DeepGPR.compute(device="cpu", save_forward_wavefield_path=123)

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

    def test_cpml_checkpoint_shapes_are_validated_before_native_call(self):
        nx, ny, nt = 8, 10, 4
        er = torch.full((nx, ny), 4.0)
        se = torch.zeros_like(er)
        source = torch.zeros((1, nt, 1))
        location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)
        _, _, pml_state = DeepGPR.checkpoint_initial_field(
            device="cpu",
            dx=0.02,
            dt=3.0e-11,
            source_amplitudes=source,
            source_location=location,
            receiver_location=location,
            er=er,
            se=se,
            pmlthick=2,
        )
        bad_state = list(pml_state)
        bad_state[0] = bad_state[0][..., :-1]

        with self.assertRaisesRegex(ValueError, "CPML face 0 component 0"):
            DeepGPR.compute(
                device="cpu",
                dx=0.02,
                dt=3.0e-11,
                source_amplitudes=source,
                source_location=location,
                receiver_location=location,
                er=er,
                se=se,
                pmlthick=2,
                PML=tuple(bad_state),
                mode=2,
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

    def test_low_precision_exact_snapshot_is_only_allocated_for_model_gradients(self):
        common = dict(
            device=torch.device("cuda"),
            nx=30,
            ny=32,
            nz=28,
            nt=100,
            nstep=1,
            nsr=1,
            nrx=4,
            source_waveforms=1,
            pml=[4, 4, 4, 4, 4, 4],
            mode=3,
            sampling_interval=10,
            storage_dtype=torch.float16,
            use_async_offload=True,
            se_requires_grad=False,
        )
        forward_only = _estimate_compute_memory(
            **common, er_requires_grad=False
        )
        with_gradient = _estimate_compute_memory(
            **common, er_requires_grad=True
        )

        self.assertEqual(forward_only["low_precision_exact_snapshot"], 0)
        self.assertEqual(
            with_gradient["low_precision_exact_snapshot"],
            3 * 30 * 32 * 28 * torch.tensor([], dtype=torch.float32).element_size(),
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
            debug=True,
        )
        receiver = result[-1]
        receiver.square().mean().backward()

        self.assertTrue(torch.isfinite(receiver).all().item())
        self.assertIsNotNone(er.grad)
        self.assertTrue(torch.isfinite(er.grad).all().item())
        self.assertEqual(int(torch.count_nonzero(er.grad[:6]).item()), 0)


if __name__ == "__main__":
    unittest.main()
