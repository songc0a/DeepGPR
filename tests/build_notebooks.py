from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = r"""
from pathlib import Path
import sys

cwd = Path.cwd().resolve()
candidates = [cwd, cwd / "tests"]
candidates.extend(parent / "tests" for parent in cwd.parents)
NOTEBOOK_DIR = next(
    (path for path in candidates if (path / "verification_utils.py").is_file()),
    None,
)
if NOTEBOOK_DIR is None:
    raise FileNotFoundError("verification_utils.py was not found from the current directory.")
notebook_path = str(NOTEBOOK_DIR)
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

import verification_utils as vu

REPO_ROOT = vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR

LOADED_PACKAGE = vu.assert_local_deepgpr(DeepGPR, REPO_ROOT)
print(f"Repository root: {REPO_ROOT}")
print(f"DeepGPR package: {LOADED_PACKAGE}")
"""


NOTEBOOKS: dict[str, dict] = {}


NOTEBOOKS["00_local_backend_and_contracts.ipynb"] = notebook(
    [
        markdown(
            """
            # Local Backend and API Contracts

            This notebook proves that the repository-local Python package and native CPU
            library are used. It then checks the native ABI, deterministic execution,
            finite nonzero output, CPML acquisition warnings, and invalid-input guards.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import warnings
            import torch

            torch.manual_seed(2026)
            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            print(METADATA)

            library = DeepGPR.get_deepgpr_lib(DEVICE)
            required_symbols = ("forward", "backward", "set_fdtd_order", "deepgpr_abi_version")
            missing_symbols = [name for name in required_symbols if not hasattr(library, name)]
            vu.record_check(
                CHECKS,
                "native ABI and required exports",
                METADATA["native_abi"] == 4 and not missing_symbols,
                abi=METADATA["native_abi"],
                missing_symbols=missing_symbols,
                library=METADATA["native_library"],
            )

            required_wavelets = (
                "ricker",
                "gaussian",
                "gaussian_derivative",
                "morlet",
                "sine_burst",
            )
            missing_wavelets = [
                name for name in required_wavelets
                if not callable(getattr(DeepGPR.wavelet, name, None))
            ]
            vu.record_check(
                CHECKS,
                "wavelet module exports and legacy Ricker alias",
                not missing_wavelets
                and DeepGPR.ricker is DeepGPR.wavelet.ricker,
                missing_wavelets=missing_wavelets,
                legacy_alias_matches=DeepGPR.ricker is DeepGPR.wavelet.ricker,
            )
            """
        ),
        code(
            r"""
            nx, ny, nt = 20, 24, 160
            dx, dt, pml = 0.02, 3.0e-11, 4
            er = torch.full((nx, ny), 4.0)
            se = torch.full_like(er, 2.0e-4)
            source = DeepGPR.wavelet.ricker(3.0e8, nt, dt, 1.0 / 3.0e8).reshape(1, nt, 1)
            source_location = torch.tensor([[[6, 8, 0]]], dtype=torch.int32)
            receiver_location = torch.tensor(
                [[[6, 12, 0], [6, 16, 0]]], dtype=torch.int32
            )

            def smoke_run():
                return DeepGPR.compute(
                    device=DEVICE,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er,
                    se=se,
                    pmlthick=pml,
                    fdtd_order=2,
                    mode=2,
                    debug=True,
                )

            result_a = smoke_run()
            result_b = smoke_run()
            receiver_a, receiver_b = result_a[-1], result_b[-1]
            vu.assert_finite("smoke receiver", receiver_a, receiver_b)
            vu.record_check(
                CHECKS,
                "finite nonzero CPU smoke response",
                float(receiver_a.abs().max()) > 0.0,
                receiver_absmax=float(receiver_a.abs().max()),
            )
            deterministic_error = vu.max_abs_difference(receiver_a, receiver_b)
            vu.record_check(
                CHECKS,
                "deterministic repeated CPU forward run",
                deterministic_error == 0.0,
                max_abs_difference=deterministic_error,
            )
            """
        ),
        code(
            r"""
            def expect_exception(name, exception_type, callable_object):
                caught = None
                try:
                    callable_object()
                except Exception as exc:
                    caught = exc
                vu.record_check(
                    CHECKS,
                    name,
                    isinstance(caught, exception_type),
                    expected=exception_type.__name__,
                    received=None if caught is None else type(caught).__name__,
                    message=None if caught is None else str(caught),
                )

            common_arguments = dict(
                device=DEVICE,
                dx=dx,
                dt=dt,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=er,
                se=se,
                pmlthick=pml,
            )
            expect_exception(
                "relative permittivity below one is rejected",
                ValueError,
                lambda: DeepGPR.compute(**{**common_arguments, "er": torch.full_like(er, 0.9)}),
            )
            expect_exception(
                "unstable CFL time step is rejected",
                ValueError,
                lambda: DeepGPR.compute(**{**common_arguments, "dt": 1.0e-8}),
            )
            expect_exception(
                "unsupported FDTD order is rejected",
                ValueError,
                lambda: DeepGPR.compute(**{**common_arguments, "fdtd_order": 6}),
            )
            expect_exception(
                "overlapping CPML leaves no physical interior",
                ValueError,
                lambda: DeepGPR.compute(**{**common_arguments, "pmlthick": 10}),
            )

            pml_source = torch.tensor([[[4, 8, 0]]], dtype=torch.int32)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                DeepGPR.compute(
                    **{**common_arguments, "source_location": pml_source}
                )
            warning_messages = [str(item.message) for item in captured]
            vu.record_check(
                CHECKS,
                "acquisition point inside CPML emits a warning",
                any("inside CPML" in message for message in warning_messages),
                warnings=warning_messages,
            )
            """
        ),
        code(
            r"""
            REPORT_PATH = vu.save_report(
                "00_local_backend_and_contracts",
                CHECKS,
                METADATA,
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["01_forward_physics.ipynb"] = notebook(
    [
        markdown(
            """
            # Forward Physics Verification

            This notebook checks independent forward-model invariants: zero response,
            linear source scaling, source superposition, electromagnetic reciprocity,
            state-continuation equivalence, and homogeneous-medium travel time.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import torch

            torch.manual_seed(2026)
            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            dx, dt, pml = 0.02, 2.0e-11, 8
            nx, ny, nt = 48, 72, 320
            er = torch.full((nx, ny), 4.0)
            se = torch.full_like(er, 2.0e-4)
            wavelet = DeepGPR.wavelet.ricker(3.5e8, nt, dt, 3.0e-9).reshape(1, nt, 1)

            def simulate(source_amplitudes, source_location, receiver_location, **kwargs):
                return DeepGPR.compute(
                    device=DEVICE,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source_amplitudes,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=kwargs.pop("er", er),
                    se=kwargs.pop("se", se),
                    pmlthick=kwargs.pop("pmlthick", pml),
                    fdtd_order=kwargs.pop("fdtd_order", 2),
                    mode=2,
                    **kwargs,
                )

            source_a = torch.tensor([[[24, 24, 0]]], dtype=torch.int32)
            source_b = torch.tensor([[[24, 32, 0]]], dtype=torch.int32)
            receivers = torch.tensor(
                [[[24, 40, 0], [20, 44, 0]]], dtype=torch.int32
            )
            """
        ),
        code(
            r"""
            zero_result = simulate(torch.zeros_like(wavelet), source_a, receivers)[-1]
            vu.record_check(
                CHECKS,
                "zero source produces an exactly zero response",
                float(zero_result.abs().max()) == 0.0,
                response_absmax=float(zero_result.abs().max()),
            )

            response = simulate(wavelet, source_a, receivers)[-1]
            scale = 2.5
            scaled_response = simulate(scale * wavelet, source_a, receivers)[-1]
            linearity_error = vu.relative_l2(scaled_response, scale * response)
            vu.record_check(
                CHECKS,
                "source amplitude linearity",
                linearity_error < 2.0e-6,
                relative_l2=linearity_error,
                tolerance=2.0e-6,
            )

            wavelet_b = 0.65 * wavelet
            response_a = response
            response_b = simulate(wavelet_b, source_b, receivers)[-1]
            combined_sources = torch.cat((source_a, source_b), dim=1)
            combined_wavelets = torch.cat((wavelet, wavelet_b), dim=0)
            combined_response = simulate(combined_wavelets, combined_sources, receivers)[-1]
            superposition_error = vu.relative_l2(
                combined_response, response_a + response_b
            )
            vu.record_check(
                CHECKS,
                "multiple-source superposition",
                superposition_error < 3.0e-6,
                relative_l2=superposition_error,
                tolerance=3.0e-6,
            )
            """
        ),
        code(
            r"""
            point_a = torch.tensor([[[20, 24, 0]]], dtype=torch.int32)
            point_b = torch.tensor([[[30, 42, 0]]], dtype=torch.int32)
            trace_ab = simulate(wavelet, point_a, point_b)[-1]
            trace_ba = simulate(wavelet, point_b, point_a)[-1]
            reciprocity_error = vu.relative_l2(trace_ab, trace_ba)
            vu.record_check(
                CHECKS,
                "same-component source-receiver reciprocity",
                reciprocity_error < 2.0e-4,
                relative_l2=reciprocity_error,
                tolerance=2.0e-4,
            )
            """
        ),
        code(
            r"""
            split = 130
            full_result = simulate(wavelet, source_a, receivers)
            first_result = simulate(wavelet[:, :split], source_a, receivers)
            second_result = simulate(
                wavelet[:, split:],
                source_a,
                receivers,
                E=first_result[1],
                H=first_result[2],
                PML=first_result[3],
            )
            continued_response = torch.cat((first_result[-1], second_result[-1]), dim=1)
            continuation_error = vu.relative_l2(continued_response, full_result[-1])
            vu.record_check(
                CHECKS,
                "state continuation matches a single uninterrupted run",
                continuation_error < 2.0e-6,
                relative_l2=continuation_error,
                tolerance=2.0e-6,
            )
            """
        ),
        code(
            r"""
            travel_nx, travel_ny, travel_nt = 64, 112, 700
            travel_er = torch.full((travel_nx, travel_ny), 4.0)
            travel_se = torch.zeros_like(travel_er)
            travel_source = torch.tensor([[[32, 24, 0]]], dtype=torch.int32)
            travel_receivers = torch.tensor(
                [[[32, 44, 0], [32, 64, 0]]], dtype=torch.int32
            )
            travel_wavelet = DeepGPR.wavelet.ricker(
                3.0e8, travel_nt, dt, 1.0 / 3.0e8
            ).reshape(1, travel_nt, 1)
            expected_delta = vu.expected_travel_time(20 * dx, 4.0)
            travel_rows = []
            for order in (2, 4, 8):
                traces = simulate(
                    travel_wavelet,
                    travel_source,
                    travel_receivers,
                    er=travel_er,
                    se=travel_se,
                    pmlthick=10,
                    fdtd_order=order,
                )[-1][0]
                peak_near = vu.peak_sample(traces[:, 0])
                peak_far = vu.peak_sample(traces[:, 1])
                measured_delta = (peak_far - peak_near) * dt
                error = abs(measured_delta - expected_delta)
                travel_rows.append(
                    {
                        "order": order,
                        "peak_near": peak_near,
                        "peak_far": peak_far,
                        "measured_delta_s": measured_delta,
                        "expected_delta_s": expected_delta,
                        "absolute_error_s": error,
                    }
                )
                vu.record_check(
                    CHECKS,
                    f"homogeneous travel time at order {order}",
                    error < 2.5e-10,
                    **travel_rows[-1],
                    tolerance_s=2.5e-10,
                )
            """
        ),
        code(
            r"""
            vu.save_report(
                "01_forward_physics",
                CHECKS,
                METADATA,
                extra={"travel_time_rows": travel_rows},
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["02_cpml_absorption.ipynb"] = notebook(
    [
        markdown(
            """
            # CPML Absorption Verification

            This notebook checks that CPML is transparent before a boundary return can
            arrive, substantially suppresses late reflected energy, remains effective
            across several thicknesses, and is excluded from material gradients.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import torch

            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            nx, ny, nt = 64, 80, 1200
            dx, dt = 0.02, 2.0e-11
            er_base = torch.full((nx, ny), 4.0)
            se_base = torch.zeros_like(er_base)
            source_location = torch.tensor([[[32, 40, 0]]], dtype=torch.int32)
            receiver_location = torch.tensor([[[32, 42, 0]]], dtype=torch.int32)
            source = DeepGPR.wavelet.ricker(4.0e8, nt, dt, 2.5e-9).reshape(1, nt, 1)

            def simulate(pml, er=er_base, se=se_base):
                return DeepGPR.compute(
                    device=DEVICE,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er,
                    se=se,
                    pmlthick=pml,
                    fdtd_order=2,
                    mode=2,
                )

            no_pml = simulate(0)[-1]
            pml_10 = simulate(10)[-1]
            early_stop = 300
            early_error = vu.relative_l2(
                pml_10[:, :early_stop], no_pml[:, :early_stop]
            )
            vu.record_check(
                CHECKS,
                "CPML does not alter the causal early interior trace",
                early_error < 2.0e-6,
                relative_l2=early_error,
                early_stop_sample=early_stop,
                tolerance=2.0e-6,
            )
            """
        ),
        code(
            r"""
            late_start = 600
            reference_late_rms = vu.signal_rms(no_pml[:, late_start:])
            thickness_rows = []
            for thickness in (6, 10, 14):
                response = pml_10 if thickness == 10 else simulate(thickness)[-1]
                late_rms = vu.signal_rms(response[:, late_start:])
                ratio = late_rms / max(reference_late_rms, 1.0e-30)
                row = {
                    "pml_thickness": thickness,
                    "late_start_sample": late_start,
                    "late_rms": late_rms,
                    "no_pml_late_rms": reference_late_rms,
                    "reflection_rms_ratio": ratio,
                }
                thickness_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"late reflected energy is suppressed for CPML thickness {thickness}",
                    ratio < 0.25,
                    **row,
                    tolerance=0.25,
                )
            """
        ),
        code(
            r"""
            er = er_base.clone().requires_grad_(True)
            se = torch.full_like(er, 2.0e-4, requires_grad=True)
            gradient_result = simulate(10, er=er, se=se)
            gradient_result[-1].square().mean().backward()
            vu.assert_finite("CPML model gradients", er.grad, se.grad)
            boundary = vu.pml_boundary_mask(er.shape, 10, DEVICE)
            interior = ~boundary
            er_boundary_absmax = vu.boundary_absmax(er.grad, boundary)
            se_boundary_absmax = vu.boundary_absmax(se.grad, boundary)
            er_interior_absmax = float(er.grad[interior].abs().max())
            se_interior_absmax = float(se.grad[interior].abs().max())
            vu.record_check(
                CHECKS,
                "relative-permittivity gradient is exactly zero in CPML cells",
                er_boundary_absmax == 0.0 and er_interior_absmax > 0.0,
                boundary_absmax=er_boundary_absmax,
                interior_absmax=er_interior_absmax,
            )
            vu.record_check(
                CHECKS,
                "conductivity gradient is exactly zero in CPML cells",
                se_boundary_absmax == 0.0 and se_interior_absmax > 0.0,
                boundary_absmax=se_boundary_absmax,
                interior_absmax=se_interior_absmax,
            )
            """
        ),
        code(
            r"""
            vu.save_report(
                "02_cpml_absorption",
                CHECKS,
                METADATA,
                extra={"thickness_rows": thickness_rows},
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["03_gradient_2d.ipynb"] = notebook(
    [
        markdown(
            """
            # Two-Dimensional Adjoint Gradient Verification

            Central finite differences are compared with adjoint directional derivatives
            for relative permittivity and conductivity. Orders 2, 4, and 8 are tested with
            float32 wavefield storage and no temporal subsampling.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import torch

            torch.manual_seed(2026)
            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            nx, ny, nt = 24, 30, 180
            dx, dt, pml = 0.02, 3.0e-11, 4
            x = torch.arange(nx, dtype=torch.float32)[:, None]
            y = torch.arange(ny, dtype=torch.float32)[None, :]
            anomaly = torch.exp(
                -0.5 * (((x - 15.0) / 3.5) ** 2 + ((y - 17.0) / 4.5) ** 2)
            )
            er0 = torch.full((nx, ny), 4.0)
            se0 = torch.full((nx, ny), 3.0e-4)
            er_true = er0 + 0.35 * anomaly
            se_true = se0 + 1.5e-4 * anomaly
            source_location = torch.tensor([[[6, 10, 0]]], dtype=torch.int32)
            receiver_location = torch.tensor(
                [[[6, 14, 0], [6, 18, 0], [6, 22, 0]]], dtype=torch.int32
            )
            source = DeepGPR.wavelet.ricker(2.5e8, nt, dt, 4.0e-9).reshape(1, nt, 1)
            interior = vu.normalized_interior_mask((nx, ny), pml, DEVICE)
            boundary = ~interior
            """
        ),
        code(
            r"""
            all_rows = []
            for order in (2, 4, 8):
                def simulate(er_value, se_value):
                    return DeepGPR.compute(
                        device=DEVICE,
                        dx=dx,
                        dt=dt,
                        source_amplitudes=source,
                        source_location=source_location,
                        receiver_location=receiver_location,
                        er=er_value,
                        se=se_value,
                        pmlthick=pml,
                        fdtd_order=order,
                        mode=2,
                        model_gradient_sampling_interval=1,
                        wavefield_storage_dtype=torch.float32,
                    )[-1]

                with torch.no_grad():
                    observed = simulate(er_true, se_true)
                data_scale = observed.abs().max().clamp_min(1.0e-12)

                def objective(er_value, se_value):
                    residual = (simulate(er_value, se_value) - observed) / data_scale
                    return 0.5 * residual.square().sum()

                er = er0.clone().requires_grad_(True)
                se = se0.clone().requires_grad_(True)
                loss = objective(er, se)
                loss.backward()
                vu.assert_finite(f"order {order} gradients", er.grad, se.grad)

                direction_er = vu.gradient_direction(er.grad, interior)
                direction_se = vu.gradient_direction(se.grad, interior)
                rows_er = vu.directional_derivative_rows(
                    lambda value: objective(value, se.detach()),
                    er.detach(),
                    direction_er,
                    er.grad,
                    (8.0e-2, 4.0e-2, 2.0e-2, 1.0e-2),
                )
                rows_se = vu.directional_derivative_rows(
                    lambda value: objective(er.detach(), value),
                    se.detach(),
                    direction_se,
                    se.grad,
                    (5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5),
                )
                best_er = vu.best_relative_error(rows_er)
                best_se = vu.best_relative_error(rows_se)
                row = {
                    "order": order,
                    "loss": float(loss.detach()),
                    "er_best_relative_error": best_er,
                    "se_best_relative_error": best_se,
                    "er_rows": rows_er,
                    "se_rows": rows_se,
                }
                all_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"order {order} relative-permittivity directional derivative",
                    best_er < 5.0e-3,
                    best_relative_error=best_er,
                    tolerance=5.0e-3,
                )
                vu.record_check(
                    CHECKS,
                    f"order {order} conductivity directional derivative",
                    best_se < 5.0e-3,
                    best_relative_error=best_se,
                    tolerance=5.0e-3,
                )
                vu.record_check(
                    CHECKS,
                    f"order {order} CPML material-gradient exclusion",
                    vu.boundary_absmax(er.grad, boundary) == 0.0
                    and vu.boundary_absmax(se.grad, boundary) == 0.0,
                    er_boundary_absmax=vu.boundary_absmax(er.grad, boundary),
                    se_boundary_absmax=vu.boundary_absmax(se.grad, boundary),
                )
            """
        ),
        code(
            r"""
            vu.save_report(
                "03_gradient_2d",
                CHECKS,
                METADATA,
                extra={"directional_derivative_rows": all_rows},
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["04_gradient_3d.ipynb"] = notebook(
    [
        markdown(
            """
            # Three-Dimensional Adjoint Gradient Verification

            This notebook checks the full-vector `mode=3` gradient for all electric
            source/receiver polarizations and spatial orders 2, 4, and 8. Both material
            parameters are compared with central finite differences.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import torch

            torch.manual_seed(2026)
            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            nx = ny = nz = 16
            nt, dx, dt, pml = 120, 0.02, 2.0e-11, 3
            axis = torch.arange(nx, dtype=torch.float32)
            x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
            anomaly = torch.exp(
                -0.5
                * (
                    ((x - 9.0) / 2.0) ** 2
                    + ((y - 9.0) / 2.3) ** 2
                    + ((z - 8.0) / 2.0) ** 2
                )
            )
            er0 = torch.full((nx, ny, nz), 4.0)
            se0 = torch.full_like(er0, 3.0e-4)
            er_true = er0 + 0.25 * anomaly
            se_true = se0 + 1.0e-4 * anomaly
            source_location = torch.tensor([[[8, 5, 8]]], dtype=torch.int32)
            receiver_location = torch.tensor(
                [[[6, 5, 6], [10, 5, 10]]], dtype=torch.int32
            )
            source = DeepGPR.wavelet.ricker(4.0e8, nt, dt, 2.5e-9).reshape(1, nt, 1)
            interior = vu.normalized_interior_mask((nx, ny, nz), pml, DEVICE)
            boundary = ~interior
            """
        ),
        code(
            r"""
            all_rows = []
            for order in (2, 4, 8):
                for polarization in (0, 1, 2):
                    def simulate(er_value, se_value):
                        return DeepGPR.compute(
                            device=DEVICE,
                            dx=dx,
                            dt=dt,
                            source_amplitudes=source,
                            source_location=source_location,
                            receiver_location=receiver_location,
                            er=er_value,
                            se=se_value,
                            pmlthick=pml,
                            source_direction=polarization,
                            reciever_direction=polarization,
                            fdtd_order=order,
                            mode=3,
                            model_gradient_sampling_interval=1,
                            wavefield_storage_dtype=torch.float32,
                        )[-1]

                    with torch.no_grad():
                        observed = simulate(er_true, se_true)
                    data_scale = observed.abs().max().clamp_min(1.0e-12)

                    def objective(er_value, se_value):
                        residual = (simulate(er_value, se_value) - observed) / data_scale
                        return 0.5 * residual.square().sum()

                    er = er0.clone().requires_grad_(True)
                    se = se0.clone().requires_grad_(True)
                    loss = objective(er, se)
                    loss.backward()
                    vu.assert_finite(
                        f"order {order} polarization {polarization} gradients",
                        er.grad,
                        se.grad,
                    )
                    direction_er = vu.gradient_direction(er.grad, interior)
                    direction_se = vu.gradient_direction(se.grad, interior)
                    rows_er = vu.directional_derivative_rows(
                        lambda value: objective(value, se.detach()),
                        er.detach(),
                        direction_er,
                        er.grad,
                        (4.0e-2, 2.0e-2, 1.0e-2),
                    )
                    rows_se = vu.directional_derivative_rows(
                        lambda value: objective(er.detach(), value),
                        se.detach(),
                        direction_se,
                        se.grad,
                        (2.0e-4, 1.0e-4, 5.0e-5),
                    )
                    best_er = vu.best_relative_error(rows_er)
                    best_se = vu.best_relative_error(rows_se)
                    case = {
                        "order": order,
                        "polarization": polarization,
                        "loss": float(loss.detach()),
                        "er_best_relative_error": best_er,
                        "se_best_relative_error": best_se,
                        "er_rows": rows_er,
                        "se_rows": rows_se,
                    }
                    all_rows.append(case)
                    case_name = f"order {order}, polarization {polarization}"
                    vu.record_check(
                        CHECKS,
                        f"3D relative-permittivity derivative: {case_name}",
                        best_er < 1.0e-2,
                        best_relative_error=best_er,
                        tolerance=1.0e-2,
                    )
                    vu.record_check(
                        CHECKS,
                        f"3D conductivity derivative: {case_name}",
                        best_se < 1.0e-2,
                        best_relative_error=best_se,
                        tolerance=1.0e-2,
                    )
                    vu.record_check(
                        CHECKS,
                        f"3D CPML gradient exclusion: {case_name}",
                        vu.boundary_absmax(er.grad, boundary) == 0.0
                        and vu.boundary_absmax(se.grad, boundary) == 0.0,
                        er_boundary_absmax=vu.boundary_absmax(er.grad, boundary),
                        se_boundary_absmax=vu.boundary_absmax(se.grad, boundary),
                    )
            """
        ),
        code(
            r"""
            caught = None
            try:
                invalid_er = er0.clone().requires_grad_(True)
                DeepGPR.compute(
                    device=DEVICE,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=invalid_er,
                    se=se0,
                    pmlthick=pml,
                    mode=2,
                )
            except Exception as exc:
                caught = exc
            vu.record_check(
                CHECKS,
                "mode=2 gradient is rejected for a 3D model",
                isinstance(caught, ValueError),
                received=None if caught is None else type(caught).__name__,
                message=None if caught is None else str(caught),
            )
            """
        ),
        code(
            r"""
            vu.save_report(
                "04_gradient_3d",
                CHECKS,
                METADATA,
                extra={"directional_derivative_rows": all_rows},
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["05_wavefield_storage.ipynb"] = notebook(
    [
        markdown(
            """
            # Wavefield Storage and Sampling Verification

            The propagated fields remain float32, while saved gradient wavefields may use
            float16 or bfloat16 and may be temporally subsampled. This notebook quantifies
            the resulting gradient approximation and verifies CUDA asynchronous offload
            when a CUDA backend is available.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import torch

            torch.manual_seed(2026)
            CPU = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, CPU)
            nx, ny, nt = 24, 30, 240
            dx, dt, pml = 0.02, 3.0e-11, 4
            x = torch.arange(nx, dtype=torch.float32)[:, None]
            y = torch.arange(ny, dtype=torch.float32)[None, :]
            anomaly = torch.exp(
                -0.5 * (((x - 15.0) / 3.5) ** 2 + ((y - 17.0) / 4.5) ** 2)
            )
            er0 = torch.full((nx, ny), 4.0)
            se0 = torch.full((nx, ny), 3.0e-4)
            er_true = er0 + 0.35 * anomaly
            se_true = se0 + 1.5e-4 * anomaly
            source_cpu = DeepGPR.wavelet.ricker(2.5e8, nt, dt, 4.0e-9).reshape(1, nt, 1)
            source_location_cpu = torch.tensor([[[6, 10, 0]]], dtype=torch.int32)
            receiver_location_cpu = torch.tensor(
                [[[6, 14, 0], [6, 18, 0], [6, 22, 0]]], dtype=torch.int32
            )
            """
        ),
        code(
            r"""
            def forward_only(device, er_value, se_value):
                return DeepGPR.compute(
                    device=device,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source_cpu.to(device),
                    source_location=source_location_cpu.to(device),
                    receiver_location=receiver_location_cpu.to(device),
                    er=er_value.to(device),
                    se=se_value.to(device),
                    pmlthick=pml,
                    fdtd_order=2,
                    mode=2,
                )[-1]

            with torch.no_grad():
                observed_cpu = forward_only(CPU, er_true, se_true)
            data_scale_cpu = observed_cpu.abs().max().clamp_min(1.0e-12)

            def gradient_run(device, storage_dtype, sampling_interval, use_async_offload=False):
                er = er0.to(device).clone().requires_grad_(True)
                se = se0.to(device).clone().requires_grad_(True)
                result = DeepGPR.compute(
                    device=device,
                    dx=dx,
                    dt=dt,
                    source_amplitudes=source_cpu.to(device),
                    source_location=source_location_cpu.to(device),
                    receiver_location=receiver_location_cpu.to(device),
                    er=er,
                    se=se,
                    pmlthick=pml,
                    fdtd_order=2,
                    mode=2,
                    model_gradient_sampling_interval=sampling_interval,
                    wavefield_storage_dtype=storage_dtype,
                    use_async_offload=use_async_offload,
                )
                observed = observed_cpu.to(device)
                scale = data_scale_cpu.to(device)
                residual = (result[-1] - observed) / scale
                loss = 0.5 * residual.square().sum()
                loss.backward()
                vu.assert_finite("storage gradients", er.grad, se.grad, result[-1])
                return {
                    "receiver": result[-1].detach().cpu(),
                    "grad_er": er.grad.detach().cpu(),
                    "grad_se": se.grad.detach().cpu(),
                    "eall_dtype": result[0].dtype,
                    "loss": float(loss.detach().cpu()),
                }

            reference = gradient_run(CPU, torch.float32, 1)
            vu.record_check(
                CHECKS,
                "reference saved wavefield dtype is float32",
                reference["eall_dtype"] == torch.float32,
                dtype=reference["eall_dtype"],
            )
            """
        ),
        code(
            r"""
            storage_rows = []
            storage_tolerances = {
                torch.float16: (1.0e-3, 0.99999),
                torch.bfloat16: (2.0e-3, 0.9999),
            }
            for storage_dtype, (relative_tolerance, cosine_tolerance) in storage_tolerances.items():
                candidate = gradient_run(CPU, storage_dtype, 1)
                receiver_difference = vu.max_abs_difference(
                    candidate["receiver"], reference["receiver"]
                )
                er_relative = vu.relative_l2(candidate["grad_er"], reference["grad_er"])
                se_relative = vu.relative_l2(candidate["grad_se"], reference["grad_se"])
                er_cosine = vu.cosine_similarity(candidate["grad_er"], reference["grad_er"])
                se_cosine = vu.cosine_similarity(candidate["grad_se"], reference["grad_se"])
                row = {
                    "dtype": str(storage_dtype),
                    "receiver_max_abs_difference": receiver_difference,
                    "er_relative_l2": er_relative,
                    "se_relative_l2": se_relative,
                    "er_cosine": er_cosine,
                    "se_cosine": se_cosine,
                }
                storage_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"{storage_dtype} changes saved wavefields but not forward data",
                    candidate["eall_dtype"] == storage_dtype and receiver_difference == 0.0,
                    **row,
                )
                vu.record_check(
                    CHECKS,
                    f"{storage_dtype} gradient approximation remains controlled",
                    max(er_relative, se_relative) < relative_tolerance
                    and min(er_cosine, se_cosine) > cosine_tolerance,
                    **row,
                    relative_l2_tolerance=relative_tolerance,
                    cosine_tolerance=cosine_tolerance,
                )
            """
        ),
        code(
            r"""
            sampling_rows = []
            sampling_tolerances = {2: (2.0e-3, 0.9999), 4: (1.0e-2, 0.999)}
            for sampling_interval, (relative_tolerance, cosine_tolerance) in sampling_tolerances.items():
                candidate = gradient_run(CPU, torch.float32, sampling_interval)
                er_relative = vu.relative_l2(candidate["grad_er"], reference["grad_er"])
                se_relative = vu.relative_l2(candidate["grad_se"], reference["grad_se"])
                er_cosine = vu.cosine_similarity(candidate["grad_er"], reference["grad_er"])
                se_cosine = vu.cosine_similarity(candidate["grad_se"], reference["grad_se"])
                row = {
                    "sampling_interval": sampling_interval,
                    "er_relative_l2": er_relative,
                    "se_relative_l2": se_relative,
                    "er_cosine": er_cosine,
                    "se_cosine": se_cosine,
                }
                sampling_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"sampling interval {sampling_interval} gradient approximation",
                    max(er_relative, se_relative) < relative_tolerance
                    and min(er_cosine, se_cosine) > cosine_tolerance,
                    **row,
                    relative_l2_tolerance=relative_tolerance,
                    cosine_tolerance=cosine_tolerance,
                )
            """
        ),
        code(
            r"""
            cuda_device = vu.selected_cuda_device()
            async_row = None
            if cuda_device is None:
                vu.record_skip(
                    CHECKS,
                    "CUDA asynchronous wavefield offload",
                    "CUDA is not available on this machine.",
                )
            else:
                cuda_metadata = vu.runtime_metadata(DeepGPR, cuda_device)
                direct = gradient_run(cuda_device, torch.float32, 1, False)
                offloaded = gradient_run(cuda_device, torch.float32, 1, True)
                async_row = {
                    "device": str(cuda_device),
                    "receiver_relative_l2": vu.relative_l2(
                        offloaded["receiver"], direct["receiver"]
                    ),
                    "er_gradient_relative_l2": vu.relative_l2(
                        offloaded["grad_er"], direct["grad_er"]
                    ),
                    "se_gradient_relative_l2": vu.relative_l2(
                        offloaded["grad_se"], direct["grad_se"]
                    ),
                    "cuda_metadata": cuda_metadata,
                }
                vu.record_check(
                    CHECKS,
                    "CUDA asynchronous offload matches device-resident storage",
                    max(
                        async_row["receiver_relative_l2"],
                        async_row["er_gradient_relative_l2"],
                        async_row["se_gradient_relative_l2"],
                    ) < 2.0e-5,
                    **async_row,
                    tolerance=2.0e-5,
                )
            """
        ),
        code(
            r"""
            vu.save_report(
                "05_wavefield_storage",
                CHECKS,
                METADATA,
                extra={
                    "storage_rows": storage_rows,
                    "sampling_rows": sampling_rows,
                    "cuda_async_row": async_row,
                },
            )
            print(f"Completed {len(CHECKS)} checks, including optional checks.")
            """
        ),
    ]
)


NOTEBOOKS["06_cpu_cuda_parity.ipynb"] = notebook(
    [
        markdown(
            """
            # CPU and CUDA Backend Parity

            Forward traces and material gradients are compared between repository-local
            CPU and CUDA libraries for 2D and 3D models at orders 2, 4, and 8. The test is
            skipped only when CUDA is unavailable, unless `DEEPGPR_REQUIRE_CUDA=1`.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import os
            import torch

            torch.manual_seed(2026)
            CPU = torch.device("cpu")
            CUDA = vu.selected_cuda_device()
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, CPU)
            REQUIRE_CUDA = os.environ.get("DEEPGPR_REQUIRE_CUDA", "0") == "1"
            parity_rows = []

            if CUDA is None:
                vu.record_check(
                    CHECKS,
                    "required CUDA availability",
                    not REQUIRE_CUDA,
                    require_cuda=REQUIRE_CUDA,
                    cuda_available=False,
                )
                vu.record_skip(
                    CHECKS,
                    "CPU/CUDA numerical parity matrix",
                    "CUDA is not available on this machine.",
                )
                CUDA_METADATA = None
            else:
                CUDA_METADATA = vu.runtime_metadata(DeepGPR, CUDA)
                vu.record_check(
                    CHECKS,
                    "repository-local CUDA backend is available",
                    True,
                    device=str(CUDA),
                    library=CUDA_METADATA["native_library"],
                    abi=CUDA_METADATA["native_abi"],
                )
            """
        ),
        code(
            r"""
            def parity_run(device, shape, nt, order, mode):
                is_3d = len(shape) == 3
                er = torch.full(shape, 4.0, device=device, requires_grad=True)
                se = torch.full(shape, 3.0e-4, device=device, requires_grad=True)
                if is_3d:
                    source_location = torch.tensor([[[8, 5, 8]]], dtype=torch.int32, device=device)
                    receiver_location = torch.tensor(
                        [[[6, 5, 6], [10, 5, 10]]], dtype=torch.int32, device=device
                    )
                else:
                    source_location = torch.tensor([[[6, 10, 0]]], dtype=torch.int32, device=device)
                    receiver_location = torch.tensor(
                        [[[6, 14, 0], [6, 18, 0]]], dtype=torch.int32, device=device
                    )
                source = DeepGPR.wavelet.ricker(3.5e8, nt, 2.0e-11, 3.0e-9).reshape(1, nt, 1).to(device)
                result = DeepGPR.compute(
                    device=device,
                    dx=0.02,
                    dt=2.0e-11,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er,
                    se=se,
                    pmlthick=3 if is_3d else 4,
                    fdtd_order=order,
                    mode=mode,
                    model_gradient_sampling_interval=1,
                    wavefield_storage_dtype=torch.float32,
                )
                loss = result[-1].square().mean()
                loss.backward()
                vu.assert_finite("backend parity", result[-1], er.grad, se.grad)
                return {
                    "receiver": result[-1].detach().cpu(),
                    "grad_er": er.grad.detach().cpu(),
                    "grad_se": se.grad.detach().cpu(),
                }
            """
        ),
        code(
            r"""
            if CUDA is not None:
                cases = [
                    {"name": "2D", "shape": (24, 30), "nt": 180, "mode": 2},
                    {"name": "3D", "shape": (16, 16, 16), "nt": 100, "mode": 3},
                ]
                for case in cases:
                    for order in (2, 4, 8):
                        cpu_result = parity_run(CPU, case["shape"], case["nt"], order, case["mode"])
                        cuda_result = parity_run(CUDA, case["shape"], case["nt"], order, case["mode"])
                        row = {
                            "case": case["name"],
                            "order": order,
                            "receiver_relative_l2": vu.relative_l2(
                                cuda_result["receiver"], cpu_result["receiver"]
                            ),
                            "er_gradient_relative_l2": vu.relative_l2(
                                cuda_result["grad_er"], cpu_result["grad_er"]
                            ),
                            "se_gradient_relative_l2": vu.relative_l2(
                                cuda_result["grad_se"], cpu_result["grad_se"]
                            ),
                            "er_gradient_cosine": vu.cosine_similarity(
                                cuda_result["grad_er"], cpu_result["grad_er"]
                            ),
                            "se_gradient_cosine": vu.cosine_similarity(
                                cuda_result["grad_se"], cpu_result["grad_se"]
                            ),
                        }
                        parity_rows.append(row)
                        vu.record_check(
                            CHECKS,
                            f"CPU/CUDA parity for {case['name']} order {order}",
                            row["receiver_relative_l2"] < 2.0e-4
                            and max(
                                row["er_gradient_relative_l2"],
                                row["se_gradient_relative_l2"],
                            ) < 5.0e-3
                            and min(
                                row["er_gradient_cosine"],
                                row["se_gradient_cosine"],
                            ) > 0.999,
                            **row,
                            receiver_tolerance=2.0e-4,
                            gradient_tolerance=5.0e-3,
                            cosine_tolerance=0.999,
                        )
            """
        ),
        code(
            r"""
            vu.save_report(
                "06_cpu_cuda_parity",
                CHECKS,
                METADATA,
                extra={"cuda_metadata": CUDA_METADATA, "parity_rows": parity_rows},
            )
            print(f"Completed {len(CHECKS)} checks, including optional checks.")
            """
        ),
    ]
)


NOTEBOOKS["07_long_run_stability.ipynb"] = notebook(
    [
        markdown(
            """
            # Long-Run CPML Stability

            Long 2D and 3D simulations exercise forward propagation, reverse-time CPML,
            float32 saved wavefields, and both material gradients for orders 2, 4, and 8.
            Every available backend is tested and every returned field is checked for
            finite values.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import time
            import torch

            torch.manual_seed(2026)
            CPU = torch.device("cpu")
            CUDA = vu.selected_cuda_device()
            DEVICES = [CPU] + ([] if CUDA is None else [CUDA])
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, CPU)
            DEVICE_METADATA = [vu.runtime_metadata(DeepGPR, device) for device in DEVICES]
            stress_rows = []
            """
        ),
        code(
            r"""
            def stress_case(device, order, dimension):
                if dimension == 2:
                    shape, nt, pml, mode = (24, 32), 2000, 5, 2
                    source_location = torch.tensor([[[12, 8, 0]]], dtype=torch.int32, device=device)
                    receiver_location = torch.tensor(
                        [[[10, 9, 0], [14, 9, 0]]], dtype=torch.int32, device=device
                    )
                    axis_x = torch.arange(shape[0], dtype=torch.float32, device=device)[:, None]
                    axis_y = torch.arange(shape[1], dtype=torch.float32, device=device)[None, :]
                    anomaly = torch.exp(
                        -0.5 * (((axis_x - 14.0) / 3.0) ** 2 + ((axis_y - 18.0) / 4.0) ** 2)
                    )
                else:
                    shape, nt, pml, mode = (20, 24, 24), 1500, 5, 3
                    source_location = torch.tensor([[[10, 7, 12]]], dtype=torch.int32, device=device)
                    receiver_location = torch.tensor(
                        [[[8, 7, 10], [12, 7, 14]]], dtype=torch.int32, device=device
                    )
                    x = torch.arange(shape[0], dtype=torch.float32, device=device)[:, None, None]
                    y = torch.arange(shape[1], dtype=torch.float32, device=device)[None, :, None]
                    z = torch.arange(shape[2], dtype=torch.float32, device=device)[None, None, :]
                    anomaly = torch.exp(
                        -0.5
                        * (
                            ((x - 11.0) / 3.0) ** 2
                            + ((y - 14.0) / 4.0) ** 2
                            + ((z - 13.0) / 3.5) ** 2
                        )
                    )
                er = (4.0 + 4.0 * anomaly).requires_grad_(True)
                se = (1.0e-3 + 1.9e-2 * anomaly).requires_grad_(True)
                source = DeepGPR.wavelet.ricker(5.0e8, nt, 2.0e-11, 2.0e-9).reshape(1, nt, 1).to(device)
                start = time.perf_counter()
                result = DeepGPR.compute(
                    device=device,
                    dx=0.02,
                    dt=2.0e-11,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er,
                    se=se,
                    pmlthick=pml,
                    fdtd_order=order,
                    mode=mode,
                    model_gradient_sampling_interval=10,
                    wavefield_storage_dtype=torch.float32,
                    debug=True,
                )
                loss = result[-1].square().mean()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start

                fields = (*result[1], *result[2], *result[3], result[-1])
                vu.assert_finite("long-run fields", *fields)
                vu.assert_finite("long-run gradients", er.grad, se.grad)
                boundary = vu.pml_boundary_mask(shape, pml, device)
                return {
                    "device": str(device),
                    "dimension": dimension,
                    "order": order,
                    "nt": nt,
                    "elapsed_seconds": elapsed,
                    "receiver_absmax": float(result[-1].detach().abs().max().cpu()),
                    "er_gradient_absmax": float(er.grad.detach().abs().max().cpu()),
                    "se_gradient_absmax": float(se.grad.detach().abs().max().cpu()),
                    "er_boundary_absmax": vu.boundary_absmax(er.grad, boundary),
                    "se_boundary_absmax": vu.boundary_absmax(se.grad, boundary),
                }
            """
        ),
        code(
            r"""
            for device in DEVICES:
                for dimension in (2, 3):
                    for order in (2, 4, 8):
                        row = stress_case(device, order, dimension)
                        stress_rows.append(row)
                        vu.record_check(
                            CHECKS,
                            f"long-run {dimension}D order {order} on {device}",
                            row["receiver_absmax"] > 0.0
                            and row["er_gradient_absmax"] > 0.0
                            and row["se_gradient_absmax"] > 0.0
                            and row["er_boundary_absmax"] == 0.0
                            and row["se_boundary_absmax"] == 0.0,
                            **row,
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            """
        ),
        code(
            r"""
            vu.save_report(
                "07_long_run_stability",
                CHECKS,
                METADATA,
                extra={"device_metadata": DEVICE_METADATA, "stress_rows": stress_rows},
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["08_openmp_parallelism.ipynb"] = notebook(
    [
        markdown(
            """
            # OpenMP Parallel Consistency

            Fresh Python processes execute the same multi-shot forward and backward case
            with one, two, and four OpenMP threads. This isolates native OpenMP runtime
            initialization and checks thread-count invariance of traces and gradients.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import os
            import subprocess
            import sys
            import tempfile
            import torch

            DEVICE = torch.device("cpu")
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, DEVICE)
            worker = NOTEBOOK_DIR / "openmp_worker.py"
            if not worker.is_file():
                raise FileNotFoundError(worker)
            """
        ),
        code(
            r"""
            thread_results = {}
            with tempfile.TemporaryDirectory(prefix="deepgpr_openmp_") as temporary_directory:
                temporary_path = Path(temporary_directory)
                for thread_count in (1, 2, 4):
                    output_path = temporary_path / f"threads_{thread_count}.pt"
                    environment = os.environ.copy()
                    environment["OMP_NUM_THREADS"] = str(thread_count)
                    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
                    completed = subprocess.run(
                        [sys.executable, str(worker), str(output_path)],
                        cwd=NOTEBOOK_DIR,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    vu.record_check(
                        CHECKS,
                        f"OpenMP worker exits successfully with {thread_count} thread(s)",
                        completed.returncode == 0 and output_path.is_file(),
                        returncode=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                    )
                    thread_results[thread_count] = torch.load(
                        output_path, map_location="cpu", weights_only=True
                    )
            """
        ),
        code(
            r"""
            reference = thread_results[1]
            comparison_rows = []
            for thread_count in (2, 4):
                candidate = thread_results[thread_count]
                row = {
                    "threads": thread_count,
                    "requested_threads": candidate["requested_omp_num_threads"],
                    "elapsed_seconds": candidate["elapsed_seconds"],
                    "receiver_relative_l2": vu.relative_l2(
                        candidate["receiver"], reference["receiver"]
                    ),
                    "er_gradient_relative_l2": vu.relative_l2(
                        candidate["grad_er"], reference["grad_er"]
                    ),
                    "se_gradient_relative_l2": vu.relative_l2(
                        candidate["grad_se"], reference["grad_se"]
                    ),
                    "er_gradient_cosine": vu.cosine_similarity(
                        candidate["grad_er"], reference["grad_er"]
                    ),
                    "se_gradient_cosine": vu.cosine_similarity(
                        candidate["grad_se"], reference["grad_se"]
                    ),
                }
                comparison_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"OpenMP numerical consistency for {thread_count} threads",
                    row["requested_threads"] == thread_count
                    and row["receiver_relative_l2"] < 1.0e-7
                    and max(
                        row["er_gradient_relative_l2"],
                        row["se_gradient_relative_l2"],
                    ) < 2.0e-6
                    and min(
                        row["er_gradient_cosine"],
                        row["se_gradient_cosine"],
                    ) > 0.999999,
                    **row,
                    receiver_tolerance=1.0e-7,
                    gradient_tolerance=2.0e-6,
                    cosine_tolerance=0.999999,
                )
            """
        ),
        code(
            r"""
            vu.save_report(
                "08_openmp_parallelism",
                CHECKS,
                METADATA,
                extra={
                    "reference_elapsed_seconds": reference["elapsed_seconds"],
                    "comparison_rows": comparison_rows,
                },
            )
            print(f"Completed {len(CHECKS)} required checks.")
            """
        ),
    ]
)


NOTEBOOKS["09_anisotropic_grid.ipynb"] = notebook(
    [
        markdown(
            """
            # Unequal-Axis Grid Spacing Verification

            This notebook verifies rectilinear grids with constant but unequal `dx`,
            `dy`, and `dz`. It checks the Python contract, CFL limit, axis-specific
            field updates, source scaling, CPML coefficients, 2D and 3D forward and
            backward propagation, material gradients, and optional CPU/CUDA parity.

            Spatially varying cell sizes within a single axis are outside the scope of
            this implementation and this verification.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import math
            import torch

            from DeepGPR.common import (
                _normalize_grid_spacing,
                buildpmlcoeffs,
                check_cfl,
            )

            torch.manual_seed(2026)
            CPU = torch.device("cpu")
            CUDA = vu.selected_cuda_device()
            CHECKS = []
            METADATA = vu.runtime_metadata(DeepGPR, CPU)
            SPACING = (0.02, 0.015, 0.01)
            """
        ),
        code(
            r"""
            normalized_inputs = [
                _normalize_grid_spacing(SPACING),
                _normalize_grid_spacing(list(SPACING)),
                _normalize_grid_spacing(torch.tensor(SPACING, dtype=torch.float64)),
            ]
            vu.record_check(
                CHECKS,
                "three-value spacing accepts tuple, list, and tensor inputs",
                all(value == SPACING for value in normalized_inputs),
                normalized_inputs=normalized_inputs,
            )

            nx, ny, nt = 14, 18, 100
            er_equal = torch.full((nx, ny), 4.0)
            se_equal = torch.zeros_like(er_equal)
            source_equal = DeepGPR.wavelet.ricker(3.0e8, nt, 3.0e-11, 3.0e-9).reshape(1, nt, 1)
            source_location_equal = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)
            receiver_location_equal = torch.tensor([[[6, 11, 0]]], dtype=torch.int32)
            equal_arguments = dict(
                device=CPU,
                dt=3.0e-11,
                source_amplitudes=source_equal,
                source_location=source_location_equal,
                receiver_location=receiver_location_equal,
                er=er_equal,
                se=se_equal,
                pmlthick=3,
                fdtd_order=2,
                mode=2,
            )
            scalar_receiver = DeepGPR.compute(dx=0.02, **equal_arguments)[-1]
            sequence_receiver = DeepGPR.compute(dx=[0.02, 0.02, 0.02], **equal_arguments)[-1]
            tensor_receiver = DeepGPR.compute(
                dx=torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64),
                **equal_arguments,
            )[-1]
            sequence_error = vu.max_abs_difference(sequence_receiver, scalar_receiver)
            tensor_error = vu.max_abs_difference(tensor_receiver, scalar_receiver)
            vu.record_check(
                CHECKS,
                "scalar spacing preserves exact equal-axis behavior",
                sequence_error == 0.0 and tensor_error == 0.0,
                sequence_max_abs_difference=sequence_error,
                tensor_max_abs_difference=tensor_error,
            )
            """
        ),
        code(
            r"""
            cfl_rows = []
            for shape in ((12, 14, 1), (10, 12, 14)):
                active_spacing = [
                    spacing for size, spacing in zip(shape, SPACING) if size > 1
                ]
                dt_limit = math.sqrt(4.0) / (
                    vu.C0 * math.sqrt(sum(1.0 / spacing**2 for spacing in active_spacing))
                )
                er_cfl = torch.full(shape, 4.0)
                mr_cfl = torch.ones_like(er_cfl)
                check_cfl(
                    SPACING,
                    0.99 * dt_limit,
                    *shape,
                    er=er_cfl,
                    mr=mr_cfl,
                    fdtd_order=2,
                )
                rejected_above_limit = False
                try:
                    check_cfl(
                        SPACING,
                        1.01 * dt_limit,
                        *shape,
                        er=er_cfl,
                        mr=mr_cfl,
                        fdtd_order=2,
                    )
                except ValueError as exc:
                    rejected_above_limit = "CFL" in str(exc)
                row = {
                    "shape": shape,
                    "independent_dt_limit": dt_limit,
                    "accepted_dt": 0.99 * dt_limit,
                    "rejected_dt": 1.01 * dt_limit,
                    "rejected_above_limit": rejected_above_limit,
                }
                cfl_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"{len(active_spacing)}D CFL boundary uses every active axis spacing",
                    rejected_above_limit,
                    **row,
                )
            """
        ),
        code(
            r"""
            def updated_ez_from_x(dx_value):
                nx, ny = 8, 10
                shape = (1, nx + 1, ny + 1, 2)
                electric = tuple(torch.zeros(shape) for _ in range(3))
                hy = torch.zeros(shape)
                hy[:] = torch.arange(nx + 1, dtype=torch.float32).reshape(1, nx + 1, 1, 1)
                magnetic = (torch.zeros(shape), hy, torch.zeros(shape))
                model = torch.full((nx, ny), 4.0)
                location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)
                return DeepGPR.compute(
                    device=CPU,
                    dx=[dx_value, 0.02, 0.02],
                    dt=3.0e-11,
                    source_amplitudes=torch.zeros((1, 1, 1)),
                    source_location=location,
                    receiver_location=location,
                    er=model,
                    se=torch.zeros_like(model),
                    E=electric,
                    H=magnetic,
                    pmlthick=0,
                    fdtd_order=2,
                    mode=2,
                )[1][2][0, 4, 5, 0]


            def updated_ez_from_y(dy_value):
                nx, ny = 8, 10
                shape = (1, nx + 1, ny + 1, 2)
                electric = tuple(torch.zeros(shape) for _ in range(3))
                hx = torch.zeros(shape)
                hx[:] = torch.arange(ny + 1, dtype=torch.float32).reshape(1, 1, ny + 1, 1)
                magnetic = (hx, torch.zeros(shape), torch.zeros(shape))
                model = torch.full((nx, ny), 4.0)
                location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)
                return DeepGPR.compute(
                    device=CPU,
                    dx=[0.02, dy_value, 0.02],
                    dt=3.0e-11,
                    source_amplitudes=torch.zeros((1, 1, 1)),
                    source_location=location,
                    receiver_location=location,
                    er=model,
                    se=torch.zeros_like(model),
                    E=electric,
                    H=magnetic,
                    pmlthick=0,
                    fdtd_order=2,
                    mode=2,
                )[1][2][0, 4, 5, 0]


            def updated_ex_from_z(dz_value):
                nx, ny, nz = 6, 7, 8
                shape = (1, nx + 1, ny + 1, nz + 1)
                electric = tuple(torch.zeros(shape) for _ in range(3))
                hy = torch.zeros(shape)
                hy[:] = torch.arange(nz + 1, dtype=torch.float32).reshape(1, 1, 1, nz + 1)
                magnetic = (torch.zeros(shape), hy, torch.zeros(shape))
                model = torch.full((nx, ny, nz), 4.0)
                location = torch.tensor([[[3, 3, 4]]], dtype=torch.int32)
                return DeepGPR.compute(
                    device=CPU,
                    dx=[0.02, 0.02, dz_value],
                    dt=2.0e-11,
                    source_amplitudes=torch.zeros((1, 1, 1)),
                    source_location=location,
                    receiver_location=location,
                    er=model,
                    se=torch.zeros_like(model),
                    E=electric,
                    H=magnetic,
                    pmlthick=0,
                    source_direction=0,
                    reciever_direction=0,
                    fdtd_order=2,
                    mode=3,
                )[1][0][0, 3, 3, 4]


            directional_rows = []
            for axis, update in (
                ("x", updated_ez_from_x),
                ("y", updated_ez_from_y),
                ("z", updated_ex_from_z),
            ):
                coarse = update(0.02)
                fine = update(0.01)
                ratio = float((fine / coarse).item())
                row = {
                    "axis": axis,
                    "coarse_update": float(coarse.item()),
                    "fine_update": float(fine.item()),
                    "fine_to_coarse_ratio": ratio,
                }
                directional_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"{axis}-direction field derivative scales as inverse spacing",
                    float(coarse.abs()) > 0.0 and math.isclose(ratio, 2.0, rel_tol=2.0e-6),
                    **row,
                    expected_ratio=2.0,
                )
            """
        ),
        code(
            r"""
            nx, ny, nz = 10, 12, 14
            er_pml = torch.full((nx, ny, nz), 4.0)
            mr_pml = torch.ones((nx + 1, ny + 1, nz + 1))
            pml_coefficients = buildpmlcoeffs(
                er_pml,
                mr_pml,
                2.0e-11,
                SPACING,
                nx,
                ny,
                nz,
                torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32),
                CPU,
                torch.float32,
            )
            x_electric, y_electric, z_electric = (
                pml_coefficients[6],
                pml_coefficients[10],
                pml_coefficients[14],
            )
            vu.assert_finite("axis-specific CPML coefficients", x_electric, y_electric, z_electric)
            cpml_rows = {
                "x_y_relative_l2": vu.relative_l2(x_electric, y_electric),
                "y_z_relative_l2": vu.relative_l2(y_electric, z_electric),
                "x_z_relative_l2": vu.relative_l2(x_electric, z_electric),
            }
            vu.record_check(
                CHECKS,
                "CPML coefficients use the boundary-normal axis spacing",
                min(cpml_rows.values()) > 0.0,
                **cpml_rows,
            )

            def z_source_sample(dy_value):
                model = torch.full((8, 10), 4.0)
                location = torch.tensor([[[4, 5, 0]]], dtype=torch.int32)
                return DeepGPR.compute(
                    device=CPU,
                    dx=[0.02, dy_value, 0.01],
                    dt=2.0e-11,
                    source_amplitudes=torch.ones((1, 1, 1)),
                    source_location=location,
                    receiver_location=location,
                    er=model,
                    se=torch.zeros_like(model),
                    pmlthick=0,
                    source_direction=2,
                    reciever_direction=2,
                    fdtd_order=2,
                    mode=2,
                )[-1][0, 0, 0]

            source_coarse = z_source_sample(0.02)
            source_fine = z_source_sample(0.01)
            source_ratio = float((source_fine / source_coarse).item())
            vu.record_check(
                CHECKS,
                "source injection uses the transverse cell dimensions",
                float(source_coarse.abs()) > 0.0
                and math.isclose(source_ratio, 2.0, rel_tol=2.0e-6),
                coarse_sample=float(source_coarse.item()),
                fine_sample=float(source_fine.item()),
                fine_to_coarse_ratio=source_ratio,
                expected_ratio=2.0,
            )
            """
        ),
        code(
            r"""
            nx, ny, nt = 16, 20, 160
            dt, pml = 3.0e-11, 3
            source = DeepGPR.wavelet.ricker(3.0e8, nt, dt, 3.0e-9).reshape(1, nt, 1)
            source_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32)
            receiver_location = torch.tensor([[[6, 13, 0]]], dtype=torch.int32)
            boundary_2d = vu.pml_boundary_mask((nx, ny), pml, CPU)
            gradient_rows = []

            def simulate_2d(er_value, se_value, order):
                return DeepGPR.compute(
                    device=CPU,
                    dx=SPACING,
                    dt=dt,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er_value,
                    se=se_value,
                    pmlthick=pml,
                    fdtd_order=order,
                    mode=2,
                    model_gradient_sampling_interval=1,
                    wavefield_storage_dtype=torch.float32,
                )[-1]

            for order in (2, 4, 8):
                er = torch.full((nx, ny), 4.0, requires_grad=True)
                se = torch.full((nx, ny), 2.0e-4, requires_grad=True)
                receiver = simulate_2d(er, se, order)
                data_scale = receiver.detach().abs().max().clamp_min(1.0e-12)
                loss = 0.5 * (receiver / data_scale).square().sum()
                loss.backward()
                vu.assert_finite(f"2D order {order}", receiver, er.grad, se.grad)
                row = {
                    "order": order,
                    "receiver_absmax": float(receiver.detach().abs().max()),
                    "er_gradient_absmax": float(er.grad.detach().abs().max()),
                    "se_gradient_absmax": float(se.grad.detach().abs().max()),
                    "er_boundary_absmax": vu.boundary_absmax(er.grad, boundary_2d),
                    "se_boundary_absmax": vu.boundary_absmax(se.grad, boundary_2d),
                }
                gradient_rows.append(row)
                vu.record_check(
                    CHECKS,
                    f"2D unequal-spacing forward and backward remain finite at order {order}",
                    row["receiver_absmax"] > 0.0
                    and row["er_gradient_absmax"] > 0.0
                    and row["se_gradient_absmax"] > 0.0
                    and row["er_boundary_absmax"] == 0.0
                    and row["se_boundary_absmax"] == 0.0,
                    **row,
                )

                if order == 2:
                    cell = (6, 10)

                    def objective(er_value, se_value):
                        value = simulate_2d(er_value, se_value, order) / data_scale
                        return 0.5 * value.square().sum()

                    fd_rows = []
                    for parameter_name, base, gradient, step, evaluator in (
                        (
                            "relative permittivity",
                            er.detach(),
                            er.grad,
                            1.0e-2,
                            lambda value: objective(value, se.detach()),
                        ),
                        (
                            "conductivity",
                            se.detach(),
                            se.grad,
                            5.0e-5,
                            lambda value: objective(er.detach(), value),
                        ),
                    ):
                        direction = torch.zeros_like(base)
                        direction[cell] = 1.0
                        with torch.no_grad():
                            finite_difference = float(
                                (evaluator(base + step * direction) - evaluator(base - step * direction))
                                / (2.0 * step)
                            )
                        adjoint = float(gradient[cell])
                        relative_error = abs(adjoint - finite_difference) / max(
                            abs(adjoint), abs(finite_difference), 1.0e-30
                        )
                        fd_row = {
                            "parameter": parameter_name,
                            "cell": cell,
                            "step": step,
                            "adjoint": adjoint,
                            "finite_difference": finite_difference,
                            "relative_error": relative_error,
                        }
                        fd_rows.append(fd_row)
                        vu.record_check(
                            CHECKS,
                            f"unequal-spacing {parameter_name} gradient matches finite differences",
                            relative_error < 2.0e-2,
                            **fd_row,
                            tolerance=2.0e-2,
                        )
            """
        ),
        code(
            r"""
            shape_3d = (12, 14, 16)
            nt_3d, dt_3d, pml_3d = 180, 2.0e-11, 2
            er_3d = torch.full(shape_3d, 4.0, requires_grad=True)
            se_3d = torch.full(shape_3d, 2.0e-4, requires_grad=True)
            source_3d = DeepGPR.wavelet.ricker(4.0e8, nt_3d, dt_3d, 2.0e-9).reshape(1, nt_3d, 1)
            source_location_3d = torch.tensor([[[6, 5, 8]]], dtype=torch.int32)
            receiver_location_3d = torch.tensor(
                [[[6, 8, 8], [6, 10, 8]]], dtype=torch.int32
            )
            result_3d = DeepGPR.compute(
                device=CPU,
                dx=SPACING,
                dt=dt_3d,
                source_amplitudes=source_3d,
                source_location=source_location_3d,
                receiver_location=receiver_location_3d,
                er=er_3d,
                se=se_3d,
                pmlthick=pml_3d,
                source_direction=0,
                reciever_direction=0,
                fdtd_order=2,
                mode=3,
                model_gradient_sampling_interval=1,
                wavefield_storage_dtype=torch.float32,
            )
            receiver_3d = result_3d[-1]
            scale_3d = receiver_3d.detach().abs().max().clamp_min(1.0e-12)
            (receiver_3d / scale_3d).square().mean().backward()
            vu.assert_finite(
                "3D unequal-spacing fields and gradients",
                *result_3d[1],
                *result_3d[2],
                *result_3d[3],
                receiver_3d,
                er_3d.grad,
                se_3d.grad,
            )
            boundary_3d = vu.pml_boundary_mask(shape_3d, pml_3d, CPU)
            row_3d = {
                "receiver_absmax": float(receiver_3d.detach().abs().max()),
                "er_gradient_absmax": float(er_3d.grad.detach().abs().max()),
                "se_gradient_absmax": float(se_3d.grad.detach().abs().max()),
                "er_boundary_absmax": vu.boundary_absmax(er_3d.grad, boundary_3d),
                "se_boundary_absmax": vu.boundary_absmax(se_3d.grad, boundary_3d),
            }
            vu.record_check(
                CHECKS,
                "3D unequal-spacing forward and backward remain finite",
                row_3d["receiver_absmax"] > 0.0
                and row_3d["er_gradient_absmax"] > 0.0
                and row_3d["se_gradient_absmax"] > 0.0
                and row_3d["er_boundary_absmax"] == 0.0
                and row_3d["se_boundary_absmax"] == 0.0,
                **row_3d,
            )
            """
        ),
        code(
            r"""
            cuda_row = None

            def parity_case(device):
                shape = (14, 18)
                er = torch.full(shape, 4.0, device=device, requires_grad=True)
                se = torch.full(shape, 2.0e-4, device=device, requires_grad=True)
                source = DeepGPR.wavelet.ricker(3.0e8, 120, 3.0e-11, 3.0e-9).reshape(1, 120, 1).to(device)
                source_location = torch.tensor([[[6, 7, 0]]], dtype=torch.int32, device=device)
                receiver_location = torch.tensor([[[6, 11, 0]]], dtype=torch.int32, device=device)
                receiver = DeepGPR.compute(
                    device=device,
                    dx=SPACING,
                    dt=3.0e-11,
                    source_amplitudes=source,
                    source_location=source_location,
                    receiver_location=receiver_location,
                    er=er,
                    se=se,
                    pmlthick=3,
                    fdtd_order=2,
                    mode=2,
                    model_gradient_sampling_interval=1,
                    wavefield_storage_dtype=torch.float32,
                )[-1]
                receiver.square().mean().backward()
                vu.assert_finite("unequal-spacing backend parity", receiver, er.grad, se.grad)
                return {
                    "receiver": receiver.detach().cpu(),
                    "grad_er": er.grad.detach().cpu(),
                    "grad_se": se.grad.detach().cpu(),
                }

            if CUDA is None:
                CUDA_METADATA = None
                vu.record_skip(
                    CHECKS,
                    "unequal-spacing CPU/CUDA parity",
                    "CUDA is not available on this machine.",
                )
            else:
                CUDA_METADATA = vu.runtime_metadata(DeepGPR, CUDA)
                cpu_parity = parity_case(CPU)
                cuda_parity = parity_case(CUDA)
                cuda_row = {
                    "device": str(CUDA),
                    "receiver_relative_l2": vu.relative_l2(
                        cuda_parity["receiver"], cpu_parity["receiver"]
                    ),
                    "er_gradient_relative_l2": vu.relative_l2(
                        cuda_parity["grad_er"], cpu_parity["grad_er"]
                    ),
                    "se_gradient_relative_l2": vu.relative_l2(
                        cuda_parity["grad_se"], cpu_parity["grad_se"]
                    ),
                    "er_gradient_cosine": vu.cosine_similarity(
                        cuda_parity["grad_er"], cpu_parity["grad_er"]
                    ),
                    "se_gradient_cosine": vu.cosine_similarity(
                        cuda_parity["grad_se"], cpu_parity["grad_se"]
                    ),
                }
                vu.record_check(
                    CHECKS,
                    "unequal-spacing CPU/CUDA parity",
                    cuda_row["receiver_relative_l2"] < 2.0e-4
                    and max(
                        cuda_row["er_gradient_relative_l2"],
                        cuda_row["se_gradient_relative_l2"],
                    ) < 5.0e-3
                    and min(
                        cuda_row["er_gradient_cosine"],
                        cuda_row["se_gradient_cosine"],
                    ) > 0.999,
                    **cuda_row,
                    receiver_tolerance=2.0e-4,
                    gradient_tolerance=5.0e-3,
                    cosine_tolerance=0.999,
                )
            """
        ),
        code(
            r"""
            vu.save_report(
                "09_anisotropic_grid",
                CHECKS,
                METADATA,
                extra={
                    "spacing": SPACING,
                    "cfl_rows": cfl_rows,
                    "directional_rows": directional_rows,
                    "cpml_rows": cpml_rows,
                    "gradient_rows": gradient_rows,
                    "finite_difference_rows": fd_rows,
                    "three_dimensional_row": row_3d,
                    "cuda_metadata": CUDA_METADATA,
                    "cuda_parity_row": cuda_row,
                },
            )
            print(f"Completed {len(CHECKS)} checks, including optional checks.")
            """
        ),
    ]
)


NOTEBOOKS["99_verification_summary.ipynb"] = notebook(
    [
        markdown(
            """
            # Verification Summary

            This notebook requires all component reports, rejects failed reports, and
            displays skipped checks separately from required passing checks.
            """
        ),
        code(BOOTSTRAP),
        code(
            r"""
            import json

            expected_reports = [
                "00_local_backend_and_contracts",
                "01_forward_physics",
                "02_cpml_absorption",
                "03_gradient_2d",
                "04_gradient_3d",
                "05_wavefield_storage",
                "06_cpu_cuda_parity",
                "07_long_run_stability",
                "08_openmp_parallelism",
                "09_anisotropic_grid",
            ]
            reports = []
            for report_name in expected_reports:
                path = vu.RESULTS_DIR / f"{report_name}.json"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing report {path}. Run the component notebooks in order."
                    )
                report = json.loads(path.read_text())
                if report.get("overall_status") != "PASS":
                    raise AssertionError(f"Report failed: {path}")
                reports.append(report)
            """
        ),
        code(
            r"""
            total_passed = 0
            total_skipped = 0
            print(f"{'Report':38s} {'Passed':>8s} {'Skipped':>8s} {'Status':>8s}")
            print("-" * 68)
            for report in reports:
                passed = sum(item["status"] == "PASS" for item in report["checks"])
                skipped = sum(item["status"] == "SKIPPED" for item in report["checks"])
                total_passed += passed
                total_skipped += skipped
                print(
                    f"{report['report']:38s} {passed:8d} {skipped:8d} "
                    f"{report['overall_status']:>8s}"
                )
            print("-" * 68)
            print(f"{'Total':38s} {total_passed:8d} {total_skipped:8d} {'PASS':>8s}")

            cuda_report = next(
                report for report in reports if report["report"] == "06_cpu_cuda_parity"
            )
            cuda_skipped = any(
                item["status"] == "SKIPPED" for item in cuda_report["checks"]
            )
            print(f"CUDA parity evidence present: {not cuda_skipped}")
            """
        ),
        code(
            r"""
            summary_metadata = {
                "component_reports": expected_reports,
                "passed_checks": total_passed,
                "skipped_checks": total_skipped,
                "cuda_parity_evidence_present": not cuda_skipped,
            }
            vu.save_report(
                "99_verification_summary",
                [
                    {
                        "name": "all component reports passed",
                        "status": "PASS",
                        **summary_metadata,
                    }
                ],
                {"repository_root": str(REPO_ROOT), "deepgpr_package": str(LOADED_PACKAGE)},
            )
            """
        ),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepGPR verification notebooks.")
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Optional notebook filenames to generate; the default is every notebook.",
    )
    arguments = parser.parse_args()
    selected = arguments.notebooks or list(NOTEBOOKS)
    unknown = sorted(set(selected) - set(NOTEBOOKS))
    if unknown:
        parser.error(f"Unknown notebook filename(s): {', '.join(unknown)}")
    for filename in selected:
        content = NOTEBOOKS[filename]
        path = HERE / filename
        path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
