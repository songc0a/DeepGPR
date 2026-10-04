"""Memory estimate and human-readable pre-flight report for :func:`compute`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ..config.constants import CPU_ADJOINT_WEIGHT_CHANNELS, MEMORY_SAFETY_MARGIN
from ..utils.formatting import format_memory_size
from .pml import pml_phi_element_count
from .storage import (
    BlockSize,
    default_compression_block_size,
    history_spatial_shape,
    int8_history_layout,
    saved_history_shape,
    saved_time_steps,
)


def _element_size(dtype: torch.dtype) -> int:
    """Bytes per element of ``dtype`` (works on every supported torch version)."""
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int):
        return itemsize
    return torch.empty(0, dtype=dtype).element_size()


def estimate_compute_memory(
    *,
    device: torch.device,
    nx: int,
    ny: int,
    nz: int,
    nt: int,
    nstep: int,
    nsr: int,
    nrx: int,
    source_waveforms: int,
    pml: Sequence[int],
    mode: int,
    sampling_interval: int,
    storage_dtype: torch.dtype,
    use_async_offload: bool,
    er_requires_grad: bool,
    se_requires_grad: bool,
    save_wavefield_history: bool = True,
    wavefield_compression: str = "none",
    compression_block_size: Optional[BlockSize] = None,
    reconstruct_rhs: Optional[bool] = None,
    history_region: str = "extended",
) -> Dict[str, Any]:
    """Estimate the tensor payload of one :func:`compute` call.

    Args:
        device: Execution device.
        nx, ny, nz: Extended grid dimensions (including CPML).
        nt: Time steps.
        nstep, nsr, nrx: Shots, sources per shot, receivers per shot.
        source_waveforms: Number of source waveforms.
        pml: Six face thicknesses.
        mode: Gradient mode (3 stores three components).
        sampling_interval: History sampling interval.
        storage_dtype: History dtype.
        use_async_offload: Whether CUDA histories are offloaded to pinned RAM.
        er_requires_grad, se_requires_grad: Whether material gradients are needed.
        save_wavefield_history: Whether histories are stored at all.
        wavefield_compression: ``"none"`` or ``"int8"``.
        compression_block_size: INT8 block, default chosen from ``nz``.
        reconstruct_rhs: Whether the E-only history (``wavefield_rhs_history``)
            is used. ``None`` assumes the default ``"auto"`` with current native
            libraries: E-only for uncompressed float32 storage at
            ``sampling_interval == 1``.
        history_region: ``"extended"`` or ``"physical"`` history frames.

    Returns:
        Byte counts per category plus peaks, recommended capacities (with a
        20 % margin), ``nt_saved``, ``components_saved`` and
        ``effective_async_offload``.
    """
    float_bytes = _element_size(torch.float32)
    int_bytes = _element_size(torch.int32)
    storage_bytes = _element_size(storage_dtype)
    model_cells = nx * ny * nz
    field_cells = (nx + 1) * (ny + 1) * (nz + 1)
    int8_block = (
        (compression_block_size or default_compression_block_size(2 if nz == 1 else 3))
        if wavefield_compression == "int8"
        else None
    )
    history_cells = history_spatial_shape(nx, ny, nz, pml, history_region, int8_block)
    snapshot_cells = nstep * history_cells[0] * history_cells[1] * history_cells[2]
    components = 3 if mode == 3 else 1
    nt_saved = saved_time_steps(nt, sampling_interval)

    model_bytes = (2 * model_cells + 3 * field_cells) * float_bytes
    field_bytes = 6 * nstep * field_cells * float_bytes
    pml_phi_bytes = pml_phi_element_count(nx, ny, nz, nstep, pml) * float_bytes
    pml_coefficient_bytes = (
        8 * sum(pml) * float_bytes + 7 * sum(thickness > 0 for thickness in pml) * int_bytes
    )
    acquisition_bytes = (
        source_waveforms * nt * float_bytes
        + nstep * nsr * 3 * int_bytes
        + nstep * nrx * 3 * int_bytes
    )
    core_bytes = (
        model_bytes + field_bytes + pml_phi_bytes + pml_coefficient_bytes + acquisition_bytes
    )

    needs_backward = save_wavefield_history and (er_requires_grad or se_requires_grad)
    if reconstruct_rhs is None:
        reconstruct_rhs = (
            sampling_interval == 1
            and wavefield_compression == "none"
            and storage_dtype == torch.float32
        )
    reconstruct_rhs = bool(reconstruct_rhs and needs_backward)
    # E-only histories keep one extra (final) E frame on the device instead of R.
    final_frame_bytes = 0
    if not save_wavefield_history:
        packed_history_bytes = 0
        saved_wavefield_bytes = 0
    elif wavefield_compression == "int8":
        block_size = compression_block_size or default_compression_block_size(2 if nz == 1 else 3)
        history_mode = 3 if components == 3 else 2
        history_shape = saved_history_shape(history_mode, nt_saved, nstep, *history_cells)
        packed_history_bytes = int8_history_layout(history_shape, block_size)["packed_bytes"]
        if reconstruct_rhs:
            final_frame_bytes = int8_history_layout(
                saved_history_shape(history_mode, 1, nstep, *history_cells), block_size
            )["packed_bytes"]
            saved_wavefield_bytes = packed_history_bytes + final_frame_bytes
        else:
            saved_wavefield_bytes = (1 + int(needs_backward)) * packed_history_bytes
    else:
        packed_history_bytes = 0
        one_history = components * nt_saved * snapshot_cells * storage_bytes
        if reconstruct_rhs:
            final_frame_bytes = components * snapshot_cells * storage_bytes
            saved_wavefield_bytes = one_history + final_frame_bytes
        else:
            saved_wavefield_bytes = (1 + int(needs_backward)) * one_history
    update_coefficient_bytes = 6 * field_cells * float_bytes
    receiver_bytes = nstep * nt * nrx * float_bytes
    gradient_bytes = (int(er_requires_grad) + int(se_requires_grad)) * model_cells * float_bytes
    adjoint_state_bytes = field_bytes + pml_phi_bytes if needs_backward else 0
    receiver_adjoint_bytes = nstep * nt * nrx * float_bytes if needs_backward else 0
    # The deterministic CPU adjoint keeps one weight buffer per transposed-curl
    # channel (six field-sized arrays per shot) during backward.
    adjoint_stencil_bytes = (
        CPU_ADJOINT_WEIGHT_CHANNELS * nstep * field_cells * float_bytes
        if needs_backward and device.type != "cuda"
        else 0
    )
    exact_old_bytes = (
        components * snapshot_cells * float_bytes
        if save_wavefield_history
        and needs_backward
        and not reconstruct_rhs
        and (storage_dtype != torch.float32 or wavefield_compression == "int8")
        else 0
    )
    effective_async = bool(use_async_offload and device.type == "cuda")

    if device.type == "cuda" and effective_async:
        # Two E staging slots, plus two R slots unless R is rebuilt from E.
        stored_r = int(needs_backward and not reconstruct_rhs)
        forward_transfer_bytes = (2 + 2 * stored_r) * components * snapshot_cells * storage_bytes
        backward_transfer_bytes = (
            2 * int(needs_backward) * components * snapshot_cells * storage_bytes
        )
        forward_device_peak = (
            core_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + exact_old_bytes
            + forward_transfer_bytes
            + final_frame_bytes
        )
        backward_device_peak = (
            core_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + gradient_bytes
            + adjoint_state_bytes
            + receiver_adjoint_bytes
            + backward_transfer_bytes
            + final_frame_bytes
        )
        device_peak_bytes = max(forward_device_peak, backward_device_peak)
        host_peak_bytes = saved_wavefield_bytes - final_frame_bytes
        transfer_buffer_bytes = max(forward_transfer_bytes, backward_transfer_bytes)
    else:
        transfer_buffer_bytes = 0
        forward_peak = (
            core_bytes
            + saved_wavefield_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + exact_old_bytes
        )
        backward_peak = (
            core_bytes
            + saved_wavefield_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + gradient_bytes
            + adjoint_state_bytes
            + receiver_adjoint_bytes
            + adjoint_stencil_bytes
        )
        peak_bytes = max(forward_peak, backward_peak)
        if device.type == "cuda":
            device_peak_bytes, host_peak_bytes = peak_bytes, 0
        else:
            device_peak_bytes, host_peak_bytes = 0, peak_bytes

    return {
        "model_and_padded_materials": model_bytes,
        "electric_and_magnetic_fields": field_bytes,
        "cpml_auxiliary_fields": pml_phi_bytes,
        "cpml_coefficients": pml_coefficient_bytes,
        "source_and_locations": acquisition_bytes,
        "saved_gradient_wavefields": saved_wavefield_bytes,
        "final_e_frame": final_frame_bytes,
        "packed_history_per_quantity": packed_history_bytes,
        "fdtd_update_coefficients": update_coefficient_bytes,
        "receiver_data": receiver_bytes,
        "material_gradients": gradient_bytes,
        "adjoint_fields_and_cpml": adjoint_state_bytes,
        "receiver_adjoint": receiver_adjoint_bytes,
        "adjoint_stencil_weights": adjoint_stencil_bytes,
        "low_precision_exact_snapshot": exact_old_bytes,
        "cuda_transfer_buffers": transfer_buffer_bytes,
        "estimated_device_peak": device_peak_bytes,
        "estimated_host_peak": host_peak_bytes,
        "recommended_device_capacity": int(device_peak_bytes * MEMORY_SAFETY_MARGIN),
        "recommended_host_capacity": int(host_peak_bytes * MEMORY_SAFETY_MARGIN),
        "nt_saved": nt_saved,
        "components_saved": components,
        "history_shape": history_cells,
        "reconstruct_rhs": reconstruct_rhs,
        "effective_async_offload": effective_async,
    }


_BREAKDOWN_LABELS = (
    ("model and padded materials", "model_and_padded_materials"),
    ("electric and magnetic fields", "electric_and_magnetic_fields"),
    ("CPML auxiliary fields", "cpml_auxiliary_fields"),
    ("CPML coefficients", "cpml_coefficients"),
    ("source and acquisition locations", "source_and_locations"),
    ("saved E_saved and R_saved wavefields", "saved_gradient_wavefields"),
    ("  of which final E frame (E-only history)", "final_e_frame"),
    ("FDTD update coefficients", "fdtd_update_coefficients"),
    ("receiver data", "receiver_data"),
    ("material gradients", "material_gradients"),
    ("adjoint fields and CPML", "adjoint_fields_and_cpml"),
    ("receiver adjoint", "receiver_adjoint"),
    ("CPU adjoint stencil weights", "adjoint_stencil_weights"),
    ("low-precision exact snapshot", "low_precision_exact_snapshot"),
    ("CUDA transfer buffers", "cuda_transfer_buffers"),
)


def format_compute_preview(
    *,
    device: torch.device,
    grid_spacing: Sequence[float],
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    nt: int,
    nstep: int,
    nsr: int,
    nrx: int,
    source_amplitudes: torch.Tensor,
    source_location: torch.Tensor,
    receiver_location: torch.Tensor,
    er: torch.Tensor,
    se: torch.Tensor,
    mr: torch.Tensor,
    mr_supplied: bool,
    pmlthick: torch.Tensor,
    source_direction: int,
    receiver_component: int,
    model_gradient_sampling_interval: int,
    wavefield_storage_dtype: torch.dtype,
    wavefield_conversion_backend: str,
    int8_reduction_backend: str,
    wavefield_compression: str,
    wavefield_compression_block_size: Optional[BlockSize],
    save_wavefield_history: bool,
    use_async_offload: bool,
    fdtd_order: int,
    mode: int,
    debug: bool,
    save_forward_wavefield_path: Any,
    reconstruct_rhs: bool = False,
    history_region: str = "extended",
    E: Any,
    H: Any,
    PML: Any,
) -> str:
    """Return the pre-flight configuration and memory report as text.

    The arguments mirror the resolved state inside :func:`compute`
    (``nx``/``ny``/``nz`` include the CPML, ``mr`` carries the Yee halo).
    """
    pml = [int(value) for value in pmlthick.cpu().tolist()]
    estimate = estimate_compute_memory(
        device=device,
        nx=nx,
        ny=ny,
        nz=nz,
        nt=nt,
        nstep=nstep,
        nsr=nsr,
        nrx=nrx,
        source_waveforms=source_amplitudes.shape[0],
        pml=pml,
        mode=mode,
        sampling_interval=model_gradient_sampling_interval,
        storage_dtype=wavefield_storage_dtype,
        wavefield_compression=wavefield_compression,
        compression_block_size=wavefield_compression_block_size,
        save_wavefield_history=save_wavefield_history,
        use_async_offload=use_async_offload,
        er_requires_grad=er.requires_grad,
        se_requires_grad=se.requires_grad,
        reconstruct_rhs=reconstruct_rhs,
        history_region=history_region,
    )
    er_min = float(er.detach().amin().item())
    er_max = float(er.detach().amax().item())
    se_min = float(se.detach().amin().item())
    se_max = float(se.detach().amax().item())
    physical_mr = mr[:nx, :ny, :nz]
    mr_min = float(physical_mr.detach().amin().item())
    mr_max = float(physical_mr.detach().amax().item())
    source_min = float(source_amplitudes.detach().amin().item())
    source_max = float(source_amplitudes.detach().amax().item())
    model_shape = tuple(
        size - pml[2 * axis] - pml[2 * axis + 1] for axis, size in enumerate((nx, ny, nz))
    )
    dx, dy, dz = grid_spacing

    lines: List[str] = [
        "",
        "=== DeepGPR compute preview ===",
        "Simulation",
        f"  device: {device}",
        f"  model shape: {model_shape}",
        f"  computational shape (including PML): ({nx}, {ny}, {nz})",
        f"  padded field shape: ({nx + 1}, {ny + 1}, {nz + 1})",
        f"  shots / sources per shot / receivers per shot: {nstep} / {nsr} / {nrx}",
        f"  time steps: {nt}",
        f"  dx / dy / dz: {dx:.6e} m / {dy:.6e} m / {dz:.6e} m",
        f"  dt / simulated duration: {dt:.6e} s / {nt * dt:.6e} s",
        f"  FDTD order / gradient mode: {fdtd_order} / {mode}",
        f"  source / receiver component: {source_direction} / {receiver_component}",
        f"  PML thickness [x0, x1, y0, y1, z0, z1]: {pml}",
        "Model",
        f"  eps_r range / requires_grad: [{er_min:.6e}, {er_max:.6e}] / {er.requires_grad}",
        f"  sigma range / requires_grad: [{se_min:.6e}, {se_max:.6e}] / {se.requires_grad}",
        f"  mu_r range / supplied: [{mr_min:.6e}, {mr_max:.6e}] / {mr_supplied}",
        f"  propagation dtype: {er.dtype}",
        "Wavefield and runtime options",
        f"  save wavefield history: {save_wavefield_history}",
        "  gradient sampling interval / saved time steps: "
        f"{model_gradient_sampling_interval} / {estimate['nt_saved']}",
        f"  saved components / compression: {estimate['components_saved']} / {wavefield_compression}",
        f"  history region / frame cells: {history_region} / {estimate['history_shape']}",
        "  R history: "
        + (
            "rebuilt from consecutive E frames (E-only history)"
            if estimate["reconstruct_rhs"]
            else (
                "stored with E"
                if save_wavefield_history and (er.requires_grad or se.requires_grad)
                else "not stored (no material gradient)"
            )
        ),
    ]
    if wavefield_compression == "int8":
        lines += [
            "  INT8 block / packed bytes per E or R history: "
            f"{wavefield_compression_block_size} / "
            f"{format_memory_size(estimate['packed_history_per_quantity'])}",
            f"  INT8 reduction backend: {int8_reduction_backend}",
        ]
    else:
        lines += [
            f"  saved wavefield storage dtype: {wavefield_storage_dtype}",
            f"  CUDA conversion backend: {wavefield_conversion_backend}",
        ]
    save_directory = (
        save_forward_wavefield_path if save_forward_wavefield_path is not None else "disabled"
    )
    lines += [
        f"  async offload requested / effective: {bool(use_async_offload)} / "
        f"{estimate['effective_async_offload']}",
        f"  debug validation: {bool(debug)}",
        "  print parameters: True",
        f"  forward wavefield save directory: {save_directory}",
        f"  initial E / H / PML supplied: {E is not None} / {H is not None} / {PML is not None}",
        "Input tensors",
        f"  source amplitudes shape / range: {tuple(source_amplitudes.shape)} / "
        f"[{source_min:.6e}, {source_max:.6e}]",
        f"  source locations shape: {tuple(source_location.shape)}",
        f"  receiver locations shape: {tuple(receiver_location.shape)}",
        "Estimated tensor payload",
    ]
    lines += [f"  {label}: {format_memory_size(estimate[key])}" for label, key in _BREAKDOWN_LABELS]

    if device.type == "cuda":
        lines += [
            "  estimated peak CUDA device memory: "
            f"{format_memory_size(estimate['estimated_device_peak'])}",
            "  recommended CUDA capacity with 20% margin: "
            f"{format_memory_size(estimate['recommended_device_capacity'])}",
        ]
        if estimate["estimated_host_peak"]:
            lines += [
                "  estimated pinned host memory: "
                f"{format_memory_size(estimate['estimated_host_peak'])}",
                "  recommended host capacity with 20% margin: "
                f"{format_memory_size(estimate['recommended_host_capacity'])}",
            ]
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            lines.append(
                "  currently free / total CUDA memory: "
                f"{format_memory_size(free_bytes)} / {format_memory_size(total_bytes)}"
            )
        except (RuntimeError, TypeError):
            pass
    else:
        lines += [
            f"  estimated peak CPU memory: {format_memory_size(estimate['estimated_host_peak'])}",
            "  recommended CPU capacity with 20% margin: "
            f"{format_memory_size(estimate['recommended_host_capacity'])}",
        ]
    lines += [
        "  note: estimates exclude the CUDA context, PyTorch allocator cache, autograd "
        "metadata, Python objects, and other tensors owned by the calling program.",
        "=== End DeepGPR compute preview ===",
        "",
    ]
    return "\n".join(lines)


__all__ = ["estimate_compute_memory", "format_compute_preview"]
