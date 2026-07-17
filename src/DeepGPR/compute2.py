import torch
import ctypes
from . import get_deepgpr_lib, set_library_fdtd_order
from .common import (
    _normalize_grid_spacing,
    initialization,
    build_pml_phi,
    create_or_separate,
    buildpmlcoeffs,
    check_tensors_for_nan_inf,
)


_WAVEFIELD_STORAGE_TYPES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _format_memory_size(num_bytes):
    """Format a byte count using binary memory units."""
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _pml_phi_elements(nx, ny, nz, nstep, pml):
    """Return the exact number of float32 CPML auxiliary elements."""
    total = 0
    for thickness in pml[:2]:
        if thickness > 0:
            total += nstep * (
                (thickness + 1) * ny * (nz + 1)
                + (thickness + 1) * (ny + 1) * nz
                + thickness * (ny + 1) * nz
                + thickness * ny * (nz + 1)
            )
    for thickness in pml[2:4]:
        if thickness > 0:
            total += nstep * (
                nx * (thickness + 1) * (nz + 1)
                + (nx + 1) * (thickness + 1) * nz
                + (nx + 1) * thickness * nz
                + nx * thickness * (nz + 1)
            )
    for thickness in pml[4:6]:
        if thickness > 0:
            total += nstep * (
                nx * (ny + 1) * (thickness + 1)
                + (nx + 1) * ny * (thickness + 1)
                + (nx + 1) * ny * thickness
                + nx * (ny + 1) * thickness
            )
    return total


def _estimate_compute_memory(
    *,
    device,
    nx,
    ny,
    nz,
    nt,
    nstep,
    nsr,
    nrx,
    source_waveforms,
    pml,
    mode,
    sampling_interval,
    storage_dtype,
    use_async_offload,
    er_requires_grad,
    se_requires_grad,
):
    """Estimate tensor payload memory for one compute call."""
    float_bytes = torch.tensor([], dtype=torch.float32).element_size()
    int_bytes = torch.tensor([], dtype=torch.int32).element_size()
    storage_bytes = torch.tensor([], dtype=storage_dtype).element_size()
    model_cells = nx * ny * nz
    field_cells = (nx + 1) * (ny + 1) * (nz + 1)
    snapshot_cells = nstep * model_cells
    components = 3 if mode == 3 else 1
    nt_saved = (nt + sampling_interval - 1) // sampling_interval

    model_bytes = (2 * model_cells + 3 * field_cells) * float_bytes
    field_bytes = 6 * nstep * field_cells * float_bytes
    pml_phi_bytes = _pml_phi_elements(nx, ny, nz, nstep, pml) * float_bytes
    pml_coefficient_bytes = (
        8 * sum(pml) * float_bytes
        + 7 * sum(thickness > 0 for thickness in pml) * int_bytes
    )
    acquisition_bytes = (
        source_waveforms * nt * float_bytes
        + nstep * nsr * 3 * int_bytes
        + nstep * nrx * 3 * int_bytes
    )
    core_bytes = (
        model_bytes
        + field_bytes
        + pml_phi_bytes
        + pml_coefficient_bytes
        + acquisition_bytes
    )

    saved_wavefield_bytes = (
        2 * components * nt_saved * snapshot_cells * storage_bytes
    )
    update_coefficient_bytes = 6 * field_cells * float_bytes
    receiver_bytes = nstep * 6 * nt * nrx * float_bytes
    gradient_bytes = (
        int(er_requires_grad) + int(se_requires_grad)
    ) * model_cells * float_bytes
    needs_backward = er_requires_grad or se_requires_grad
    adjoint_state_bytes = (
        field_bytes + pml_phi_bytes if needs_backward else 0
    )
    receiver_adjoint_bytes = (
        nstep * nt * nrx * float_bytes if needs_backward else 0
    )
    exact_old_bytes = (
        components * snapshot_cells * float_bytes
        if storage_dtype != torch.float32
        else 0
    )
    effective_async = bool(use_async_offload and device.type == "cuda")

    if device.type == "cuda" and effective_async:
        forward_transfer_bytes = 4 * components * snapshot_cells * storage_bytes
        backward_transfer_bytes = 2 * components * snapshot_cells * storage_bytes
        forward_device_peak = (
            core_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + exact_old_bytes
            + forward_transfer_bytes
        )
        backward_device_peak = (
            core_bytes
            + update_coefficient_bytes
            + receiver_bytes
            + gradient_bytes
            + adjoint_state_bytes
            + receiver_adjoint_bytes
            + backward_transfer_bytes
        )
        device_peak_bytes = max(forward_device_peak, backward_device_peak)
        host_peak_bytes = saved_wavefield_bytes
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
        )
        peak_bytes = max(forward_peak, backward_peak)
        if device.type == "cuda":
            device_peak_bytes = peak_bytes
            host_peak_bytes = 0
        else:
            device_peak_bytes = 0
            host_peak_bytes = peak_bytes

    return {
        "model_and_padded_materials": model_bytes,
        "electric_and_magnetic_fields": field_bytes,
        "cpml_auxiliary_fields": pml_phi_bytes,
        "cpml_coefficients": pml_coefficient_bytes,
        "source_and_locations": acquisition_bytes,
        "saved_gradient_wavefields": saved_wavefield_bytes,
        "fdtd_update_coefficients": update_coefficient_bytes,
        "six_component_receiver_buffer": receiver_bytes,
        "material_gradients": gradient_bytes,
        "adjoint_fields_and_cpml": adjoint_state_bytes,
        "receiver_adjoint": receiver_adjoint_bytes,
        "low_precision_exact_snapshot": exact_old_bytes,
        "cuda_transfer_buffers": transfer_buffer_bytes,
        "estimated_device_peak": device_peak_bytes,
        "estimated_host_peak": host_peak_bytes,
        "recommended_device_capacity": int(device_peak_bytes * 1.20),
        "recommended_host_capacity": int(host_peak_bytes * 1.20),
        "nt_saved": nt_saved,
        "components_saved": components,
        "effective_async_offload": effective_async,
    }


def _print_compute_preview(
    *,
    device,
    grid_spacing,
    dt,
    nx,
    ny,
    nz,
    nt,
    nstep,
    nsr,
    nrx,
    source_amplitudes,
    source_location,
    receiver_location,
    er,
    se,
    mr,
    mr_supplied,
    pmlthick,
    source_direction,
    reciever_direction,
    model_gradient_sampling_interval,
    wavefield_storage_dtype,
    use_async_offload,
    fdtd_order,
    mode,
    debug,
    E,
    H,
    PML,
):
    """Print a preflight configuration and memory estimate."""
    pml = [int(value) for value in pmlthick.cpu().tolist()]
    estimate = _estimate_compute_memory(
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
        use_async_offload=use_async_offload,
        er_requires_grad=er.requires_grad,
        se_requires_grad=se.requires_grad,
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

    print("\n=== DeepGPR compute preview ===")
    print("Simulation")
    print(f"  device: {device}")
    print(f"  model shape: ({nx}, {ny}, {nz})")
    print(f"  padded field shape: ({nx + 1}, {ny + 1}, {nz + 1})")
    print(f"  shots / sources per shot / receivers per shot: {nstep} / {nsr} / {nrx}")
    print(f"  time steps: {nt}")
    dx, dy, dz = grid_spacing
    print(
        "  dx / dy / dz: "
        f"{dx:.6e} m / {dy:.6e} m / {dz:.6e} m"
    )
    print(f"  dt / simulated duration: {dt:.6e} s / {nt * dt:.6e} s")
    print(f"  FDTD order / gradient mode: {fdtd_order} / {mode}")
    print(f"  source / receiver direction: {source_direction} / {reciever_direction}")
    print(f"  PML thickness [x0, x1, y0, y1, z0, z1]: {pml}")
    print("Model")
    print(
        f"  er range / requires_grad: [{er_min:.6e}, {er_max:.6e}] "
        f"/ {er.requires_grad}"
    )
    print(
        f"  se range / requires_grad: [{se_min:.6e}, {se_max:.6e}] "
        f"/ {se.requires_grad}"
    )
    print(f"  mr range / supplied: [{mr_min:.6e}, {mr_max:.6e}] / {mr_supplied}")
    print(f"  propagation dtype: {er.dtype}")
    print("Wavefield and runtime options")
    print(
        "  gradient sampling interval / saved time steps: "
        f"{model_gradient_sampling_interval} / {estimate['nt_saved']}"
    )
    print(
        f"  saved components / storage dtype: {estimate['components_saved']} / "
        f"{wavefield_storage_dtype}"
    )
    print(
        f"  async offload requested / effective: {bool(use_async_offload)} / "
        f"{estimate['effective_async_offload']}"
    )
    print(f"  debug validation: {bool(debug)}")
    print("  print parameters: True")
    print(
        f"  initial E / H / PML supplied: {E is not None} / "
        f"{H is not None} / {PML is not None}"
    )
    print("Input tensors")
    print(
        f"  source amplitudes shape / range: {tuple(source_amplitudes.shape)} / "
        f"[{source_min:.6e}, {source_max:.6e}]"
    )
    print(f"  source locations shape: {tuple(source_location.shape)}")
    print(f"  receiver locations shape: {tuple(receiver_location.shape)}")
    print("Estimated tensor payload")
    breakdown_labels = (
        ("model and padded materials", "model_and_padded_materials"),
        ("electric and magnetic fields", "electric_and_magnetic_fields"),
        ("CPML auxiliary fields", "cpml_auxiliary_fields"),
        ("CPML coefficients", "cpml_coefficients"),
        ("source and acquisition locations", "source_and_locations"),
        ("saved Eall and Rall wavefields", "saved_gradient_wavefields"),
        ("FDTD update coefficients", "fdtd_update_coefficients"),
        ("six-component receiver buffer", "six_component_receiver_buffer"),
        ("material gradients", "material_gradients"),
        ("adjoint fields and CPML", "adjoint_fields_and_cpml"),
        ("receiver adjoint", "receiver_adjoint"),
        ("low-precision exact snapshot", "low_precision_exact_snapshot"),
        ("CUDA transfer buffers", "cuda_transfer_buffers"),
    )
    for label, key in breakdown_labels:
        print(f"  {label}: {_format_memory_size(estimate[key])}")

    if device.type == "cuda":
        print(
            "  estimated peak CUDA device memory: "
            f"{_format_memory_size(estimate['estimated_device_peak'])}"
        )
        print(
            "  recommended CUDA capacity with 20% margin: "
            f"{_format_memory_size(estimate['recommended_device_capacity'])}"
        )
        if estimate["estimated_host_peak"]:
            print(
                "  estimated pinned host memory: "
                f"{_format_memory_size(estimate['estimated_host_peak'])}"
            )
            print(
                "  recommended host capacity with 20% margin: "
                f"{_format_memory_size(estimate['recommended_host_capacity'])}"
            )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            print(
                "  currently free / total CUDA memory: "
                f"{_format_memory_size(free_bytes)} / {_format_memory_size(total_bytes)}"
            )
        except (RuntimeError, TypeError):
            pass
    else:
        print(
            "  estimated peak CPU memory: "
            f"{_format_memory_size(estimate['estimated_host_peak'])}"
        )
        print(
            "  recommended CPU capacity with 20% margin: "
            f"{_format_memory_size(estimate['recommended_host_capacity'])}"
        )
    print(
        "  note: estimates exclude the CUDA context, PyTorch allocator cache, autograd "
        "metadata, Python objects, and other tensors owned by the calling program."
    )
    print("=== End DeepGPR compute preview ===\n")


def _normalize_wavefield_storage_dtype(value):
    """Return a supported torch dtype for saved model-gradient wavefields."""
    if isinstance(value, str):
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        value = aliases.get(value.lower())
    if value not in _WAVEFIELD_STORAGE_TYPES:
        raise ValueError(
            "wavefield_storage_dtype must be float32, float16, or bfloat16."
        )
    return value


def _sync_cuda_device(device):
    """Synchronize a CUDA device when the selected backend is CUDA.

    Args:
        device: PyTorch device object to synchronize.
    """
    if device.type != "cuda":
        return

    if device.index is None:
        torch.cuda.synchronize()
        return

    torch.cuda.set_device(device.index)
    torch.cuda.synchronize(device.index)


def _check_nonzero_source_created_fields(c_lib, source_amplitudes, *fields):
    """Check that a nonzero source creates at least one nonzero field.

    Args:
        c_lib: Loaded native DeepGPR library.
        source_amplitudes: Source waveform tensor.
        *fields: Field tensors returned by the native backend.
    """
    if source_amplitudes.abs().max().item() == 0.0:
        return

    max_field = 0.0
    for field in fields:
        max_field = max(max_field, field.abs().max().item())

    if max_field == 0.0:
        lib_path = getattr(c_lib, "_deepgpr_path", "unknown")
        raise RuntimeError(
            "The native DeepGPR backend returned all-zero fields even though the "
            f"source waveform is nonzero. Loaded library: {lib_path}. "
            "This usually means the shared library is stale or incompatible with "
            "the current source code/device. Rebuild or pull the latest generated "
            "native libraries."
        )

def compute(device, dx=None, dt=None, 
            source_amplitudes=None,
            source_location=None, 
            receiver_location=None, 
            er=None, se=None,mr=None, 
            E=None,H=None,
            PML=None,
            pmlthick=10, source_direction=2, reciever_direction=2,
            model_gradient_sampling_interval=1,
            wavefield_storage_dtype=torch.float32,
            use_async_offload=False,
            fdtd_order=2,
            mode=2,
            debug=False,
            print_parameters=False):
    """Run DeepGPR FDTD forward modeling with autograd support.

    Args:
        device: PyTorch device or device string, such as "cpu" or "cuda".
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step size.
        source_amplitudes: Source waveform tensor with shape (nwaveforms, nt, 1).
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        er: Relative permittivity tensor with shape (nx, ny) or (nx, ny, nz).
        se: Electrical conductivity tensor with the same shape as er.
        mr: Relative permeability tensor, or None to use ones.
        E: Optional initial electric field tuple (Ex, Ey, Ez).
        H: Optional initial magnetic field tuple (Hx, Hy, Hz).
        PML: Optional tuple of 24 PML auxiliary tensors.
        pmlthick: PML thickness as an int, list, or tensor.
        source_direction: Source electric-field polarization, 0 for x, 1 for y, 2 for z.
        reciever_direction: Receiver component to return, 0 for x, 1 for y, 2 for z.
        model_gradient_sampling_interval: Forward wavefield sampling interval for FWI gradients.
        wavefield_storage_dtype: Saved E/R wavefield dtype: float32, float16, or bfloat16.
        use_async_offload: Whether CUDA should offload saved wavefields to pinned CPU memory.
        fdtd_order: Spatial finite-difference order, supported values are 2, 4, and 8.
        mode: FWI gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
        debug: Whether to run expensive tensor validation checks.
        print_parameters: Whether to print a preflight parameter and memory preview.
    """
    device = torch.device(device)
    if fdtd_order not in (2, 4, 8):
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")
    if mode not in (2, 3):
        raise ValueError("mode must be 2 or 3.")
    if not isinstance(print_parameters, bool):
        raise TypeError("print_parameters must be a bool.")
    if not isinstance(model_gradient_sampling_interval, int) or model_gradient_sampling_interval < 1:
        raise ValueError("model_gradient_sampling_interval must be a positive integer.")
    wavefield_storage_dtype = _normalize_wavefield_storage_dtype(wavefield_storage_dtype)
    if source_direction not in (0, 1, 2) or reciever_direction not in (0, 1, 2):
        raise ValueError("source_direction and reciever_direction must be 0, 1, or 2.")
    if getattr(source_amplitudes, "requires_grad", False):
        raise NotImplementedError("DeepGPR does not currently return source-amplitude gradients.")
    if getattr(mr, "requires_grad", False):
        raise NotImplementedError("DeepGPR does not currently return relative-permeability gradients.")

    grid_spacing = _normalize_grid_spacing(dx)
    mr_supplied = mr is not None
    er,se,nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,spatial_mode,dtype,pmlthick,source_amplitudes=initialization(device,er,se,mr,source_amplitudes,source_location,receiver_location,grid_spacing,dt,pmlthick,fdtd_order)

    needs_model_gradient = er.requires_grad or se.requires_grad
    if needs_model_gradient and mode == 2:
        if spatial_mode != 2 or source_direction != 2 or reciever_direction != 2:
            raise ValueError(
                "mode=2 is an exact model-gradient mode only for 2D Ez-TM modeling. "
                "Use mode=3 for 3D or other electric-field components."
            )

    if print_parameters:
        _print_compute_preview(
            device=device,
            grid_spacing=grid_spacing,
            dt=float(dt),
            nx=nx,
            ny=ny,
            nz=nz,
            nt=nt,
            nstep=nstep,
            nsr=nsr,
            nrx=nrx,
            source_amplitudes=source_amplitudes,
            source_location=source_location,
            receiver_location=receiver_location,
            er=er,
            se=se,
            mr=mr,
            mr_supplied=mr_supplied,
            pmlthick=pmlthick,
            source_direction=source_direction,
            reciever_direction=reciever_direction,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            wavefield_storage_dtype=wavefield_storage_dtype,
            use_async_offload=use_async_offload,
            fdtd_order=fdtd_order,
            mode=mode,
            debug=debug,
            E=E,
            H=H,
            PML=PML,
        )

    Ex,Ey,Ez=create_or_separate(E,nx,ny,nz,nstep,device,dtype)
    Hx,Hy,Hz=create_or_separate(H,nx,ny,nz,nstep,device,dtype)

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=buildpmlcoeffs(er,mr,dt,grid_spacing,nx,ny,nz,pmlthick,device,dtype)

    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,Eall,receiver_amplitudes = DeepGPR.apply(
        er, se,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mr,grid_spacing,nx,ny,nz,dt,nt,nstep,source_amplitudes,source_location,receiver_location,pmlthick,nsr,nrx,device,dtype,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see,source_direction, reciever_direction, model_gradient_sampling_interval, wavefield_storage_dtype, use_async_offload, fdtd_order, mode, debug)

    return Eall,(Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2),receiver_amplitudes


class DeepGPR(torch.autograd.Function):
    """PyTorch autograd bridge for the native DeepGPR backends."""

    @staticmethod
    def forward(ctx, er, se,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mr,grid_spacing,nx,ny,nz,dt,
                nt,nstep,source_amplitudes,source_location,receiver_location,
                pmlthick,nsr,nrx,device,dtype,x0,xm,
                y0,ym,z0,zm,x01,x02,xm1,xm2,
                y01,y02,ym1,ym2,z01,z02,zm1,zm2,
                ere,see,source_direction, reciever_direction, 
                model_gradient_sampling_interval, wavefield_storage_dtype, use_async_offload, fdtd_order, mode, debug):
        """Call the native forward solver and save tensors for backward.

        Args:
            ctx: PyTorch autograd context.
            er: Trainable relative permittivity tensor.
            se: Trainable electrical conductivity tensor.
            Ex, Ey, Ez: Electric field component tensors.
            Hx, Hy, Hz: Magnetic field component tensors.
            x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2: Low-x PML auxiliary tensors.
            xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2: High-x PML auxiliary tensors.
            y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2: Low-y PML auxiliary tensors.
            ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2: High-y PML auxiliary tensors.
            z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2: Low-z PML auxiliary tensors.
            zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2: High-z PML auxiliary tensors.
            mr: Padded relative permeability tensor.
            grid_spacing: Three-value ``(dx, dy, dz)`` grid spacing tuple.
            nx: Number of model cells along the x axis.
            ny: Number of model cells along the y axis.
            nz: Number of model cells along the z axis.
            dt: Time step size.
            nt: Number of time steps.
            nstep: Number of shots or simulations in the batch.
            source_amplitudes: Source waveform tensor.
            source_location: Source coordinates with shape (nstep, nsr, 3).
            receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
            pmlthick: Six-boundary PML thickness tensor.
            nsr: Number of sources per shot.
            nrx: Number of receivers per shot.
            device: PyTorch device used by the solver.
            dtype: PyTorch dtype used by allocated tensors.
            x0, xm, y0, ym, z0, zm: PML boundary descriptor tensors.
            x01, x02, xm1, xm2: X-boundary PML coefficient tensors.
            y01, y02, ym1, ym2: Y-boundary PML coefficient tensors.
            z01, z02, zm1, zm2: Z-boundary PML coefficient tensors.
            ere: Padded relative permittivity tensor.
            see: Padded electrical conductivity tensor.
            source_direction: Source electric-field polarization, 0 for x, 1 for y, 2 for z.
            reciever_direction: Receiver component to return, 0 for x, 1 for y, 2 for z.
            model_gradient_sampling_interval: Forward wavefield sampling interval.
            wavefield_storage_dtype: Dtype used by saved E/R wavefield buffers.
            use_async_offload: Whether CUDA should offload saved wavefields to pinned CPU memory.
            fdtd_order: Spatial finite-difference order.
            mode: FWI gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
            debug: Whether to run expensive tensor validation checks.
        """
        
        source_amplitudes = source_amplitudes.contiguous()
        source_location=source_location.to(torch.int32).contiguous()
        receiver_location=receiver_location.to(torch.int32).contiguous()
        c_lib = get_deepgpr_lib(device)
        set_library_fdtd_order(c_lib, fdtd_order)
        ctx.save_for_backward(er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see)
        dx, dy, dz = grid_spacing
        ctx.grid_spacing=grid_spacing
        ctx.nx=nx
        ctx.ny=ny
        ctx.nz=nz
        ctx.dt=dt
        ctx.nt=nt
        ctx.nrx=nrx
        ctx.nsr=nsr
        ctx.nstep=nstep
        ctx.pmlthick=pmlthick
        ctx.device=device
        ctx.dtype=dtype
        ctx.model_gradient_sampling_interval = model_gradient_sampling_interval
        ctx.wavefield_storage_type = _WAVEFIELD_STORAGE_TYPES[wavefield_storage_dtype]
        ctx.use_async_offload = bool(use_async_offload and device.type == "cuda")
        ctx.fdtd_order = fdtd_order
        ctx.mode = mode
        ctx.reciever_direction = reciever_direction
        ctx.debug = bool(debug)

        nt_saved = (nt + model_gradient_sampling_interval - 1) // model_gradient_sampling_interval
        e_components = 3 if mode == 3 else 1
        
        if mode == 3:
            eall_shape = (e_components, nt_saved, nstep, nx, ny, nz)
        else:
            eall_shape = (nt_saved, nstep, nx, ny, nz)

        if ctx.use_async_offload:
            Eall = torch.zeros(eall_shape, device='cpu', dtype=wavefield_storage_dtype).pin_memory()
            Rall = torch.zeros(eall_shape, device='cpu', dtype=wavefield_storage_dtype).pin_memory()
        else:
            Eall = torch.zeros(eall_shape, device=device, dtype=wavefield_storage_dtype).contiguous()
            Rall = torch.zeros(eall_shape, device=device, dtype=wavefield_storage_dtype).contiguous()

        Eupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)

        receiver_amplitudes = torch.zeros((nstep, 6, nt, nrx),device=device, dtype=dtype).contiguous()
        pml = [int(pmlthick[i]) for i in range(6)]

        _sync_cuda_device(device)
        c_lib.forward(
                ctypes.cast(ere.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(see.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(mr.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(Eall.data_ptr()),
                ctypes.c_void_p(Rall.data_ptr()),
                ctypes.cast(Ex.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ey.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ez.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hz.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(Eupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Eupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(x0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(x0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(x0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(x0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(xmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(xmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(y0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(y0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ymEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ymEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ymHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ymHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(z0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(z0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(z0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(z0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(zmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(zmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                pml[0],pml[1],pml[2],
                pml[3],pml[4],pml[5],

                ctypes.cast(x01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym1.data_ptr(), ctypes.POINTER(ctypes.c_float)),       
                ctypes.cast(z01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),                
                ctypes.cast(x02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(y02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym2.data_ptr(), ctypes.POINTER(ctypes.c_float)),         
                ctypes.cast(z02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                dt, nt, nstep, nrx, dx, dy, dz,
                ctypes.cast(receiver_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(receiver_amplitudes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                nx+1, ny+1, nz+1, nsr,
                ctypes.cast(source_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(source_amplitudes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                source_direction,
                model_gradient_sampling_interval,
                mode,
                ctx.wavefield_storage_type)
        _sync_cuda_device(device)

        if ctx.debug:
            check_tensors_for_nan_inf(d="forward",
                Ex=Ex, Ey=Ey, Ez=Ez,
                Hx=Hx, Hy=Hy, Hz=Hz,
                Eall=Eall, Rall=Rall,
                x0EPhi1=x0EPhi1, x0EPhi2=x0EPhi2,
                x0HPhi1=x0HPhi1, x0HPhi2=x0HPhi2,
                xmEPhi1=xmEPhi1, xmEPhi2=xmEPhi2,
                xmHPhi1=xmHPhi1, xmHPhi2=xmHPhi2,
                y0EPhi1=y0EPhi1, y0EPhi2=y0EPhi2,
                y0HPhi1=y0HPhi1, y0HPhi2=y0HPhi2,
                ymEPhi1=ymEPhi1, ymEPhi2=ymEPhi2,
                ymHPhi1=ymHPhi1, ymHPhi2=ymHPhi2,
                z0EPhi1=z0EPhi1, z0EPhi2=z0EPhi2,
                z0HPhi1=z0HPhi1, z0HPhi2=z0HPhi2,
                zmEPhi1=zmEPhi1, zmEPhi2=zmEPhi2,
                zmHPhi1=zmHPhi1, zmHPhi2=zmHPhi2
            )
            _check_nonzero_source_created_fields(c_lib, source_amplitudes, Ex, Ey, Ez, Hx, Hy, Hz)

        ctx.Eall = Eall
        ctx.Rall = Rall
        ctx.mark_non_differentiable(Eall)
        return (Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,Eall,receiver_amplitudes[:,reciever_direction,:,:])

    @staticmethod
    def backward(ctx,gEx,gEy,gEz,gHx,gHy,gHz,gx0EPhi1,gx0EPhi2,gx0HPhi1,gx0HPhi2,gxmEPhi1,gxmEPhi2,gxmHPhi1,gxmHPhi2,gy0EPhi1,gy0EPhi2,gy0HPhi1,gy0HPhi2,gymEPhi1,gymEPhi2,gymHPhi1,gymHPhi2,gz0EPhi1,gz0EPhi2,gz0HPhi1,gz0HPhi2,gzmEPhi1,gzmEPhi2,gzmHPhi1,gzmHPhi2,gEall,gezreciver):
        """Call the native adjoint solver and return gradients.

        Args:
            ctx: PyTorch autograd context saved by forward.
            gEx, gEy, gEz: Incoming gradients for electric field components.
            gHx, gHy, gHz: Incoming gradients for magnetic field components.
            gx0EPhi1, gx0EPhi2, gx0HPhi1, gx0HPhi2: Gradients for low-x PML tensors.
            gxmEPhi1, gxmEPhi2, gxmHPhi1, gxmHPhi2: Gradients for high-x PML tensors.
            gy0EPhi1, gy0EPhi2, gy0HPhi1, gy0HPhi2: Gradients for low-y PML tensors.
            gymEPhi1, gymEPhi2, gymHPhi1, gymHPhi2: Gradients for high-y PML tensors.
            gz0EPhi1, gz0EPhi2, gz0HPhi1, gz0HPhi2: Gradients for low-z PML tensors.
            gzmEPhi1, gzmEPhi2, gzmHPhi1, gzmHPhi2: Gradients for high-z PML tensors.
            gEall: Incoming gradient for the saved forward electric field history.
            gezreciver: Incoming gradient for receiver amplitudes.
        """
        
        sourceamp=gezreciver
        er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see=ctx.saved_tensors
        
        ere=ere.contiguous()
        see=see.contiguous()
        er=er.contiguous()
        se=se.contiguous()
        mr=mr.contiguous()
        receiver_location=receiver_location.contiguous()
        x0=x0.contiguous()
        xm=xm.contiguous()
        y0=y0.contiguous()
        ym=ym.contiguous()
        z0=z0.contiguous()
        zm=zm.contiguous()
        x01=x01.contiguous()
        x02=x02.contiguous()
        xm1=xm1.contiguous()
        xm2=xm2.contiguous()
        y01=y01.contiguous()
        y02=y02.contiguous()
        ym1=ym1.contiguous()
        ym2=ym2.contiguous()
        z01=z01.contiguous()
        z02=z02.contiguous()
        zm1=zm1.contiguous()
        zm2=zm2.contiguous()

        dx, dy, dz=ctx.grid_spacing
        nx=ctx.nx
        ny=ctx.ny
        nz=ctx.nz
        dt=ctx.dt
        nt=ctx.nt
        nsr=ctx.nrx
        nrx=ctx.nsr
        dtype=ctx.dtype
        nstep=ctx.nstep
        pmlthick=ctx.pmlthick
        device=ctx.device
        Eall=ctx.Eall
        Rall=ctx.Rall
        model_gradient_sampling_interval = ctx.model_gradient_sampling_interval
        c_lib = get_deepgpr_lib(device)
        set_library_fdtd_order(c_lib, ctx.fdtd_order)

        Eall=Eall.contiguous()
        Rall=Rall.contiguous()
        gEx=gEx.contiguous()
        gEy=gEy.contiguous()
        gEz=gEz.contiguous()
        gHx=gHx.contiguous()
        gHy=gHy.contiguous()
        gHz=gHz.contiguous()
        gx0EPhi1=gx0EPhi1.contiguous()
        gx0EPhi2=gx0EPhi2.contiguous()
        gx0HPhi1=gx0HPhi1.contiguous()
        gx0HPhi2=gx0HPhi2.contiguous()
        gxmEPhi1=gxmEPhi1.contiguous()
        gxmEPhi2=gxmEPhi2.contiguous()
        gxmHPhi1=gxmHPhi1.contiguous()
        gxmHPhi2=gxmHPhi2.contiguous()
        gy0EPhi1=gy0EPhi1.contiguous()
        gy0EPhi2=gy0EPhi2.contiguous()
        gy0HPhi1=gy0HPhi1.contiguous()
        gy0HPhi2=gy0HPhi2.contiguous()
        gymEPhi1=gymEPhi1.contiguous()
        gymEPhi2=gymEPhi2.contiguous()
        gymHPhi1=gymHPhi1.contiguous()
        gymHPhi2=gymHPhi2.contiguous()
        gz0EPhi1=gz0EPhi1.contiguous()
        gz0EPhi2=gz0EPhi2.contiguous()
        gz0HPhi1=gz0HPhi1.contiguous()
        gz0HPhi2=gz0HPhi2.contiguous()
        gzmEPhi1=gzmEPhi1.contiguous()
        gzmEPhi2=gzmEPhi2.contiguous()
        gzmHPhi1=gzmHPhi1.contiguous()
        gzmHPhi2=gzmHPhi2.contiguous()
        sourceamp=sourceamp.contiguous()

        Eupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)

        if er.requires_grad:
            grad_er=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            errequiregrad=1
        else:
            grad_er=torch.empty(0)
            errequiregrad=0

        if se.requires_grad: 
            grad_se=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            serequiregrad=1
        else:
            grad_se=torch.empty(0)
            serequiregrad=0

        pml = [int(pmlthick[i]) for i in range(6)]

        _sync_cuda_device(device)
        c_lib.backward(
                ctypes.cast(ere.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(see.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(mr.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(Eall.data_ptr()),
                ctypes.c_void_p(Rall.data_ptr()),
                ctypes.cast(gEx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gEy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gEz.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHz.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                ctypes.cast(Eupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Eupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                ctypes.cast(gx0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gx0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gx0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gx0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gxmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gxmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gxmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gxmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gy0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gy0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gy0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gy0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gymEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gymEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gymHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gymHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gz0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gz0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gz0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gz0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gzmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gzmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gzmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gzmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                pml[0],pml[1],pml[2],
                pml[3],pml[4],pml[5],

                ctypes.cast(x01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym1.data_ptr(), ctypes.POINTER(ctypes.c_float)),       
                ctypes.cast(z01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),                
                ctypes.cast(x02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(y02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym2.data_ptr(), ctypes.POINTER(ctypes.c_float)),         
                ctypes.cast(z02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm2.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                dt, nt, nstep, nrx, dx, dy, dz,
                nx+1, ny+1, nz+1, nsr,
                ctypes.cast(receiver_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(sourceamp.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctx.reciever_direction,
                ctypes.cast(grad_er.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(grad_se.data_ptr(), ctypes.POINTER(ctypes.c_float)),errequiregrad,serequiregrad,
                model_gradient_sampling_interval,
                ctx.mode,
                ctx.wavefield_storage_type)
        _sync_cuda_device(device)
        
        tensors_to_check = dict(
            gEx=gEx, gEy=gEy, gEz=gEz,
            gHx=gHx, gHy=gHy, gHz=gHz,
            gx0EPhi1=gx0EPhi1, gx0EPhi2=gx0EPhi2,
            gx0HPhi1=gx0HPhi1, gx0HPhi2=gx0HPhi2,
            gxmEPhi1=gxmEPhi1, gxmEPhi2=gxmEPhi2,
            gxmHPhi1=gxmHPhi1, gxmHPhi2=gxmHPhi2,
            gy0EPhi1=gy0EPhi1, gy0EPhi2=gy0EPhi2,
            gy0HPhi1=gy0HPhi1, gy0HPhi2=gy0HPhi2,
            gymEPhi1=gymEPhi1, gymEPhi2=gymEPhi2,
            gymHPhi1=gymHPhi1, gymHPhi2=gymHPhi2,
            gz0EPhi1=gz0EPhi1, gz0EPhi2=gz0EPhi2,
            gz0HPhi1=gz0HPhi1, gz0HPhi2=gz0HPhi2,
            gzmEPhi1=gzmEPhi1, gzmEPhi2=gzmEPhi2,
            gzmHPhi1=gzmHPhi1, gzmHPhi2=gzmHPhi2
        )
        if errequiregrad == 1:
            tensors_to_check["grad_er"] = grad_er
        if serequiregrad == 1:
            tensors_to_check["grad_se"] = grad_se

        if ctx.debug:
            check_tensors_for_nan_inf(d="backward", **tensors_to_check)

        ctx.Eall = None
        ctx.Rall = None
        del Eall,Rall,er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see, Eupdatecoffs0, Eupdatecoffs1, Eupdatecoffs4, Hupdatecoffs0, Hupdatecoffs1, Hupdatecoffs4

        return (
                    grad_er, grad_se,         
                    gEx,gEy,gEz, gHx,gHy,gHz, 
                    gx0EPhi1,gx0EPhi2,gx0HPhi1,gx0HPhi2,
                    gxmEPhi1,gxmEPhi2,gxmHPhi1,gxmHPhi2,
                    gy0EPhi1,gy0EPhi2,gy0HPhi1,gy0HPhi2,
                    gymEPhi1,gymEPhi2,gymHPhi1,gymHPhi2,
                    gz0EPhi1,gz0EPhi2,gz0HPhi1,gz0HPhi2,
                    gzmEPhi1,gzmEPhi2,gzmHPhi1,gzmHPhi2,   
                    None, None, None, None, None, None, None, None,   
                    None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, 
                    None, None, None, None, None, None, None, None, 
                    None, None, None, None, None
                )
