import torch
import ctypes
import warnings
from . import get_deepgpr_lib, set_library_fdtd_order
from .common import (
    _normalize_grid_spacing,
    initialization,
    build_pml_phi,
    create_or_separate,
    build_pml_coeffs,
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

    needs_backward = er_requires_grad or se_requires_grad
    saved_wavefield_bytes = (
        (1 + int(needs_backward))
        * components * nt_saved * snapshot_cells * storage_bytes
    )
    update_coefficient_bytes = 6 * field_cells * float_bytes
    receiver_bytes = nstep * nt * nrx * float_bytes
    gradient_bytes = (
        int(er_requires_grad) + int(se_requires_grad)
    ) * model_cells * float_bytes
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
        forward_transfer_bytes = (
            (2 + 2 * int(needs_backward))
            * components * snapshot_cells * storage_bytes
        )
        backward_transfer_bytes = (
            2 * int(needs_backward) * components * snapshot_cells * storage_bytes
        )
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
        "receiver_data": receiver_bytes,
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
    receiver_component,
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
    print(f"  source / receiver component: {source_direction} / {receiver_component}")
    print(f"  PML thickness [x0, x1, y0, y1, z0, z1]: {pml}")
    print("Model")
    print(
        f"  eps_r range / requires_grad: [{er_min:.6e}, {er_max:.6e}] "
        f"/ {er.requires_grad}"
    )
    print(
        f"  sigma range / requires_grad: [{se_min:.6e}, {se_max:.6e}] "
        f"/ {se.requires_grad}"
    )
    print(f"  mu_r range / supplied: [{mr_min:.6e}, {mr_max:.6e}] / {mr_supplied}")
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
        ("saved E_saved and R_saved wavefields", "saved_gradient_wavefields"),
        ("FDTD update coefficients", "fdtd_update_coefficients"),
        ("receiver data", "receiver_data"),
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


def _begin_native_call(device):
    """Order the native default stream after PyTorch's current CUDA stream."""
    if device.type != "cuda":
        return None

    current_stream = torch.cuda.current_stream(device)
    default_stream = torch.cuda.default_stream(device)
    if current_stream.cuda_stream != default_stream.cuda_stream:
        default_stream.wait_stream(current_stream)
        return current_stream, default_stream
    return None


def _end_native_call(streams):
    """Make PyTorch's current stream wait for an asynchronous native call."""
    if streams is not None:
        current_stream, default_stream = streams
        current_stream.wait_stream(default_stream)


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
            print_parameters=False,
            *, eps_r=None, sigma=None, mu_r=None, receiver_component=None):
    """Run DeepGPR FDTD forward modeling with autograd support.

    Args:
        device: PyTorch device or device string, such as "cpu" or "cuda".
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step size.
        source_amplitudes: Source waveform tensor with shape (nwaveforms, nt, 1).
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        er: Deprecated alias for eps_r.
        se: Deprecated alias for sigma.
        mr: Deprecated alias for mu_r.
        E: Optional initial electric field tuple (Ex, Ey, Ez).
        H: Optional initial magnetic field tuple (Hx, Hy, Hz).
        PML: Optional tuple of 24 PML auxiliary tensors.
        pmlthick: PML thickness as an int, list, or tensor.
        source_direction: Source electric-field polarization, 0 for x, 1 for y, 2 for z.
        reciever_direction: Deprecated alias for receiver_component.
        model_gradient_sampling_interval: Forward wavefield sampling interval for FWI gradients.
        wavefield_storage_dtype: Saved E/R wavefield dtype: float32, float16, or bfloat16.
        use_async_offload: Whether CUDA should offload saved wavefields to pinned CPU memory.
        fdtd_order: Spatial finite-difference order, supported values are 2, 4, and 8.
        mode: FWI gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
        debug: Whether to run expensive tensor validation checks.
        print_parameters: Whether to print a preflight parameter and memory preview.
        eps_r: Relative permittivity tensor with shape (nx, ny) or (nx, ny, nz).
        sigma: Electrical conductivity tensor with the same shape as eps_r.
        mu_r: Relative permeability tensor, or None to use ones.
        receiver_component: Receiver component to return, 0 for x, 1 for y, 2 for z.
    """
    if eps_r is not None and er is not None:
        raise TypeError("Specify only one of eps_r and its deprecated alias er.")
    if sigma is not None and se is not None:
        raise TypeError("Specify only one of sigma and its deprecated alias se.")
    if mu_r is not None and mr is not None:
        raise TypeError("Specify only one of mu_r and its deprecated alias mr.")
    eps_r = er if eps_r is None else eps_r
    sigma = se if sigma is None else sigma
    mu_r = mr if mu_r is None else mu_r
    receiver_component = reciever_direction if receiver_component is None else receiver_component

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
    if source_direction not in (0, 1, 2) or receiver_component not in (0, 1, 2):
        raise ValueError("source_direction and receiver_component must be 0, 1, or 2.")
    if getattr(mu_r, "requires_grad", False):
        raise NotImplementedError("DeepGPR does not currently return relative-permeability gradients.")

    grid_spacing = _normalize_grid_spacing(dx)
    mu_r_supplied = mu_r is not None
    eps_r,sigma,nx,ny,nz,nt,nstep,nsr,nrx,eps_r_pad,sigma_pad,mu_r,spatial_mode,dtype,pmlthick,source_amplitudes=initialization(device,eps_r,sigma,mu_r,source_amplitudes,source_location,receiver_location,grid_spacing,dt,pmlthick,fdtd_order)

    needs_model_gradient = eps_r.requires_grad or sigma.requires_grad
    if needs_model_gradient and model_gradient_sampling_interval > 1:
        warnings.warn(
            "model_gradient_sampling_interval > 1 uses a weighted temporal-sampling "
            "approximation; use interval 1 with float32 storage for exact gradients.",
            RuntimeWarning,
            stacklevel=2,
        )
    if needs_model_gradient and mode == 2:
        if spatial_mode != 2 or source_direction != 2 or receiver_component != 2:
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
            er=eps_r,
            se=sigma,
            mr=mu_r,
            mr_supplied=mu_r_supplied,
            pmlthick=pmlthick,
            source_direction=source_direction,
            receiver_component=receiver_component,
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

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=build_pml_coeffs(eps_r,mu_r,dt,grid_spacing,nx,ny,nz,pmlthick,device,dtype)

    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,E_saved,receiver_amplitudes = DeepGPR.apply(
        eps_r, sigma,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mu_r,grid_spacing,nx,ny,nz,dt,nt,nstep,source_amplitudes,source_location,receiver_location,pmlthick,nsr,nrx,device,dtype,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,eps_r_pad,sigma_pad,source_direction, receiver_component, model_gradient_sampling_interval, wavefield_storage_dtype, use_async_offload, fdtd_order, mode, debug)

    return E_saved,(Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2),receiver_amplitudes


class DeepGPR(torch.autograd.Function):
    """PyTorch autograd bridge for the native DeepGPR backends."""

    @staticmethod
    def forward(ctx, eps_r, sigma,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mu_r,grid_spacing,nx,ny,nz,dt,
                nt,nstep,source_amplitudes,source_location,receiver_location,
                pmlthick,nsr,nrx,device,dtype,x0,xm,
                y0,ym,z0,zm,x01,x02,xm1,xm2,
                y01,y02,ym1,ym2,z01,z02,zm1,zm2,
                eps_r_pad,sigma_pad,source_direction, receiver_component,
                model_gradient_sampling_interval, wavefield_storage_dtype, use_async_offload, fdtd_order, mode, debug):
        """Call the native forward solver and save tensors for backward.

        Args:
            ctx: PyTorch autograd context.
            eps_r: Trainable relative permittivity tensor.
            sigma: Trainable electrical conductivity tensor.
            Ex, Ey, Ez: Electric field component tensors.
            Hx, Hy, Hz: Magnetic field component tensors.
            x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2: Low-x PML auxiliary tensors.
            xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2: High-x PML auxiliary tensors.
            y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2: Low-y PML auxiliary tensors.
            ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2: High-y PML auxiliary tensors.
            z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2: Low-z PML auxiliary tensors.
            zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2: High-z PML auxiliary tensors.
            mu_r: Padded relative permeability tensor.
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
            eps_r_pad: Padded relative permittivity tensor.
            sigma_pad: Padded electrical conductivity tensor.
            source_direction: Source electric-field polarization, 0 for x, 1 for y, 2 for z.
            receiver_component: Receiver component to return, 0 for x, 1 for y, 2 for z.
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
        ctx.save_for_backward(mu_r, source_location, receiver_location,
                              x01,x02,xm1,xm2,
                              y01,y02,ym1,ym2,z01,z02,zm1,zm2,eps_r_pad,sigma_pad)
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
        ctx.eps_r_requires_grad = bool(eps_r.requires_grad)
        ctx.sigma_requires_grad = bool(sigma.requires_grad)
        ctx.source_requires_grad = bool(source_amplitudes.requires_grad)
        ctx.source_shape = tuple(source_amplitudes.shape)
        ctx.source_direction = source_direction
        ctx.pmlthick=pmlthick
        ctx.device=device
        ctx.dtype=dtype
        ctx.model_gradient_sampling_interval = model_gradient_sampling_interval
        ctx.wavefield_storage_type = _WAVEFIELD_STORAGE_TYPES[wavefield_storage_dtype]
        ctx.use_async_offload = bool(use_async_offload and device.type == "cuda")
        ctx.fdtd_order = fdtd_order
        ctx.mode = mode
        ctx.receiver_component = receiver_component
        ctx.debug = bool(debug)

        nt_saved = (nt + model_gradient_sampling_interval - 1) // model_gradient_sampling_interval
        e_components = 3 if mode == 3 else 1
        
        if mode == 3:
            saved_shape = (e_components, nt_saved, nstep, nx, ny, nz)
        else:
            saved_shape = (nt_saved, nstep, nx, ny, nz)

        if ctx.use_async_offload:
            E_saved = torch.empty(
                saved_shape,
                device="cpu",
                dtype=wavefield_storage_dtype,
                pin_memory=True,
            )
            R_saved = (
                torch.empty(
                    saved_shape,
                    device="cpu",
                    dtype=wavefield_storage_dtype,
                    pin_memory=True,
                )
                if ctx.eps_r_requires_grad or ctx.sigma_requires_grad
                else torch.empty(0, device='cpu', dtype=wavefield_storage_dtype)
            )
        else:
            E_saved = torch.empty(
                saved_shape, device=device, dtype=wavefield_storage_dtype
            )
            R_saved = (
                torch.empty(saved_shape, device=device, dtype=wavefield_storage_dtype)
                if ctx.eps_r_requires_grad or ctx.sigma_requires_grad
                else torch.empty(0, device=device, dtype=wavefield_storage_dtype)
            )

        update_coeffs = torch.zeros(
            (6, nx + 1, ny + 1, nz + 1), device=device, dtype=dtype
        )
        ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs = update_coeffs.unbind(0)

        receiver_amplitudes = torch.empty((nstep, nt, nrx), device=device, dtype=dtype)
        pml = [int(pmlthick[i]) for i in range(6)]

        native_streams = _begin_native_call(device)
        c_lib.forward(
                ctypes.cast(eps_r_pad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(sigma_pad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(mu_r.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(E_saved.data_ptr()),
                ctypes.c_void_p(R_saved.data_ptr()),
                ctypes.cast(Ex.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ey.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ez.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hz.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(ce_hist.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ce_curl.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ce_rhs.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ch_hist.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ch_curl.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ch_rhs.data_ptr(), ctypes.POINTER(ctypes.c_float)),

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
                receiver_component,
                nx+1, ny+1, nz+1, nsr,
                ctypes.cast(source_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(source_amplitudes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                source_direction,
                model_gradient_sampling_interval,
                mode,
                ctx.wavefield_storage_type,
                int(ctx.eps_r_requires_grad or ctx.sigma_requires_grad),
                int(ctx.use_async_offload))
        _end_native_call(native_streams)

        if ctx.debug:
            check_tensors_for_nan_inf(d="forward",
                Ex=Ex, Ey=Ey, Ez=Ez,
                Hx=Hx, Hy=Hy, Hz=Hz,
                E_saved=E_saved, R_saved=R_saved,
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

        ctx.E_saved = E_saved
        ctx.R_saved = R_saved
        ctx.mark_non_differentiable(E_saved)
        return (Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,E_saved,receiver_amplitudes)

    @staticmethod
    def backward(ctx,lambda_ex,lambda_ey,lambda_ez,lambda_hx,lambda_hy,lambda_hz,lambda_x0_e_phi1,lambda_x0_e_phi2,lambda_x0_h_phi1,lambda_x0_h_phi2,lambda_xm_e_phi1,lambda_xm_e_phi2,lambda_xm_h_phi1,lambda_xm_h_phi2,lambda_y0_e_phi1,lambda_y0_e_phi2,lambda_y0_h_phi1,lambda_y0_h_phi2,lambda_ym_e_phi1,lambda_ym_e_phi2,lambda_ym_h_phi1,lambda_ym_h_phi2,lambda_z0_e_phi1,lambda_z0_e_phi2,lambda_z0_h_phi1,lambda_z0_h_phi2,lambda_zm_e_phi1,lambda_zm_e_phi2,lambda_zm_h_phi1,lambda_zm_h_phi2,_g_E_saved,data_grad):
        """Call the native adjoint solver and return gradients.

        Args:
            ctx: PyTorch autograd context saved by forward.
            lambda_ex, lambda_ey, lambda_ez: Incoming gradients for electric field components.
            lambda_hx, lambda_hy, lambda_hz: Incoming gradients for magnetic field components.
            lambda_x0_e_phi1, lambda_x0_e_phi2, lambda_x0_h_phi1, lambda_x0_h_phi2: Gradients for low-x PML tensors.
            lambda_xm_e_phi1, lambda_xm_e_phi2, lambda_xm_h_phi1, lambda_xm_h_phi2: Gradients for high-x PML tensors.
            lambda_y0_e_phi1, lambda_y0_e_phi2, lambda_y0_h_phi1, lambda_y0_h_phi2: Gradients for low-y PML tensors.
            lambda_ym_e_phi1, lambda_ym_e_phi2, lambda_ym_h_phi1, lambda_ym_h_phi2: Gradients for high-y PML tensors.
            lambda_z0_e_phi1, lambda_z0_e_phi2, lambda_z0_h_phi1, lambda_z0_h_phi2: Gradients for low-z PML tensors.
            lambda_zm_e_phi1, lambda_zm_e_phi2, lambda_zm_h_phi1, lambda_zm_h_phi2: Gradients for high-z PML tensors.
            _g_E_saved: Ignored gradient for the non-differentiable saved wavefield.
            data_grad: Incoming gradient for receiver amplitudes.
        """
        
        del _g_E_saved
        (
            mu_r, source_location, receiver_location,
            x01, x02, xm1, xm2, y01, y02, ym1, ym2,
            z01, z02, zm1, zm2, eps_r_pad, sigma_pad,
        ) = (tensor.contiguous() for tensor in ctx.saved_tensors)

        dx, dy, dz=ctx.grid_spacing
        nx=ctx.nx
        ny=ctx.ny
        nz=ctx.nz
        dt=ctx.dt
        nt=ctx.nt
        nsource=ctx.nsr
        nreceiver=ctx.nrx
        dtype=ctx.dtype
        nstep=ctx.nstep
        pmlthick=ctx.pmlthick
        device=ctx.device
        E_saved=ctx.E_saved
        R_saved=ctx.R_saved
        model_gradient_sampling_interval = ctx.model_gradient_sampling_interval
        c_lib = get_deepgpr_lib(device)
        set_library_fdtd_order(c_lib, ctx.fdtd_order)

        E_saved=E_saved.contiguous()
        R_saved=R_saved.contiguous()
        lambda_ex=lambda_ex.contiguous()
        lambda_ey=lambda_ey.contiguous()
        lambda_ez=lambda_ez.contiguous()
        lambda_hx=lambda_hx.contiguous()
        lambda_hy=lambda_hy.contiguous()
        lambda_hz=lambda_hz.contiguous()
        lambda_x0_e_phi1=lambda_x0_e_phi1.contiguous()
        lambda_x0_e_phi2=lambda_x0_e_phi2.contiguous()
        lambda_x0_h_phi1=lambda_x0_h_phi1.contiguous()
        lambda_x0_h_phi2=lambda_x0_h_phi2.contiguous()
        lambda_xm_e_phi1=lambda_xm_e_phi1.contiguous()
        lambda_xm_e_phi2=lambda_xm_e_phi2.contiguous()
        lambda_xm_h_phi1=lambda_xm_h_phi1.contiguous()
        lambda_xm_h_phi2=lambda_xm_h_phi2.contiguous()
        lambda_y0_e_phi1=lambda_y0_e_phi1.contiguous()
        lambda_y0_e_phi2=lambda_y0_e_phi2.contiguous()
        lambda_y0_h_phi1=lambda_y0_h_phi1.contiguous()
        lambda_y0_h_phi2=lambda_y0_h_phi2.contiguous()
        lambda_ym_e_phi1=lambda_ym_e_phi1.contiguous()
        lambda_ym_e_phi2=lambda_ym_e_phi2.contiguous()
        lambda_ym_h_phi1=lambda_ym_h_phi1.contiguous()
        lambda_ym_h_phi2=lambda_ym_h_phi2.contiguous()
        lambda_z0_e_phi1=lambda_z0_e_phi1.contiguous()
        lambda_z0_e_phi2=lambda_z0_e_phi2.contiguous()
        lambda_z0_h_phi1=lambda_z0_h_phi1.contiguous()
        lambda_z0_h_phi2=lambda_z0_h_phi2.contiguous()
        lambda_zm_e_phi1=lambda_zm_e_phi1.contiguous()
        lambda_zm_e_phi2=lambda_zm_e_phi2.contiguous()
        lambda_zm_h_phi1=lambda_zm_h_phi1.contiguous()
        lambda_zm_h_phi2=lambda_zm_h_phi2.contiguous()
        data_grad=data_grad.contiguous()

        update_coeffs = torch.zeros(
            (6, nx + 1, ny + 1, nz + 1), device=device, dtype=dtype
        )
        ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs = update_coeffs.unbind(0)

        if ctx.eps_r_requires_grad:
            grad_eps_r=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            eps_r_requires_grad=1
        else:
            grad_eps_r=torch.empty(0, device=device, dtype=dtype)
            eps_r_requires_grad=0

        if ctx.sigma_requires_grad:
            grad_sigma=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            sigma_requires_grad=1
        else:
            grad_sigma=torch.empty(0, device=device, dtype=dtype)
            sigma_requires_grad=0

        if ctx.source_requires_grad:
            grad_source = torch.zeros(ctx.source_shape, device=device, dtype=dtype).contiguous()
            source_requires_grad = 1
        else:
            grad_source = torch.empty(0, device=device, dtype=dtype)
            source_requires_grad = 0

        pml = [int(pmlthick[i]) for i in range(6)]

        native_streams = _begin_native_call(device)
        c_lib.backward(
                ctypes.cast(eps_r_pad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(sigma_pad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(mu_r.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(E_saved.data_ptr()),
                ctypes.c_void_p(R_saved.data_ptr()),
                ctypes.cast(lambda_ex.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_ey.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_ez.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_hx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_hy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_hz.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(ce_hist.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ce_curl.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ce_rhs.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ch_hist.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ch_curl.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ch_rhs.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(lambda_x0_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_x0_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_x0_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_x0_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_xm_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_xm_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_xm_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_xm_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_y0_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_y0_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_y0_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_y0_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_ym_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_ym_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_ym_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_ym_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_z0_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_z0_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_z0_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_z0_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_zm_e_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_zm_e_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(lambda_zm_h_phi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(lambda_zm_h_phi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                pml[0],pml[1],pml[2],
                pml[3],pml[4],pml[5],

                ctypes.cast(x01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym1.data_ptr(), ctypes.POINTER(ctypes.c_float)),       
                ctypes.cast(z01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),                
                ctypes.cast(x02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(y02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym2.data_ptr(), ctypes.POINTER(ctypes.c_float)),         
                ctypes.cast(z02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm2.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                dt, nt, nstep, nreceiver, dx, dy, dz,
                nx+1, ny+1, nz+1, nreceiver,
                ctypes.cast(receiver_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(data_grad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctx.receiver_component,
                nsource,
                ctypes.cast(source_location.data_ptr(), ctypes.POINTER(ctypes.c_int)),
                ctx.source_direction,
                ctypes.cast(grad_source.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                source_requires_grad,
                ctypes.cast(grad_eps_r.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(grad_sigma.data_ptr(), ctypes.POINTER(ctypes.c_float)),eps_r_requires_grad,sigma_requires_grad,
                model_gradient_sampling_interval,
                ctx.mode,
                ctx.wavefield_storage_type,
                int(ctx.use_async_offload))
        _end_native_call(native_streams)
        
        state_gradient_names = (
            "lambda_ex", "lambda_ey", "lambda_ez",
            "lambda_hx", "lambda_hy", "lambda_hz",
            "lambda_x0_e_phi1", "lambda_x0_e_phi2",
            "lambda_x0_h_phi1", "lambda_x0_h_phi2",
            "lambda_xm_e_phi1", "lambda_xm_e_phi2",
            "lambda_xm_h_phi1", "lambda_xm_h_phi2",
            "lambda_y0_e_phi1", "lambda_y0_e_phi2",
            "lambda_y0_h_phi1", "lambda_y0_h_phi2",
            "lambda_ym_e_phi1", "lambda_ym_e_phi2",
            "lambda_ym_h_phi1", "lambda_ym_h_phi2",
            "lambda_z0_e_phi1", "lambda_z0_e_phi2",
            "lambda_z0_h_phi1", "lambda_z0_h_phi2",
            "lambda_zm_e_phi1", "lambda_zm_e_phi2",
            "lambda_zm_h_phi1", "lambda_zm_h_phi2",
        )
        state_gradients = (
            lambda_ex, lambda_ey, lambda_ez,
            lambda_hx, lambda_hy, lambda_hz,
            lambda_x0_e_phi1, lambda_x0_e_phi2,
            lambda_x0_h_phi1, lambda_x0_h_phi2,
            lambda_xm_e_phi1, lambda_xm_e_phi2,
            lambda_xm_h_phi1, lambda_xm_h_phi2,
            lambda_y0_e_phi1, lambda_y0_e_phi2,
            lambda_y0_h_phi1, lambda_y0_h_phi2,
            lambda_ym_e_phi1, lambda_ym_e_phi2,
            lambda_ym_h_phi1, lambda_ym_h_phi2,
            lambda_z0_e_phi1, lambda_z0_e_phi2,
            lambda_z0_h_phi1, lambda_z0_h_phi2,
            lambda_zm_e_phi1, lambda_zm_e_phi2,
            lambda_zm_h_phi1, lambda_zm_h_phi2,
        )
        state_needs_grad = ctx.needs_input_grad[2:32]
        tensors_to_check = {
            name: gradient
            for name, gradient, required in zip(
                state_gradient_names, state_gradients, state_needs_grad
            )
            if required
        }
        if eps_r_requires_grad == 1:
            tensors_to_check["grad_eps_r"] = grad_eps_r
        if sigma_requires_grad == 1:
            tensors_to_check["grad_sigma"] = grad_sigma
        if source_requires_grad == 1:
            tensors_to_check["grad_source"] = grad_source

        if ctx.debug:
            check_tensors_for_nan_inf(d="backward", **tensors_to_check)

        ctx.E_saved = None
        ctx.R_saved = None
        del E_saved,R_saved,mu_r,source_location,receiver_location,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,eps_r_pad,sigma_pad,ce_hist,ce_curl,ce_rhs,ch_hist,ch_curl,ch_rhs

        gradients = [None] * 76
        gradients[0] = grad_eps_r if eps_r_requires_grad else None
        gradients[1] = grad_sigma if sigma_requires_grad else None
        for index, (gradient, required) in enumerate(
            zip(state_gradients, state_needs_grad), start=2
        ):
            gradients[index] = gradient if required else None
        gradients[40] = grad_source if source_requires_grad else None
        return tuple(gradients)
