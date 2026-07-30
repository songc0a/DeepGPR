import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
import warnings

c = 299792458.0
m0 = 4.0 * math.pi * 1e-7
e0 = 1.0 / (m0 * c * c)


def _normalize_grid_spacing(value):
    """Return grid spacing as a validated ``(dx, dy, dz)`` tuple."""
    if torch.is_tensor(value):
        if value.numel() == 1:
            values = [value.detach().item()] * 3
        elif value.numel() == 3:
            values = value.detach().cpu().reshape(-1).tolist()
        else:
            raise ValueError("dx tensor must contain either one or three values.")
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        values = [value] * 3
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("dx list or tuple must contain exactly three values.")
        values = value
    else:
        raise TypeError(
            "dx must be a positive scalar or a three-value list, tuple, or tensor."
        )

    try:
        spacing = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("dx, dy, and dz must be finite positive scalars.") from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in spacing):
        raise ValueError("dx, dy, and dz must be finite positive scalars.")
    return spacing

def _locations_in_pml(locations, shape, pml):
    """Return a mask for acquisition coordinates that overlap CPML."""
    inside = torch.zeros(locations.shape[:-1], dtype=torch.bool, device=locations.device)
    for axis, size in enumerate(shape):
        low = int(pml[2 * axis])
        high = int(pml[2 * axis + 1])
        if low > 0:
            inside |= locations[..., axis] <= low
        if high > 0:
            inside |= locations[..., axis] >= size - high
    return inside


def _warn_for_pml_location_count(name, count):
    """Emit the standard acquisition-in-CPML warning for a known count."""
    if count:
        warnings.warn(
            f"{count} {name} coordinate(s) lie inside CPML. DeepGPR's CPML occupies "
            "cells inside the supplied model; place acquisition points in the physical "
            "interior (low_pml < index < size - high_pml) to avoid attenuated data and "
            "unreliable boundary sensitivity.",
            RuntimeWarning,
            stacklevel=3,
        )


def initialization(device, er,se,mr,source_amplitudes,source_location,receiver_location,dx,dt,pmlthick,fdtd_order=2):
    """Validate inputs and prepare model, source, receiver, and PML metadata.

    Args:
        device: PyTorch device where tensors will be stored.
        er: Relative permittivity tensor with shape (nx, ny) or (nx, ny, nz).
        se: Electrical conductivity tensor with the same shape as er.
        mr: Relative permeability tensor, or None to use ones.
        source_amplitudes: Source waveform tensor with shape (nwaveforms, nt, 1).
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step size.
        pmlthick: PML thickness as an int, list, or tensor.
        fdtd_order: Spatial finite-difference order used for the CFL check.
    """
    device = torch.device("cpu" if device is None else device)
    dtype=torch.float32
    spacing = _normalize_grid_spacing(dx)
    try:
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt must be a finite positive scalar.") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar.")
    if not torch.is_tensor(er) or not torch.is_tensor(se):
        raise TypeError("er and se must be PyTorch tensors.")
    if len(er.shape) == 2:
        er = er.reshape(*er.shape, 1)
    elif len(er.shape) != 3:
        raise ValueError('The shape of epsilon should be 2-d or 3-d.')
    
    if len(se.shape) == 2:
        se = se.reshape(*se.shape, 1)
    elif len(se.shape) != 3:
        raise ValueError('The shape of sigma should be 2-d or 3-d.')

    if mr is not None and not torch.is_tensor(mr):
        raise TypeError("mr must be a PyTorch tensor or None.")
    if mr is not None:
        if len(mr.shape) == 2:
            mr = mr.reshape(*mr.shape, 1)
        elif len(mr.shape) != 3:
            raise ValueError('The shape of mr should be 2-d or 3-d.')

    if er.shape != se.shape:
        raise ValueError('The shape of epsilon and sigma should be the same.')
    if any(size < 1 for size in er.shape):
        raise ValueError("The material model dimensions must all be non-empty.")
    nx, ny, nz = er.shape
    mode = 2 if nz == 1 else 3
    er=er.to(device=device, dtype=dtype)
    se=se.to(device=device, dtype=dtype)
    if mr is None:
        mr=torch.ones_like(er, device=device)
    elif mr.shape == er.shape:
        mr=mr.to(device=device, dtype=dtype)
    else:
        raise ValueError('The shape of mr should be the same as epsilon and sigma.')

    if not torch.is_tensor(source_location) or not torch.is_tensor(receiver_location):
        raise TypeError("source_location and receiver_location must be PyTorch tensors.")
    if source_location.ndim != 3 or receiver_location.ndim != 3:
        raise ValueError("source_location and receiver_location must have shape (nstep, count, 3).")
    if source_location.shape[2] != 3 or receiver_location.shape[2] != 3:
        raise ValueError("The last dimension of source_location and receiver_location must be 3.")
    if source_location.shape[0] != receiver_location.shape[0]:
        raise ValueError('The first dimension (nstep) of source_location and receiver_location should be the same.')
    nstep=source_location.shape[0]
    nsr=source_location.shape[1]
    nrx=receiver_location.shape[1]
    if nstep < 1:
        raise ValueError("At least one shot is required.")
    if nsr < 1:
        raise ValueError("At least one source per shot is required.")
    if nrx < 1:
        raise ValueError("At least one receiver per shot is required.")
    if device.type == "cuda" and nstep > 65535:
        raise ValueError(
            "CUDA supports at most 65535 shots in one DeepGPR compute call. "
            "Split a larger acquisition batch into smaller calls."
        )
    for name, locations in (
        ("source_location", source_location),
        ("receiver_location", receiver_location),
    ):
        if locations.dtype == torch.bool or locations.is_complex():
            raise TypeError(f"{name} must contain real integer-valued coordinates.")
    source_location=source_location.to(device=device).contiguous()
    receiver_location=receiver_location.to(device=device).contiguous()

    if not torch.is_tensor(source_amplitudes):
        raise TypeError("source_amplitudes must be a PyTorch tensor.")
    if source_amplitudes.ndim not in (2, 3):
        raise ValueError('source_amplitudes must have shape (num_waveforms, nt) or (num_waveforms, nt, 1).')
    if source_amplitudes.ndim == 2:
        source_amplitudes = source_amplitudes.unsqueeze(-1)
    if source_amplitudes.ndim == 3 and source_amplitudes.shape[2] != 1:
        raise ValueError('The last dimension of source_amplitudes must be 1.')
    if source_amplitudes.shape[0] < 1 or source_amplitudes.shape[1] < 1:
        raise ValueError('source_amplitudes must contain at least one waveform and one time sample.')
    source_amplitudes=source_amplitudes.to(device=device, dtype=dtype).contiguous()

    if (source_amplitudes.shape[0]>1 and source_amplitudes.shape[0]<nsr) or source_amplitudes.shape[0]>nsr :
        raise ValueError('The number of source waveforms is incorrect.')
    elif source_amplitudes.shape[0]==1 and nsr!=1:
        source_amplitudes=source_amplitudes.repeat(nsr,1,1).contiguous()
        print('Tips: The number of source waveforms is 1, but the number of sources is ',nsr,'. The source waveform is repeated for all sources.')

    nt=source_amplitudes.shape[1]
    pmlthick=pmlthick_revert(pmlthick,er)
    if pmlthick.numel() != 6:
        raise ValueError("pmlthick must contain six boundary thicknesses.")
    pml_values = [int(value) for value in pmlthick.tolist()]
    if any(value < 0 for value in pml_values):
        raise ValueError("PML thicknesses must be non-negative.")
    for axis, size in enumerate((nx, ny, nz)):
        low, high = pml_values[2 * axis:2 * axis + 2]
        if low + high > max(size - 2, 0):
            raise ValueError(
                f"PML thicknesses on axis {axis} leave no physical interior: "
                f"low={low}, high={high}, size={size}."
            )

    shape_tensor = torch.tensor((nx, ny, nz), dtype=torch.int32, device=device)
    source_integral = (
        (torch.isfinite(source_location) & (source_location == source_location.trunc())).all()
        if source_location.is_floating_point()
        else torch.ones((), dtype=torch.bool, device=device)
    )
    receiver_integral = (
        (torch.isfinite(receiver_location) & (receiver_location == receiver_location.trunc())).all()
        if receiver_location.is_floating_point()
        else torch.ones((), dtype=torch.bool, device=device)
    )
    source_valid = ((source_location >= 0) & (source_location < shape_tensor)).all()
    receiver_valid = ((receiver_location >= 0) & (receiver_location < shape_tensor)).all()
    source_in_pml = _locations_in_pml(source_location, (nx, ny, nz), pml_values)
    receiver_in_pml = _locations_in_pml(receiver_location, (nx, ny, nz), pml_values)
    stats = torch.stack(
        (
            torch.isfinite(er).all().to(dtype),
            torch.isfinite(se).all().to(dtype),
            torch.isfinite(mr).all().to(dtype),
            torch.isfinite(source_amplitudes).all().to(dtype),
            er.amin(),
            se.amin(),
            mr.amin(),
            (er.detach() * mr.detach()).amin(),
            source_valid.to(dtype),
            receiver_valid.to(dtype),
            source_integral.to(dtype),
            receiver_integral.to(dtype),
            source_in_pml.sum().to(dtype),
            receiver_in_pml.sum().to(dtype),
        )
    ).detach().cpu().tolist()
    er_finite, se_finite, mr_finite, source_finite = (bool(value) for value in stats[:4])
    er_min, se_min, mr_min, min_er_mr = stats[4:8]
    source_valid, receiver_valid = (bool(value) for value in stats[8:10])
    source_integral, receiver_integral = (bool(value) for value in stats[10:12])
    source_pml_count, receiver_pml_count = (int(value) for value in stats[12:14])

    for name, finite in (
        ("er", er_finite), ("se", se_finite), ("mr", mr_finite),
        ("source_amplitudes", source_finite),
    ):
        if not finite:
            raise ValueError(f"`{name}` contains NaN or Inf values.")
    if er_min < 1:
        raise ValueError('The values of epsilon is incorrect.(should be greater than 1)')
    if se_min < 0:
        raise ValueError('The values of sigma is incorrect.(should be non-negative)')
    if mr_min <= 0:
        raise ValueError('The values of mr are incorrect (must be positive).')
    if not source_integral:
        raise ValueError("source_location must contain finite integer-valued coordinates.")
    if not receiver_integral:
        raise ValueError("receiver_location must contain finite integer-valued coordinates.")
    if not source_valid:
        raise ValueError(
            "Error: Source coordinates out of range! "
            f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
        )
    if not receiver_valid:
        raise ValueError(
            "Error: Receiver coordinates out of range! "
            f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
        )

    check_cfl(
        spacing, dt, nx, ny, nz, fdtd_order=fdtd_order,
        _material_min_er_mr=min_er_mr,
    )
    _warn_for_pml_location_count("source", source_pml_count)
    _warn_for_pml_location_count("receiver", receiver_pml_count)

    ere=F.pad(er, (0, 1, 0, 1, 0, 1))
    see=F.pad(se, (0, 1, 0, 1, 0, 1))
    mr=F.pad(mr, (0, 1, 0, 1, 0, 1))

    return er,se,nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,mode,dtype,pmlthick,source_amplitudes


def check_cfl(dx, dt, nx,ny,nz,er=None,mr=None,fdtd_order=2, _material_min_er_mr=None):
    """Check the CFL stability condition for the simulation grid.

    Args:
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step size.
        nx: Number of cells along the x axis.
        ny: Number of cells along the y axis.
        nz: Number of cells along the z axis.
        er: Optional relative-permittivity tensor used to find the fastest material.
        mr: Optional relative-permeability tensor used to find the fastest material.
        fdtd_order: Spatial finite-difference order, 2, 4, or 8.
    """

    spacing = _normalize_grid_spacing(dx)
    try:
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt must be a finite positive scalar.") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar.")
    if fdtd_order not in (2, 4, 8):
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    coefficient_sums = {
        2: 1.0,
        4: 9.0 / 8.0 + 1.0 / 24.0,
        8: 1225.0 / 1024.0 + 245.0 / 3072.0 + 49.0 / 5120.0 + 5.0 / 7168.0,
    }
    active_inverse_spacing_squared = sum(
        1.0 / (axis_spacing * axis_spacing)
        for size, axis_spacing in zip((nx, ny, nz), spacing)
        if size > 1
    )
    if active_inverse_spacing_squared == 0.0:
        raise ValueError("At least one model dimension must contain more than one cell.")

    material_factor = 1.0
    if _material_min_er_mr is not None:
        min_er_mr = float(_material_min_er_mr)
        if not math.isfinite(min_er_mr) or min_er_mr <= 0.0:
            raise ValueError("epsilon_r * mu_r must be finite and positive for the CFL check.")
        material_factor = math.sqrt(min_er_mr)
    elif er is not None and mr is not None:
        min_er_mr = float((er.detach() * mr.detach()).amin().item())
        if not math.isfinite(min_er_mr) or min_er_mr <= 0.0:
            raise ValueError("epsilon_r * mu_r must be finite and positive for the CFL check.")
        material_factor = math.sqrt(min_er_mr)

    spectral_factor = coefficient_sums[fdtd_order]
    dt_max = material_factor / (
        c * spectral_factor * math.sqrt(active_inverse_spacing_squared)
    )

    if dt > dt_max:
        raise ValueError(f"Does not meet CFL conditions: dt={dt:.3e} > dt_max={dt_max:.3e}")


def pmlthick_revert(p, er):
    """Convert user PML thickness input to a six-boundary tensor.

    Args:
        p: PML thickness as an int, list, or tensor.
        er: Relative permittivity tensor used to detect 2D or 3D mode.
    """
    if isinstance(p, bool):
        raise TypeError("PML thickness must contain integer values, not bool.")
    if isinstance(p, int):
        values = [p, p, p, p, 0, 0] if er.shape[2] == 1 else [p] * 6
    elif isinstance(p, (list, tuple)):
        if len(p) == 4:
            values = [*p, 0, 0]
        elif len(p) == 6:
            values = list(p)
        else:
            raise ValueError(f"Unsupported PML length: {len(p)}. Must be 4 or 6.")
    elif isinstance(p, torch.Tensor):
        if p.ndim != 1:
            raise ValueError("PML thickness tensor must be one-dimensional.")
        if p.numel() == 4:
            values = [*p.detach().cpu().tolist(), 0, 0]
        elif p.numel() == 6:
            values = p.detach().cpu().tolist()
        else:
            raise ValueError(
                f"Unsupported PML length: {p.numel()}. Must be 4 or 6."
            )
    else:
        raise TypeError(f"Unsupported PML thickness type: {type(p)}")

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("PML thickness must contain integer values, not bool.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("PML thickness must contain finite integer values.") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("PML thickness must contain finite integer values.")
        integer = int(numeric)
        if integer < 0 or integer > torch.iinfo(torch.int32).max:
            raise ValueError("PML thickness values must fit in non-negative int32.")
        normalized.append(integer)
    return torch.tensor(normalized, dtype=torch.int32)


class TVRegularization(nn.Module):
    """Total variation regularization for permittivity and conductivity models.

    Args:
        weight_ep: Weight applied to the permittivity TV loss.
        weight_sigma: Weight applied to the conductivity TV loss.
        method: TV variant, either "anisotropic" or another value for isotropic.
    """
    def __init__(self, weight_ep=1, weight_sigma=0.001, method='anisotropic'):
        """Create a TV regularization module.

        Args:
            weight_ep: Weight applied to the permittivity TV loss.
            weight_sigma: Weight applied to the conductivity TV loss.
            method: TV variant, either "anisotropic" or another value for isotropic.
        """
        super(TVRegularization, self).__init__()
        self.weight_ep = weight_ep
        self.weight_sigma = weight_sigma
        self.method = method

    def _compute_tv(self, data):
        """Compute total variation for one tensor.

        Args:
            data: 2D, 3D, or batched tensor to regularize.
        """
        if data.dim() == 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
            
        if data.dim() == 2:
            d_x = data[1:, :] - data[:-1, :]
            d_y = data[:, 1:] - data[:, :-1]
            
            loss_x = torch.sum(torch.abs(d_x))
            loss_y = torch.sum(torch.abs(d_y))
            loss_z = 0.0
            
        elif data.dim() == 3:
            d_x = data[1:, :, :] - data[:-1, :, :]
            d_y = data[:, 1:, :] - data[:, :-1, :]
            d_z = data[:, :, 1:] - data[:, :, :-1]
            
            loss_x = torch.sum(torch.abs(d_x))
            loss_y = torch.sum(torch.abs(d_y))
            loss_z = torch.sum(torch.abs(d_z))
            
        elif data.dim() == 4:
             d_x = data[..., 1:, :] - data[..., :-1, :]
             d_y = data[..., :, 1:] - data[..., :, :-1]
             
             loss_x = torch.sum(torch.abs(d_x))
             loss_y = torch.sum(torch.abs(d_y))
             loss_z = 0.0
             
        else:
            return torch.tensor(0.0, device=data.device)

        if self.method == 'anisotropic':

            total_tv = loss_x + loss_y + loss_z
        else:

            total_tv = (loss_x**2 + loss_y**2 + loss_z**2 + 1e-8).sqrt()

        return total_tv 

    def forward(self, ep=None, sigma=None):
        """Return the weighted TV loss for model parameters.

        Args:
            ep: Relative permittivity tensor, or None to skip it.
            sigma: Conductivity tensor, or None to skip it.
        """
        loss = torch.tensor(0.0, device=ep.device if ep is not None else sigma.device)

        if ep is not None and self.weight_ep > 0:
            loss += self.weight_ep * self._compute_tv(ep)

        if sigma is not None and self.weight_sigma > 0:
            loss += self.weight_sigma * self._compute_tv(sigma)
            
        return loss


def create_or_separate(fields:tuple, nx,ny,nz,nstep,device: torch.device,
                  dtype: torch.dtype):
    """Create zero field components or validate existing field components.

    Args:
        fields: Existing (x, y, z) field tensors, or None to allocate zeros.
        nx: Number of model cells along the x axis.
        ny: Number of model cells along the y axis.
        nz: Number of model cells along the z axis.
        nstep: Number of shots or simulations in the batch.
        device: PyTorch device for allocated tensors.
        dtype: PyTorch dtype for allocated tensors.
    """
    if fields is None:
        return torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous()
    if not isinstance(fields, (list, tuple)) or len(fields) != 3:
        raise ValueError("E and H must each contain exactly three field tensors.")

    expected_shape = (nstep, nx + 1, ny + 1, nz + 1)
    for component in fields:
        if not isinstance(component, torch.Tensor) or component.shape != expected_shape:
            actual_shape = getattr(component, "shape", None)
            raise ValueError(
                f"Field shape mismatch: got {actual_shape}, expected {expected_shape}."
            )

    return tuple(
        component.to(device=device, dtype=dtype).contiguous()
        for component in fields
    )



def check_tensors_for_nan_inf(d,**tensors):
    """Check multiple named tensors for NaN or Inf values.

    Args:
        d: Label describing the current calculation stage.
        **tensors: Named tensors to validate.
    """
    found_issue = False

    for name, tensor in tensors.items():
        if tensor is None:
            print(f"[WARNING]{d}: {name} is None.")
            continue

        if not isinstance(tensor, torch.Tensor):
            print(f"[WARNING]{d}: {name} is not a tensor: {type(tensor)}")
            continue

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            found_issue = True
            print(f"[ERROR]{d}: Tensor `{name}` contains:", end=" ")
            if has_nan:
                print("NaN ", end="")
            if has_inf:
                print("Inf ", end="")
            print(f"| shape={tuple(tensor.shape)} | dtype={tensor.dtype}")

    if found_issue:
        raise FloatingPointError(f"{d} produced tensors containing NaN or Inf values.")


def build_pml_coeffs(eps_r,mu_r,dt,dx,nx,ny,nz,pmlthick,device,dtype):
    """Build fixed CPML region descriptors and update coefficients.

    CPML is a numerical boundary rather than an inversion parameter. Boundary
    material averages are detached explicitly, and gradients in CPML cells are
    excluded by the native material-gradient kernels.

    Args:
        eps_r: Relative permittivity tensor.
        mu_r: Relative permeability tensor.
        dt: Time step size.
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        nx: Number of model cells along the x axis.
        ny: Number of model cells along the y axis.
        nz: Number of model cells along the z axis.
        pmlthick: Six-boundary PML thickness tensor.
        device: PyTorch device for output tensors.
        dtype: PyTorch dtype for output tensors.
    """
    dx, dy, dz = _normalize_grid_spacing(dx)
    eps_r_fixed = eps_r.detach()
    mu_r_fixed = mu_r.detach()
    pml = tuple(int(value) for value in pmlthick.tolist())
    lencfs=1
    x0 = torch.empty(0)
    xm = torch.empty(0)
    y0 = torch.empty(0)
    ym = torch.empty(0)
    z0 = torch.empty(0)
    zm = torch.empty(0)
    x01 = torch.empty(0)
    x02 = torch.empty(0)
    xm1 = torch.empty(0)
    xm2 = torch.empty(0)
    y01 = torch.empty(0)
    y02 = torch.empty(0)
    ym1 = torch.empty(0)
    ym2 = torch.empty(0)
    z01 = torch.empty(0)
    z02 = torch.empty(0)
    zm1 = torch.empty(0)
    zm2 = torch.empty(0)
    if pml[0]>0:
        x0=torch.tensor((pml[0],0,pml[0],0,ny,0,nz), dtype=torch.int)
        average_eps=eps_r_fixed[0,:ny,:nz].mean()
        average_mu=mu_r_fixed[0,:ny,:nz].mean()
        CFS0=CFS(device=device)
        x01=torch.zeros((4,lencfs,pml[0]), device=device, dtype=dtype)
        x02=torch.zeros((4,lencfs,pml[0]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS0,x01,x02, average_eps, average_mu, dt,dx,pml[0])

    if pml[1]>0:
        xm=torch.tensor((pml[1],nx-pml[1],nx,0,ny,0,nz), dtype=torch.int)
        average_eps=eps_r_fixed[nx-pml[1],:ny,:nz].mean()
        average_mu=mu_r_fixed[nx-pml[1],:ny,:nz].mean()
        CFS1=CFS(device=device)
        xm1=torch.zeros((4,lencfs,pml[1]), device=device, dtype=dtype)
        xm2=torch.zeros((4,lencfs,pml[1]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS1,xm1,xm2, average_eps, average_mu, dt,dx,pml[1])

    if pml[2]>0:
        y0=torch.tensor((pml[2],0,nx,0,pml[2],0,nz), dtype=torch.int)
        average_eps=eps_r_fixed[:nx,0,:nz].mean()
        average_mu=mu_r_fixed[:nx,0,:nz].mean()
        CFS2=CFS(device=device)
        y01=torch.zeros((4,lencfs,pml[2]), device=device, dtype=dtype)
        y02=torch.zeros((4,lencfs,pml[2]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS2,y01,y02, average_eps, average_mu, dt,dy,pml[2])

    if pml[3]>0:
        ym=torch.tensor((pml[3],0,nx,ny-pml[3],ny,0,nz), dtype=torch.int)
        average_eps=eps_r_fixed[:nx,ny-pml[3],:nz].mean()
        average_mu=mu_r_fixed[:nx,ny-pml[3],:nz].mean()
        CFS3=CFS(device=device)
        ym1=torch.zeros((4,lencfs,pml[3]), device=device, dtype=dtype)
        ym2=torch.zeros((4,lencfs,pml[3]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS3,ym1,ym2, average_eps, average_mu, dt,dy,pml[3])

    if pml[4]>0:
        z0=torch.tensor((pml[4],0,nx,0,ny,0,pml[4]), dtype=torch.int)
        average_eps=eps_r_fixed[:nx,:ny,0].mean()
        average_mu=mu_r_fixed[:nx,:ny,0].mean()
        CFS4=CFS(device=device)
        z01=torch.zeros((4,lencfs,pml[4]), device=device, dtype=dtype)
        z02=torch.zeros((4,lencfs,pml[4]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS4,z01,z02, average_eps, average_mu, dt,dz,pml[4])

    if pml[5]>0:
        zm=torch.tensor((pml[5],0,nx,0,ny,nz-pml[5],nz), dtype=torch.int)
        average_eps=eps_r_fixed[:nx,:ny,nz-pml[5]].mean()
        average_mu=mu_r_fixed[:nx,:ny,nz-pml[5]].mean()
        CFS5=CFS(device=device)
        zm1=torch.zeros((4,lencfs,pml[5]), device=device, dtype=dtype)
        zm2=torch.zeros((4,lencfs,pml[5]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS5,zm1,zm2, average_eps, average_mu, dt,dz,pml[5])
    return x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2


def buildpmlcoeffs(*args, **kwargs):
    """Deprecated compatibility alias for :func:`build_pml_coeffs`."""
    if "er" in kwargs:
        kwargs["eps_r"] = kwargs.pop("er")
    if "mr" in kwargs:
        kwargs["mu_r"] = kwargs.pop("mr")
    return build_pml_coeffs(*args, **kwargs)



class CFSParameter(object):
    """Parameter settings for complex frequency shifted PML profiles.

    Args:
        ID: Parameter name, such as "alpha", "kappa", or "sigma".
        scaling: Scaling family used to generate the profile.
        scalingprofile: Profile order name used by polynomial scaling.
        min: Minimum value of the profile.
        max: Maximum value of the profile.
    """
    scalingprofiles = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5, 'sextic': 6, 'septic': 7, 'octic': 8}

    def __init__(self,ID =None, scaling='polynomial', scalingprofile=None, min=0, max=0):
        """Create one CFS profile parameter.

        Args:
            ID: Parameter name, such as "alpha", "kappa", or "sigma".
            scaling: Scaling family used to generate the profile.
            scalingprofile: Profile order name used by polynomial scaling.
            min: Minimum value of the profile.
            max: Maximum value of the profile.
        """
        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.min = min
        self.max = max


class CFS(object):
    """Complex frequency shifted PML coefficient generator.

    Args:
        device: PyTorch device where generated profiles are stored.
    """

    def __init__(self, device):
        """Create a CFS parameter container.

        Args:
            device: PyTorch device where generated profiles are stored.
        """
        self.alpha = CFSParameter(ID='alpha', scalingprofile='constant')
        self.kappa = CFSParameter(ID='kappa', scalingprofile='constant', min=1, max=1)
        self.sigma = CFSParameter(ID='sigma', scalingprofile='quartic', min=0, max=None)
        self.device = device

    def calculate_sigmamax(self, d, er, mr):
        """Calculate the maximum PML sigma value.

        Args:
            d: Spatial grid spacing normal to the PML boundary.
            er: Average relative permittivity near the boundary.
            mr: Average relative permeability near the boundary.
        """
        with torch.no_grad():
            m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
            self.sigma.max = (0.8 * (m + 1)) / (((m0 / e0) ** 0.5) * d * torch.sqrt(er * mr))


    def scaling_polynomial(self, order, Evalues, Hvalues):
        """Create staggered electric and magnetic polynomial profiles.

        Args:
            order: Polynomial order index.
            Evalues: Electric profile tensor to fill.
            Hvalues: Magnetic profile tensor to fill.
        """
        tmp = (torch.linspace(0, (len(Evalues) - 1) + 0.5, steps=2 * len(Evalues)) / (len(Evalues) - 1)) ** order
        Evalues = tmp[0:-1:2].to(self.device)
        Hvalues = tmp[1::2].to(self.device)
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):
        """Calculate electric and magnetic CFS values for one parameter.

        Args:
            thickness: PML thickness in grid cells.
            parameter: CFSParameter object to evaluate.
        """

        Evalues = torch.zeros(thickness + 1, device=self.device)
        Hvalues = torch.zeros(thickness + 1, device=self.device)
        if parameter.scalingprofile == 'constant':
            Evalues += parameter.max
            Hvalues += parameter.max
        elif parameter.scaling == 'polynomial':
            Evalues, Hvalues = self.scaling_polynomial(CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues)
            if parameter.ID == 'alpha':
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == 'kappa':
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == 'sigma':
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]
        
        return Evalues, Hvalues

def calculate_pml_update_coeffs(cfs,R1,R2, aver, avmr, dt,d,thickness):
    """Fill PML electric and magnetic update coefficient tensors.

    Args:
        cfs: CFS coefficient generator.
        R1: Electric-field PML coefficient tensor to fill.
        R2: Magnetic-field PML coefficient tensor to fill.
        aver: Average relative permittivity near the boundary.
        avmr: Average relative permeability near the boundary.
        dt: Time step size.
        d: Spatial grid spacing normal to the PML boundary.
        thickness: PML thickness in grid cells.
    """
    if not cfs.sigma.max:
        cfs.calculate_sigmamax(d, aver, avmr)

    Ealpha, Halpha = cfs.calculate_values(thickness, cfs.alpha)
    Ekappa, Hkappa = cfs.calculate_values(thickness, cfs.kappa)
    Esigma, Hsigma = cfs.calculate_values(thickness, cfs.sigma)

    R1=R1.contiguous()
    R2=R2.contiguous()

    tmp = (2 * e0 * Ekappa) + dt * (Ealpha * Ekappa + Esigma)
    R1[0,0, :] = (2 * e0 + dt * Ealpha) / tmp
    R1[1,0, :] = (2 * e0 * Ekappa) / tmp
    R1[2,0, :] = ((2 * e0 * Ekappa) - dt * (Ealpha * Ekappa + Esigma)) / tmp
    R1[3,0, :] = (2 * Esigma * dt) / (Ekappa * tmp)

    tmp = (2 * e0 * Hkappa) + dt * (Halpha * Hkappa + Hsigma)
    R2[0,0, :] = (2 * e0 + dt * Halpha) / tmp
    R2[1,0, :] = (2 * e0 * Hkappa) / tmp
    R2[2,0, :] = ((2 * e0 * Hkappa) - dt * (Halpha * Hkappa + Hsigma)) / tmp
    R2[3,0, :] = (2 * Hsigma * dt) / (Hkappa * tmp)



def build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device):
    """Create or reuse CPML auxiliary phi tensors.

    Args:
        x0: Descriptor for the low-x PML boundary.
        xm: Descriptor for the high-x PML boundary.
        y0: Descriptor for the low-y PML boundary.
        ym: Descriptor for the high-y PML boundary.
        z0: Descriptor for the low-z PML boundary.
        zm: Descriptor for the high-z PML boundary.
        nstep: Number of shots or simulations in the batch.
        PML: Existing tuple of 24 PML phi tensors, or None.
        device: PyTorch device for allocated tensors.
    """
   
    (x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2,
    xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2,
    y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2,
    ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2,
    z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2,
    zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2) = [torch.empty(0) for _ in range(24)]

    descriptors = (x0, xm, y0, ym, z0, zm)
    if PML is None:
        PML = (None,) * 24
    elif not isinstance(PML, (list, tuple)) or len(PML) != 24:
        raise ValueError("PML must contain exactly 24 CPML auxiliary tensors.")

    for face, descriptor in enumerate(descriptors):
        group = PML[4 * face:4 * face + 4]
        supplied = tuple(value is not None for value in group)
        if any(supplied) and not all(supplied):
            raise ValueError(
                f"CPML face {face} must provide all four auxiliary tensors or none."
            )
        if all(supplied) and not all(torch.is_tensor(value) for value in group):
            raise TypeError(f"CPML face {face} state must contain PyTorch tensors.")
        if (
            all(supplied)
            and descriptor.numel() == 0
            and any(value.numel() != 0 for value in group)
        ):
            raise ValueError(
                f"CPML face {face} state must be empty for a zero-thickness boundary."
            )

    if x0.numel()!=0 and PML[0]==None:
        x0EPhi1=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)),dtype=torch.float, device=device)
        x0EPhi2=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])),dtype=torch.float, device=device)
        x0HPhi1=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])), dtype=torch.float, device=device)
        x0HPhi2=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)), dtype=torch.float, device=device)
    elif PML[0]!=None:
        x0EPhi1=PML[0].contiguous()
        x0EPhi2=PML[1].contiguous()
        x0HPhi1=PML[2].contiguous()
        x0HPhi2=PML[3].contiguous()

    if xm.numel()!=0 and PML[4]==None:
        xmEPhi1=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)
        xmEPhi2=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi1=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi2=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)    
    elif PML[4]!=None:
        xmEPhi1=PML[4].contiguous()
        xmEPhi2=PML[5].contiguous()
        xmHPhi1=PML[6].contiguous()
        xmHPhi2=PML[7].contiguous()

    if y0.numel()!=0 and PML[8]==None:
        y0EPhi1=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3]+1, y0[6]-y0[5]+1), dtype=torch.float, device=device)
        y0EPhi2=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3]+1, y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi1=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3], y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi2=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3], y0[6]-y0[5]+1), dtype=torch.float, device=device)
    elif PML[8]!=None:
        y0EPhi1=PML[8].contiguous()
        y0EPhi2=PML[9].contiguous()
        y0HPhi1=PML[10].contiguous()
        y0HPhi2=PML[11].contiguous()

    if ym.numel()!=0 and PML[12]==None:
        ymEPhi1=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3]+1, ym[6]-ym[5]+1),dtype=torch.float, device=device)
        ymEPhi2=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3]+1, ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi1=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3], ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi2=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3], ym[6]-ym[5]+1), dtype=torch.float, device=device)
    elif PML[12]!=None:
        ymEPhi1=PML[12].contiguous()
        ymEPhi2=PML[13].contiguous()
        ymHPhi1=PML[14].contiguous()
        ymHPhi2=PML[15].contiguous()

    if z0.numel()!=0 and PML[16]==None:
        z0EPhi1=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0EPhi2=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0HPhi1=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]), dtype=torch.float, device=device)
        z0HPhi2=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]), dtype=torch.float, device=device)
    elif PML[16]!=None:
        z0EPhi1=PML[16].contiguous()
        z0EPhi2=PML[17].contiguous()
        z0HPhi1=PML[18].contiguous()
        z0HPhi2=PML[19].contiguous()

    if zm.numel()!=0 and PML[20]==None:
        zmEPhi1=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmEPhi2=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmHPhi1=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]), dtype=torch.float, device=device)
        zmHPhi2=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]), dtype=torch.float, device=device)
    elif PML[20]!=None:
        zmEPhi1=PML[20].contiguous()
        zmEPhi2=PML[21].contiguous()
        zmHPhi1=PML[22].contiguous()
        zmHPhi2=PML[23].contiguous()

    tensors = (
        x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,
        xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,
        y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,
        ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,
        z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,
        zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,
    )

    def expected_shapes(axis, descriptor):
        if descriptor.numel() == 0:
            return (None,) * 4
        a = int(descriptor[2] - descriptor[1])
        b = int(descriptor[4] - descriptor[3])
        c = int(descriptor[6] - descriptor[5])
        if axis == 0:
            return (
                (nstep, a + 1, b, c + 1),
                (nstep, a + 1, b + 1, c),
                (nstep, a, b + 1, c),
                (nstep, a, b, c + 1),
            )
        if axis == 1:
            return (
                (nstep, a, b + 1, c + 1),
                (nstep, a + 1, b + 1, c),
                (nstep, a + 1, b, c),
                (nstep, a, b, c + 1),
            )
        return (
            (nstep, a, b + 1, c + 1),
            (nstep, a + 1, b, c + 1),
            (nstep, a + 1, b, c),
            (nstep, a, b + 1, c),
        )

    for face, descriptor in enumerate(descriptors):
        group = tensors[4 * face:4 * face + 4]
        for component, (tensor, expected) in enumerate(
            zip(group, expected_shapes(face // 2, descriptor))
        ):
            if expected is None:
                if tensor.numel() != 0:
                    raise ValueError(
                        f"CPML face {face} component {component} must be empty."
                    )
            elif tuple(tensor.shape) != expected:
                raise ValueError(
                    f"CPML face {face} component {component} has shape "
                    f"{tuple(tensor.shape)}; expected {expected}."
                )

    return tuple(
        tensor.to(device=device, dtype=torch.float32).contiguous()
        for tensor in tensors
    )


def checkpoint_initial_field(device=None,per_nstep=None, dx=None, dt=None, 
            source_amplitudes=None,
            source_location=None, 
            receiver_location=None, 
            er=None, se=None,mr=None, 
            pmlthick=10, fdtd_order=2):
    """Create initial electric, magnetic, and PML field checkpoints.

    Args:
        device: PyTorch device where tensors will be allocated.
        per_nstep: Optional number of shots to keep from the initialized fields.
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step size.
        source_amplitudes: Source waveform tensor.
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        er: Relative permittivity tensor.
        se: Electrical conductivity tensor.
        mr: Relative permeability tensor, or None to use ones.
        pmlthick: PML thickness as an int, list, or tensor.
        fdtd_order: Spatial finite-difference order used for the CFL check.
    """
    E=None
    H=None
    PML=None

    er,se,nx,ny,nz,_,nstep,_,_,_,_,mr,_,dtype,pmlthick,source_amplitudes=initialization(
        device,er,se,mr,source_amplitudes,source_location,receiver_location,
        dx,dt,pmlthick,fdtd_order)

    Ex,Ey,Ez=create_or_separate(E,nx,ny,nz,nstep,device,dtype)
    Hx,Hy,Hz=create_or_separate(H,nx,ny,nz,nstep,device,dtype)

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=build_pml_coeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype)


    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    del x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2

    if per_nstep==None:
        return (Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2)
    elif er.shape[2]==1:
        return (Ex[:per_nstep,:,:,:],Ey[:per_nstep,:,:,:],Ez[:per_nstep,:,:,:]),(Hx[:per_nstep,:,:,:],Hy[:per_nstep,:,:,:],Hz[:per_nstep,:,:,:]),(x0EPhi1[:per_nstep,:,:,:],x0EPhi2[:per_nstep,:,:,:],x0HPhi1[:per_nstep,:,:,:],x0HPhi2[:per_nstep,:,:,:],xmEPhi1[:per_nstep,:,:,:],xmEPhi2[:per_nstep,:,:,:],xmHPhi1[:per_nstep,:,:,:],xmHPhi2[:per_nstep,:,:,:],y0EPhi1[:per_nstep,:,:,:],y0EPhi2[:per_nstep,:,:,:],y0HPhi1[:per_nstep,:,:,:],y0HPhi2[:per_nstep,:,:,:],ymEPhi1[:per_nstep,:,:,:],ymEPhi2[:per_nstep,:,:,:],ymHPhi1[:per_nstep,:,:,:],ymHPhi2[:per_nstep,:,:,:],z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2)
    else:
        return (Ex[:per_nstep,:,:,:],Ey[:per_nstep,:,:,:],Ez[:per_nstep,:,:,:]),(Hx[:per_nstep,:,:,:],Hy[:per_nstep,:,:,:],Hz[:per_nstep,:,:,:]),(x0EPhi1[:per_nstep,:,:,:],x0EPhi2[:per_nstep,:,:,:],x0HPhi1[:per_nstep,:,:,:],x0HPhi2[:per_nstep,:,:,:],xmEPhi1[:per_nstep,:,:,:],xmEPhi2[:per_nstep,:,:,:],xmHPhi1[:per_nstep,:,:,:],xmHPhi2[:per_nstep,:,:,:],y0EPhi1[:per_nstep,:,:,:],y0EPhi2[:per_nstep,:,:,:],y0HPhi1[:per_nstep,:,:,:],y0HPhi2[:per_nstep,:,:,:],ymEPhi1[:per_nstep,:,:,:],ymEPhi2[:per_nstep,:,:,:],ymHPhi1[:per_nstep,:,:,:],ymHPhi2[:per_nstep,:,:,:],z0EPhi1[:per_nstep,:,:,:],z0EPhi2[:per_nstep,:,:,:],z0HPhi1[:per_nstep,:,:,:],z0HPhi2[:per_nstep,:,:,:],zmEPhi1[:per_nstep,:,:,:],zmEPhi2[:per_nstep,:,:,:],zmHPhi1[:per_nstep,:,:,:],zmHPhi2[:per_nstep,:,:,:])


def zero_field(*tensors):
    """Return zero tensors with the same shapes as the input tensors.

    Args:
        *tensors: Tensors to zero, or None values to preserve.
    """
    return tuple(torch.zeros_like(t) if t is not None else None for t in tensors)
