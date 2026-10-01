"""Convolutional PML (CPML) profiles, update coefficients and auxiliary state.

The CPML is a fixed numerical boundary, not an inversion parameter. Boundary
material averages are detached, and the native material-gradient kernels
exclude CPML cells. All arithmetic in this module is carried over unchanged
from the original implementation so coefficients are bit-for-bit identical.

Face order everywhere is ``x0, xm, y0, ym, z0, zm``. A face *descriptor* is a
7-element ``int32`` tensor ``(thickness, x_start, x_end, y_start, y_end,
z_start, z_end)`` in extended-grid cells; disabled faces use an empty tensor.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from ..config.constants import (
    CPML_COEFFICIENTS_PER_CELL,
    CPML_FACE_COUNT,
    CPML_PHI_TENSOR_COUNT,
    CPML_PHI_TENSORS_PER_FACE,
    CPML_POLE_COUNT,
    CPML_SCALING_PROFILES,
    CPML_SIGMA_MAX_FACTOR,
    VACUUM_PERMEABILITY,
    VACUUM_PERMITTIVITY,
)
from ..preprocessing.grid import GridSpacingLike, normalize_grid_spacing

# Legacy short names, kept because the coefficient formulas below read best
# with them and because they were public module attributes before.
m0 = VACUUM_PERMEABILITY
e0 = VACUUM_PERMITTIVITY

ShapeTuple = Tuple[int, ...]


class CFSParameter:
    """One parameter (alpha, kappa or sigma) of a complex-frequency-shifted PML.

    Args:
        ID: Parameter name, ``"alpha"``, ``"kappa"`` or ``"sigma"``.
        scaling: Scaling family used to generate the profile.
        scalingprofile: Polynomial order name (see :data:`scalingprofiles`).
        min: Minimum value of the profile.
        max: Maximum value of the profile (``None`` = compute automatically).
    """

    scalingprofiles = CPML_SCALING_PROFILES

    def __init__(self, ID=None, scaling="polynomial", scalingprofile=None, min=0, max=0):
        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        return (
            f"CFSParameter(ID={self.ID!r}, scaling={self.scaling!r}, "
            f"scalingprofile={self.scalingprofile!r}, min={self.min!r}, max={self.max!r})"
        )


class CFS:
    """Complex-frequency-shifted PML profile generator (gprMax convention).

    Defaults: constant ``alpha = 0``, constant ``kappa = 1`` and a quartic
    ``sigma`` graded from 0 to the optimal ``sigma_max``.

    Args:
        device: Device on which the profiles are generated.
    """

    def __init__(self, device):
        self.alpha = CFSParameter(ID="alpha", scalingprofile="constant")
        self.kappa = CFSParameter(ID="kappa", scalingprofile="constant", min=1, max=1)
        self.sigma = CFSParameter(ID="sigma", scalingprofile="quartic", min=0, max=None)
        self.device = device

    def calculate_sigmamax(self, d, er, mr):
        """Set ``sigma.max`` to the optimal value ``0.8 (m+1) / (eta0 d sqrt(er mr))``.

        Args:
            d: Grid spacing normal to the boundary [m].
            er: Average relative permittivity next to the boundary.
            mr: Average relative permeability next to the boundary.
        """
        with torch.no_grad():
            m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
            self.sigma.max = (CPML_SIGMA_MAX_FACTOR * (m + 1)) / (
                ((m0 / e0) ** 0.5) * d * torch.sqrt(er * mr)
            )

    def scaling_polynomial(self, order, Evalues, Hvalues):
        """Return staggered electric/magnetic samples of ``x ** order``.

        Args:
            order: Polynomial order.
            Evalues: Electric profile template (only its length is used).
            Hvalues: Magnetic profile template (unused, kept for API compatibility).
        """
        tmp = (
            torch.linspace(0, (len(Evalues) - 1) + 0.5, steps=2 * len(Evalues)) / (len(Evalues) - 1)
        ) ** order
        Evalues = tmp[0:-1:2].to(self.device)
        Hvalues = tmp[1::2].to(self.device)
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):
        """Return the electric and magnetic profile of ``parameter``.

        Args:
            thickness: PML thickness in cells.
            parameter: The :class:`CFSParameter` to evaluate.

        Returns:
            ``(Evalues, Hvalues)``, each of length ``thickness``.
        """
        Evalues = torch.zeros(thickness + 1, device=self.device)
        Hvalues = torch.zeros(thickness + 1, device=self.device)
        if parameter.scalingprofile == "constant":
            Evalues += parameter.max
            Hvalues += parameter.max
        elif parameter.scaling == "polynomial":
            Evalues, Hvalues = self.scaling_polynomial(
                CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues
            )
            if parameter.ID == "alpha":
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == "kappa":
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == "sigma":
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]
        return Evalues, Hvalues


def calculate_pml_update_coeffs(cfs, R1, R2, aver, avmr, dt, d, thickness):
    """Fill the CPML update coefficients ``RA, RB, RE, RF`` in place.

    Args:
        cfs: :class:`CFS` profile generator.
        R1: Electric coefficients, shape ``(4, 1, thickness)``, filled in place.
        R2: Magnetic coefficients, shape ``(4, 1, thickness)``, filled in place.
        aver: Average relative permittivity next to the boundary.
        avmr: Average relative permeability next to the boundary.
        dt: Time step [s].
        d: Grid spacing normal to the boundary [m].
        thickness: PML thickness in cells.
    """
    if not cfs.sigma.max:
        cfs.calculate_sigmamax(d, aver, avmr)

    Ealpha, Halpha = cfs.calculate_values(thickness, cfs.alpha)
    Ekappa, Hkappa = cfs.calculate_values(thickness, cfs.kappa)
    Esigma, Hsigma = cfs.calculate_values(thickness, cfs.sigma)

    R1 = R1.contiguous()
    R2 = R2.contiguous()

    tmp = (2 * e0 * Ekappa) + dt * (Ealpha * Ekappa + Esigma)
    R1[0, 0, :] = (2 * e0 + dt * Ealpha) / tmp
    R1[1, 0, :] = (2 * e0 * Ekappa) / tmp
    R1[2, 0, :] = ((2 * e0 * Ekappa) - dt * (Ealpha * Ekappa + Esigma)) / tmp
    R1[3, 0, :] = (2 * Esigma * dt) / (Ekappa * tmp)

    tmp = (2 * e0 * Hkappa) + dt * (Halpha * Hkappa + Hsigma)
    R2[0, 0, :] = (2 * e0 + dt * Halpha) / tmp
    R2[1, 0, :] = (2 * e0 * Hkappa) / tmp
    R2[2, 0, :] = ((2 * e0 * Hkappa) - dt * (Halpha * Hkappa + Hsigma)) / tmp
    R2[3, 0, :] = (2 * Hsigma * dt) / (Hkappa * tmp)


def _face_descriptor_values(face: int, thickness: int, nx: int, ny: int, nz: int) -> ShapeTuple:
    """Return ``(p, x0, x1, y0, y1, z0, z1)`` of one CPML face on the extended grid."""
    p = thickness
    return (
        (p, 0, p, 0, ny, 0, nz),
        (p, nx - p, nx, 0, ny, 0, nz),
        (p, 0, nx, 0, p, 0, nz),
        (p, 0, nx, ny - p, ny, 0, nz),
        (p, 0, nx, 0, ny, 0, p),
        (p, 0, nx, 0, ny, nz - p, nz),
    )[face]


def _boundary_material_index(face: int, thickness: int, nx: int, ny: int, nz: int) -> tuple:
    """Index of the material plane averaged for a face's ``sigma_max``."""
    return (
        (0, slice(None, ny), slice(None, nz)),
        (nx - thickness, slice(None, ny), slice(None, nz)),
        (slice(None, nx), 0, slice(None, nz)),
        (slice(None, nx), ny - thickness, slice(None, nz)),
        (slice(None, nx), slice(None, ny), 0),
        (slice(None, nx), slice(None, ny), nz - thickness),
    )[face]


def pml_face_descriptors(nx: int, ny: int, nz: int, pmlthick: Any) -> Tuple[torch.Tensor, ...]:
    """Return the six CPML face descriptors without computing coefficients.

    Args:
        nx, ny, nz: Extended grid dimensions.
        pmlthick: Six face thicknesses (tensor or sequence).
    """
    pml = tuple(
        int(value) for value in (pmlthick.tolist() if torch.is_tensor(pmlthick) else pmlthick)
    )
    return tuple(
        torch.tensor(_face_descriptor_values(face, pml[face], nx, ny, nz), dtype=torch.int)
        if pml[face] > 0
        else torch.empty(0)
        for face in range(CPML_FACE_COUNT)
    )


def build_pml_coeffs(eps_r, mu_r, dt, dx: GridSpacingLike, nx, ny, nz, pmlthick, device, dtype):
    """Build CPML face descriptors and electric/magnetic update coefficients.

    Args:
        eps_r: Relative permittivity on the extended grid ``(nx, ny, nz)``.
        mu_r: Relative permeability (extended grid, optionally with halo).
        dt: Time step [s].
        dx: Scalar grid spacing or ``(dx, dy, dz)`` [m].
        nx, ny, nz: Extended grid dimensions.
        pmlthick: Six face thicknesses.
        device: Device of the coefficient tensors.
        dtype: Dtype of the coefficient tensors.

    Returns:
        ``(x0, xm, y0, ym, z0, zm, x01, x02, xm1, xm2, y01, y02, ym1, ym2,
        z01, z02, zm1, zm2)``: six descriptors followed by the electric
        (``*1``) and magnetic (``*2``) coefficients of each face, each of
        shape ``(4, 1, thickness)``; disabled faces yield empty tensors.
    """
    spacing = normalize_grid_spacing(dx)
    eps_r_fixed = eps_r.detach()
    mu_r_fixed = mu_r.detach()
    pml = tuple(int(value) for value in pmlthick.tolist())

    descriptors: List[torch.Tensor] = []
    coefficients: List[torch.Tensor] = []
    for face in range(CPML_FACE_COUNT):
        thickness = pml[face]
        if thickness <= 0:
            descriptors.append(torch.empty(0))
            coefficients.extend((torch.empty(0), torch.empty(0)))
            continue
        descriptors.append(
            torch.tensor(_face_descriptor_values(face, thickness, nx, ny, nz), dtype=torch.int)
        )
        index = _boundary_material_index(face, thickness, nx, ny, nz)
        average_eps = eps_r_fixed[index].mean()
        average_mu = mu_r_fixed[index].mean()
        cfs = CFS(device=device)
        electric = torch.zeros(
            (CPML_COEFFICIENTS_PER_CELL, CPML_POLE_COUNT, thickness), device=device, dtype=dtype
        )
        magnetic = torch.zeros(
            (CPML_COEFFICIENTS_PER_CELL, CPML_POLE_COUNT, thickness), device=device, dtype=dtype
        )
        calculate_pml_update_coeffs(
            cfs, electric, magnetic, average_eps, average_mu, dt, spacing[face // 2], thickness
        )
        coefficients.extend((electric, magnetic))

    return (*descriptors, *coefficients)


def buildpmlcoeffs(*args, **kwargs):
    """Deprecated alias of :func:`build_pml_coeffs` (accepts ``er``/``mr`` keywords)."""
    if "er" in kwargs:
        kwargs["eps_r"] = kwargs.pop("er")
    if "mr" in kwargs:
        kwargs["mu_r"] = kwargs.pop("mr")
    return build_pml_coeffs(*args, **kwargs)


def expected_phi_shapes(
    face: int, descriptor: torch.Tensor, nstep: int
) -> Tuple[Optional[ShapeTuple], ...]:
    """Return the shapes of the four auxiliary arrays of one CPML face.

    The order is ``(E_phi1, E_phi2, H_phi1, H_phi2)``; ``None`` marks a
    disabled face.

    Args:
        face: Face index 0..5 (x0, xm, y0, ym, z0, zm).
        descriptor: Face descriptor from :func:`pml_face_descriptors`.
        nstep: Number of shots.
    """
    if descriptor.numel() == 0:
        return (None,) * CPML_PHI_TENSORS_PER_FACE
    a = int(descriptor[2] - descriptor[1])
    b = int(descriptor[4] - descriptor[3])
    c = int(descriptor[6] - descriptor[5])
    axis = face // 2
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


def build_pml_phi(x0, xm, y0, ym, z0, zm, nstep, PML, device):
    """Allocate or validate the 24 CPML auxiliary (phi) tensors.

    Args:
        x0, xm, y0, ym, z0, zm: Face descriptors.
        nstep: Number of shots.
        PML: ``None`` to allocate zeros, or 24 tensors from a previous
            :func:`DeepGPR.compute` / :func:`DeepGPR.checkpoint_initial_field`
            call (``E_phi1, E_phi2, H_phi1, H_phi2`` per face).
        device: Device for newly allocated tensors.

    Returns:
        A tuple of 24 contiguous float32 tensors. Supplied tensors that are
        already float32 and on ``device`` are returned *as is* (the native
        solver advances them in place).

    Raises:
        ValueError, TypeError: For incomplete faces or mismatching shapes.
    """
    descriptors = (x0, xm, y0, ym, z0, zm)
    if PML is None:
        PML = (None,) * CPML_PHI_TENSOR_COUNT
    elif not isinstance(PML, (list, tuple)) or len(PML) != CPML_PHI_TENSOR_COUNT:
        raise ValueError("PML must contain exactly 24 CPML auxiliary tensors.")

    for face, descriptor in enumerate(descriptors):
        group = PML[CPML_PHI_TENSORS_PER_FACE * face : CPML_PHI_TENSORS_PER_FACE * (face + 1)]
        supplied = tuple(value is not None for value in group)
        if any(supplied) and not all(supplied):
            raise ValueError(f"CPML face {face} must provide all four auxiliary tensors or none.")
        if all(supplied) and not all(torch.is_tensor(value) for value in group):
            raise TypeError(f"CPML face {face} state must contain PyTorch tensors.")
        if all(supplied) and descriptor.numel() == 0 and any(value.numel() != 0 for value in group):
            raise ValueError(f"CPML face {face} state must be empty for a zero-thickness boundary.")

    tensors = []
    for face, descriptor in enumerate(descriptors):
        start = CPML_PHI_TENSORS_PER_FACE * face
        group = PML[start : start + CPML_PHI_TENSORS_PER_FACE]
        shapes = expected_phi_shapes(face, descriptor, nstep)
        if group[0] is not None:
            tensors.extend(value.contiguous() for value in group)
        elif descriptor.numel() != 0:
            tensors.extend(torch.zeros(shape, dtype=torch.float, device=device) for shape in shapes)
        else:
            tensors.extend(torch.empty(0) for _ in range(CPML_PHI_TENSORS_PER_FACE))

    for face, descriptor in enumerate(descriptors):
        start = CPML_PHI_TENSORS_PER_FACE * face
        group = tensors[start : start + CPML_PHI_TENSORS_PER_FACE]
        for component, (tensor, expected) in enumerate(
            zip(group, expected_phi_shapes(face, descriptor, nstep))
        ):
            if expected is None:
                if tensor.numel() != 0:
                    raise ValueError(f"CPML face {face} component {component} must be empty.")
            elif tuple(tensor.shape) != expected:
                raise ValueError(
                    f"CPML face {face} component {component} has shape "
                    f"{tuple(tensor.shape)}; expected {expected}."
                )

    return tuple(tensor.to(device=device, dtype=torch.float32).contiguous() for tensor in tensors)


def pml_phi_element_count(nx: int, ny: int, nz: int, nstep: int, pml: Sequence[int]) -> int:
    """Return the exact number of float32 elements in all CPML phi tensors.

    Args:
        nx, ny, nz: Extended grid dimensions.
        nstep: Number of shots.
        pml: Six face thicknesses.
    """
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


__all__ = [
    "CFS",
    "CFSParameter",
    "build_pml_coeffs",
    "build_pml_phi",
    "buildpmlcoeffs",
    "calculate_pml_update_coeffs",
    "e0",
    "expected_phi_shapes",
    "m0",
    "pml_face_descriptors",
    "pml_phi_element_count",
]
