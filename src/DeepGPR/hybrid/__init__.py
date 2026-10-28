"""Hybrid DGTD / Yee-FDTD Maxwell solvers for GPR with explicit topography.

2D Ez-TM (DeepGPR's 2D mode, axis 0 = depth ``x``, axis 1 = horizontal ``y``):
a band of body-fitted triangles (nodal DGTD) carries the air-ground surface
and is coupled to DeepGPR's Yee grid above and below by an energy-neutral
mortar. See ``docs/HYBRID_DGTD.md`` for the formulation and verification.

Main entry points:

* :class:`HybridTM2D` - forward solver (hybrid, pure Yee or pure DGTD);
* :class:`Surface` - flat, sine, slope, sampled or elevation-data surfaces;
* :class:`FieldSampler` - Ez on a regular display grid (snapshots, GIFs);
* :func:`make_propagator` - ``solver="fdtd"`` or ``"hybrid_dgtd"``: a
  differentiable ``data = prop(eps_r, sigma, wavelet)`` for FWI (discrete
  adjoint for the hybrid, native adjoint for FDTD).

3D (DeepGPR's 3D layout, ``x`` depth, ``y``/``z`` horizontal): a band of
body-fitted tetrahedra (nodal DGTD, fused CUDA kernels) coupled to the 3D Yee
grid; :class:`HybridMaxwell3D`, :class:`Surface3D`, :class:`TetMesh`.
"""

from .adjoint import HybridPropagator
from .mesh import TriMesh, delaunay_band_mesh, terrain_following_band_mesh
from .mesh3d import TetMesh, terrain_following_tet_mesh
from .propagators import FDTDPropagator, make_propagator, resample
from .solver import FieldSampler, HybridTM2D
from .solver3d import HybridMaxwell3D
from .topography import CallableSurface, SampledSurface, Surface, Surface3D

__all__ = [
    "CallableSurface",
    "FDTDPropagator",
    "HybridMaxwell3D",
    "HybridPropagator",
    "make_propagator",
    "resample",
    "FieldSampler",
    "HybridTM2D",
    "SampledSurface",
    "Surface",
    "Surface3D",
    "TetMesh",
    "TriMesh",
    "delaunay_band_mesh",
    "terrain_following_band_mesh",
    "terrain_following_tet_mesh",
]
