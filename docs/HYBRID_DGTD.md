# Hybrid DGTD–Yee-FDTD solver for GPR with topography (2D and 3D)

`DeepGPR.hybrid` adds an optional forward solver in which a band of body-fitted
triangles (nodal discontinuous Galerkin, DGTD) carries the air–ground surface,
and DeepGPR's Yee-FDTD grid covers the regular air above and the subsurface
below. The two discretisations are coupled through horizontal interfaces by an
energy-neutral mortar, so one leapfrog step advances the whole system.

| Phase | Content | Status |
|---|---|---|
| 1 | 2D flat DGTD–FDTD interface | implemented, verified (below) |
| 2 | 2D topography (body-fitted band) | implemented and verified: models A/B/C + slope, staircase and averaged Yee, full DGTD P1/P2 and hybrid P1/P2 against a converged reference (accuracy, B-scans, time, memory, unknowns, steps) |
| 3 | Discrete adjoint, dual-parameter FWI | implemented and verified (Section 8): exact discrete adjoint (dot-product test to round-off, finite differences to 1e-10, native-adjoint cross-check), `make_propagator(solver=...)`, FWI on topography |
| 4 | 3D tetrahedral DGTD + CUDA kernels | implemented and verified (Section 11): 3D Yee mirror = native DeepGPR, energy-conserving 3D mortar, fused CUDA DG kernels with exact transposes, stable side absorber (UPML), interface penalty, 3D discrete adjoint (finite differences to 1e-10, native-adjoint cross-check), benchmark against a converged reference, small 3D FWI; see the honest cost/accuracy summary in 11.5 |

The existing `DeepGPR.compute` API and native libraries are unchanged; the
hybrid is a separate module.

## 1. Repository findings that shaped the design

* 2D DeepGPR is the **Ez-TM** mode on the `(x, y)` plane with arrays
  `eps_r[i, j]`; axis 0 (`x`) is used as depth in the examples, axis 1 (`y`) as
  the horizontal position. Field arrays live on the extended grid (CPML) plus
  a one-cell halo; `Ez[i, j]` at `(i dx, j dy)`, `Hx[i, j]` at
  `(i dx, (j+1/2) dy)`, `Hy[i, j]` at `((i+1/2) dx, j dy)`; Ez on the outer
  rows/columns is PEC.
* One native step is: H update + CPML (from `E^n`), E update + CPML, soft
  source `Ez -= cb w[n] / (dx dy)`, receivers record `E^(n+1)`. Conductivity
  is Crank–Nicolson (`ca = (eps/dt - sigma/2)/(eps/dt + sigma/2)`).
* The native ABI (`forward`/`backward`) runs **whole time loops** only. The
  coupling needs an exchange between the H and E half steps, so the hybrid's
  Yee part is a PyTorch mirror of the native second-order update
  (`DeepGPR/hybrid/yee.py`) that uses DeepGPR's own CPML coefficients
  (`build_pml_coeffs`) and material extension (`_ExtendModel`). It agrees
  with `DeepGPR.compute` to float32 rounding (`4.8e-7` relative,
  anisotropic `dx != dy`, asymmetric CPML, conductivity, two shots;
  `tests/test_hybrid_dgtd.py::PureYeeModeTest`). A native per-step entry
  point / fused CUDA kernels are deferred to Phase 4.
* The hybrid uses the second-order Yee stencil only (`fdtd_order=2`); the
  interface closure below is derived for it.

## 2. Equations

DeepGPR convention (SI, `x` = axis 0, `y` = axis 1):

```
eps dEz/dt + sigma Ez = dHy/dx - dHx/dy - Jz
mu  dHx/dt            = -dEz/dy
mu  dHy/dt            =  dEz/dx
```

The tangential magnetic field on an edge with unit normal `n` is
`H_t = Hy n_x - Hx n_y`, and the energy identity is
`d/dt 1/2 ∫(eps Ez² + mu |H|²) = ∮ Ez H_t ds - ∫ sigma Ez²`.

## 3. DGTD band (`dgtd.py`, `reference.py`, `mesh.py`)

* Nodal DG of order `N` on triangles (warp-and-blend nodes, orthonormal
  Dubiner basis, `Dr`, `Ds`, `M`, `LIFT`), affine elements, element-wise
  constant `eps_r`, `sigma`, `mu_r` (Hesthaven & Warburton, *Nodal
  Discontinuous Galerkin Methods*, Springer 2008, doi:10.1007/978-0-387-72067-8).
  P1 is the default; P2 is supported and tested (orders up to 6 build).
* Strong form with DG derivatives that include their surface terms:
  `D_x E = ∂x E + LIFT(Fscale n_x (E* − E⁻))`, likewise `D_y E`, and
  `D_x Hy − D_y Hx = ∂x Hy − ∂y Hx + LIFT(Fscale (H_t* − H_t⁻))`.
* Fluxes (Riemann solution of the 1D normal problem, `Y = sqrt(eps/mu)`):
  * central: `E* = {E}`, `H* = {H}` — energy conserving
    (Fezoui, Lanteri, Lohrengel & Piperno, M2AN 39(6), 2005,
    doi:10.1051/m2an:2005049);
  * upwind: `E* = (Y⁻E⁻ + Y⁺E⁺ + H_t⁺ − H_t⁻)/(Y⁻+Y⁺)`,
    `H_t* = (Z⁻H_t⁻ + Z⁺H_t⁺ + E⁺ − E⁻)/(Z⁻+Z⁺)`;
  * `flux=alpha` blends them: central plus `alpha` times the upwind penalty.
    The energy rate is linear in the fluxes, so the blend dissipates
    `alpha` times the upwind dissipation.
* Boundary faces: PEC (`E* = 0`), first-order Silver–Müller absorbing
  (upwind flux against a zero exterior state), or **interface** (Section 4).
* Time stepping: the same staggered leapfrog as Yee, `H^(n+1/2)` from `E^n`,
  `E^(n+1) = ca E^n + cb (D_x Hy − D_y Hx)^(n+1/2)` with the Crank–Nicolson
  conductivity coefficients of DeepGPR. Upwind penalty terms are evaluated
  explicitly (E-penalty with `E^n`, H-penalty with `H^(n-1/2)`).
* CPML: the DG derivatives are stretched exactly as DeepGPR stretches the Yee
  differences: `d -> RA d + RB psi`, `psi <- RE psi − RF d`, with DeepGPR's
  quartic profile, `kappa = 1`, `alpha = 0` and the fixed reference speed,
  evaluated at the DG nodes. The auxiliary variables `psi` live on the nodes
  of the CPML elements only and are advanced in the same half steps as the
  fields they stretch. The Yee-optimal `sigma_max` is too steep for DG, so
  the DG profile is scaled by `dg_pml_strength` (default 0.4, Section 7).
  In the hybrid the band crosses the side CPML columns; with
  `dg_band=(None, None)` (full DGTD) the band also owns the top and bottom CPML.

## 4. Coupling on a horizontal interface (`coupling.py`, `yee.py`)

The interface `Γ: x = x_I` is an Ez row `i_I` of the Yee grid. Rows strictly
inside the band are not updated by the Yee scheme (no physical region is
computed twice); the interface row is shared.

**Yee side: summation-by-parts closure.** The interface Ez nodes (and the
Hx nodes on the same row) get weight 1/2 in the discrete energy. Their update
uses the one-sided half-cell difference

```
eps (dx/2) dE_b/dt = ± (Hy_inner − H_b) − (dx/2) ∂y Hx   (sign by side)
```

with the tangential trace `H_b` supplied by the DGTD side. This is the regular
Yee formula with the ghost value `Hy_ghost = 2 H_b − Hy_inner`, written into
the first Hy row inside the band before the E update. The existing update
loop is therefore reused unchanged.

**Non-matching mortar.** With `l_j(y)` the hat functions of the interface Ez
nodes `y_j = j dy` and `L_k` the Lagrange basis on a DG interface face,

```
C_kj = ∫_Γ L_k(y) l_j(y) dy
```

is computed exactly (4-point Gauss on every sub-segment between DG vertices
and Yee nodes). DG and Yee spacings along the interface are independent.

* DG side: `E* = I(E_b)` (hat interpolant of the Yee row) and `H* = H⁻`. In
  the LIFT form `E*` enters as its face-wise L2 projection
  `(sJ M_f)^{-1} C E_b`, which reproduces the exact surface integral.
* Yee side: `H_b = C^T Hy_DG / dy`.

**Energy identity.** The DG interface faces contribute `± Hy_DGᵀ C E_b` to the
energy rate, the Yee interface rows `∓ dy E_bᵀ H_b = ∓ Hy_DGᵀ C E_b`; the sum
vanishes. All other DG and Yee terms are skew in their mass inner products
(central fluxes, PEC), so the coupled semi-discrete system has the form

```
M_e de/dt = K h,   M_h dh/dt = −Kᵀ e
```

and staggered leapfrog conserves
`E^n = 1/2 (e^nᵀ M_e e^n + h^(n−1/2)ᵀ M_h h^(n+1/2))`, which is positive
definite for `dt < 2 / sqrt(λ_max(M_e^-1 K M_h^-1 Kᵀ))`. This is verified to
round-off (Section 7). Conductivity (Crank–Nicolson) and the dissipative
penalty terms only remove energy; CPML is not energy-dissipative by
construction and is checked by long runs.

The interface itself uses central coupling for every `flux` choice; the
penalty acts on DG interior faces.

**One time step** (`HybridTM2D._step`):

1. Yee `H^(n+1/2)` from `Ez^n` (+ CPML);
2. DG: `E*` on interface faces from the Yee rows `Ez^n`, then DG `H^(n+1/2)`;
3. `H_b = C^T Hy_DG^(n+1/2) / dy`, ghost rows;
4. Yee `Ez^(n+1)` (+ CPML);
5. DG `E^(n+1)`;
6. sources (`w[n]`) and receivers (`E^(n+1)`), exactly as DeepGPR.

## 5. Time step and flux choice

`dt` defaults to `cfl_safety · 2/sqrt(λ_max)` of the coupled conservative
operator. `λ_max` comes from 80 Lanczos iterations in the `M_e` inner product
with full re-orthogonalisation (float64, ~0.5 s for the GIF models). The
limit is sharp: random data are stable at `0.98 × limit` and blow up at
`1.03 × limit` (`CFLTest`). The DG part sets the limit; for P1 with `h = dx` it is
about 0.3–0.4 of the Yee limit.

Explicit penalty terms shrink the stable step. Measured stable fraction of the
central limit (bisection, random initial data, 15000 steps, P1/P2):

| alpha | 0 | 0.05 | 0.10 | 0.25 | 0.50 | 1 (upwind) |
|---|---|---|---|---|---|---|
| P1 | 1 | 0.96 | 0.92 | 0.82 | 0.67 | 0.46* |
| P2 | 1 | 0.96 | 0.93 | 0.84 | 0.69 | – |

\*Full upwind blew up at 0.5 in a CPML configuration and was stable at 0.3;
the default table uses 0.30 for alpha = 1.

**Default: `flux=0.1`.** Central fluxes are energy conserving but leave the
grid-scale spurious modes of DG undamped. On unstructured (Delaunay) meshes a
point source excites them strongly:

| air, source in band, 0.5–3.5 m offsets | relative L2 vs 4x finer Yee |
|---|---|
| central P1, Delaunay | 0.81 – 2.78 |
| central P2, Delaunay | 0.07 – 0.77 |
| central P1, terrain-following mesh | 0.011 – 0.037 |
| alpha = 0.1, P1, Delaunay | 0.004 – 0.012 |
| alpha = 0.25, P1, Delaunay | 0.003 – 0.009 |
| alpha = 0.1, P2 (h = 2 dx), Delaunay | 0.005 – 0.012 |

`alpha = 0.1` removes them at an 8 % time-step cost; the default
`cfl_safety` is `0.9 ×` the tabulated fraction.

## 6. Topography (`topography.py`, `mesh.py`, `solver.py`)

* `Surface.flat / sine / slope / from_samples / from_elevation` (linear or
  monotone PCHIP interpolation of elevation data, `depth = datum − z`), or any
  vectorised callable. The surface is a depth `x = d(y)` in the physical frame;
  in the side CPML it is continued flat, consistent with DeepGPR's replicate
  padding.
* Meshes put vertices **on** the surface, so the air–ground interface is a
  chain of element edges (P1 geometry, second-order accurate representation of
  a smooth curve):
  * `terrain_following_band_mesh`: stretched columns, optional grading towards
    the surface and vertex jitter (robust; thin layers under steep relief
    reduce `dt`);
  * `delaunay_band_mesh`: surface points at spacing `h_surface`, boundary
    points, and interior points from geometrically graded hexagonal lattices
    following a size function `h(d) = min(h, h_surface + 0.25 (d − w))` of the
    distance `d` to the surface (`w = refine_width`), thinned with a KD-tree
    (minimum distance 0.7 h(d)); Delaunay (SciPy). Conformity of the surface
    edges is verified (raises if an element would cut it). Minimum angles:
    about 27° for uniform meshes, 22–24° with a 2.5–2.7× refined strip;
  * any user `TriMesh` (e.g. from Gmsh) in the extended frame.
* Materials. The user gives the ground model on the regular model grid
  (`eps_r`, `sigma`, DeepGPR layout, spacing `dx`) plus the surface. DG
  elements above the surface are air; elements below take the value of the
  model pixel (the `dx` square centred on a model node) that contains their
  centroid, from the ground model continued upwards over the surface (so
  near-surface ground elements never pick up staircase air). Yee nodes take the
  value of the model pixel that contains them and are set to air above the
  surface. Both sides read the same pixels, so the interfaces add no material
  jump.
* Refined Yee part (`fdtd_refinement=r`). The Yee grid can be `r` times finer
  than the model grid (the model pixels are resampled; odd `r` keeps Yee nodes
  off pixel boundaries). The global `dt` is set by the DGTD band; for P1 with
  `h = dx` the DG limit is close to the Yee limit at `dx/3`, so `r = 3` costs
  2–20 % more steps, and `r = 2` none. Meanwhile the interface reflection and
  the Yee dispersion drop by `r²`. This turned out to be the decisive setting
  for the hybrid's accuracy (Section 7).
* Sources and receivers take physical coordinates. Points strictly inside the
  band use DG: the source is `M_K^{-1} φ(x_s)` with DeepGPR's amplitude
  convention `J = w δ` and the receiver is polynomial evaluation. A point on an
  edge or vertex is averaged over all containing elements. Other points use
  bilinear Yee weights, with half-weight rows handled on interfaces.

## 7. Verification (commands and results)

```bash
python -m unittest tests.test_hybrid_dgtd tests.test_hybrid_topography   # 22 tests, ~1.5 min CPU
python examples/hybrid/phase1_flat_interface.py         # Phase 1 tables (CUDA)
python examples/hybrid/phase2_benchmark.py              # Phase 2 tables and figures (~45 min)
python examples/hybrid/topography_gifs.py                # same-surface GIFs and mesh figures
python -m unittest discover -s tests -p "test_*.py"     # full suite: 193 OK (4 skipped)
```

**Energy (closed PEC box, central fluxes, material jump inside the band,
non-matching interfaces with DG spacing 1.73 or 2.31 cm vs Yee 2 cm):** relative
variation of the leapfrog energy after the source is switched off,
3–10 × 10⁻¹⁶ for P1/P2 on terrain-following (jittered) and Delaunay meshes,
with topography, and for full DGTD.

**Artificial interface reflection** (vacuum, 1 GHz Ricker, source 0.3 m
below the lower interface, `R = max|u_hyb − u_Yee|` in the reflection window
divided by the incident amplitude corrected for 2D spreading; same `dt` in
both; P1 central, `h = dx`):

| dx (points per λ at 1 GHz) | 0° | 24° | 42° | 53° | 61° |
|---|---|---|---|---|---|
| 2 cm (15) | 1.8e-2 | 1.1e-2 | 5.6e-3 | 2.3e-2 | 4.4e-2 |
| 1 cm (30) | 4.5e-3 | 2.5e-3 | 1.5e-3 | 5.2e-3 | 8.9e-3 |
| 0.5 cm (60) | 1.1e-3 | 6.2e-4 | 3.8e-4 | 1.3e-3 | 2.2e-3 |

The reflection converges at second order (factor 4.0–4.1 per halving).
Refining only the DG mesh (`h = 0.5 dx`), non-matching meshes (`h_y = 0.83 dx`),
P2 or (partial) upwind change it by less than 30 %. It is dominated by the
mismatch of the two discretisations' numerical impedances (both O(k²h²)),
not by the coupling. See `results/phase1_summary.json` for all variants.

**Waveform accuracy** (`eps_r = 2`, 1 GHz, wave crossing both interfaces,
relative L2 vs Yee at 1.25 mm; default flux `alpha = 0.1`):

| dx | hybrid P1 | hybrid P2 (h = 2dx) | Yee at hybrid dt | Yee at own CFL dt | full DGTD P1 |
|---|---|---|---|---|---|
| 2 cm | 0.197 | 0.233 | 0.509 | 0.259 | 0.304 |
| 1 cm | 0.031 | 0.048 | 0.130 | 0.053 | 0.071 |
| 0.5 cm | 0.0083 | 0.0094 | 0.030 | 0.023 | 0.015 |

The hybrid converges at second order (central fluxes give 0.161 / 0.025 /
0.0064). At the small hybrid `dt` Yee is less accurate than at its own CFL
step, the usual space–time dispersion compensation.

**Long runs with CPML** (60 000 steps, 0.49–0.83 µs, sine topography,
`eps_r = 6`, energy sampled every 500 steps): all runs stay finite. Final
energy over peak is 2.7e-10 for `alpha = 0.1` on the Delaunay mesh, 2.1e-6 for
central fluxes on the Delaunay mesh, and 6.3e-10 for `alpha = 0.1` on the
terrain mesh with Silver–Müller side faces. One earlier central-flux
terrain-mesh run showed a transient 8 % energy rise between samples (CPML is
not dissipative by construction) before decaying to 9e-8. Full upwind at
0.8× the central limit is unstable (Section 5).

**Absorbing boundaries.** CPML reflection, measured as the difference between
a 1 m domain and the same domain with that CPML moved 2 m further out
(vacuum, 1 GHz Ricker, 1 cm cells, 20 layers, full DGTD for the DG rows):

| | strength 1.0 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| Yee CPML (DeepGPR) | 2e-7 | – | – | – |
| DG P1 CPML, 20 layers | 1.2e-3 | 7.6e-4 | 5.3e-4 | 3.2e-4 |
| DG P1 CPML, 30 layers | 2.6e-4 | 1.6e-4 | 9.9e-5 | 5.6e-5 |
| DG P2 CPML, 20 layers | 1.8e-4 | 1.1e-4 | 7.1e-5 | 1.8e-4 |

(normal incidence; about 1.6× larger at 31°). The DG CPML reflects far more
than the Yee CPML because the steep polynomial profile is under-resolved by
P1/P2 elements. The default 0.4 keeps the continuous-PML term
`exp(-1.6 · 0.4 · N)` below 3e-6 for N = 20. In the hybrid only the side
columns of the band use DG CPML; full DGTD uses it on all faces. Thicker DG
CPML (`pmlthick=30`) is the cheaper way to reach 1e-4.

### Phase 2: topography (`examples/hybrid/phase2_benchmark.py`)

**Set-up.** 6 × 3.6 m, CPML 0.4 m, 300 MHz Ricker, ground `eps_r = 6`,
`sigma = 3 mS/m` (centre wavelength in the ground 20 cells of 2 cm). The
subsurface model is the 2 cm grid a DeepGPR user would supply; the surface is
the exact curve. Models: A flat ground at 1.2 m, B sine (0.25 m, 2.4 m),
C irregular surface from elevation samples with an `eps_r = 15` and a
`sigma = 0.05 S/m` anomaly, plus a 16.7° slope. Acquisitions: a common-shot
gather (57 receivers) and a common-offset B-scan (26 positions, 0.4 m
offset). Antennas sit about 0.1 m above the local ground, on 2 cm nodes, so
every solver uses identical positions. Every run uses its own frame-aligned
time step (Yee at 0.95 of its CFL limit), with source samples at mid-steps.
Errors are relative L2 against the reference (all traces / median per
trace); gather receivers within 0.3 m of the source are excluded because they
sample the grid-dependent near field.

**Reference.** Native DeepGPR FDTD at `dx/8 = 2.5 mm` with cell-averaged
`eps`/`sigma` at the surface. For Ez, which in 2D TM is tangential to the
surface everywhere, the arithmetic cell average is the second-order effective
medium. It agrees with `dx/16` to 0.02 % (A), 0.01 % (B) and 0.27 % (C, limited
by the anomaly interiors). The staircase reference converges only at first
order: 0.86 % for A, because a surface lying on a node row is shifted by half a
cell.

**Methods.** Yee with the staircase surface (DeepGPR default) at `dx`, `dx/2`,
`dx/4`; Yee with cell-averaged materials at `dx`, `dx/2`; full DGTD P1
(`h = dx`) and P2 (`h = 2dx`); hybrid P1/P2 with the Yee part at `dx/3`
(`fdtd_refinement=3`) and, for comparison, P1 with the Yee part at `dx`. DGTD
bands are 0.7–1.8 m (A, B, slope) and 0.7–1.95 m (C, anomalies in the Yee
region). Delaunay meshes, `alpha = 0.1`, matched CPML profiles.

| method | A flat | B sine | C irregular + anomalies | slope | time 1 shot / 26-shot B-scan | peak GPU (MB) |
|---|---|---|---|---|---|---|
| Yee dx, staircase | 13.7 / 16.0 (13.4) | 0.89 / 1.55 (1.14) | 11.4 / 11.7 (5.6) | 3.0 / 8.4 (10.9) | 0.01 / 0.1 s | 10 / 110 |
| Yee dx/2, staircase | 6.9 / 8.0 (6.7) | 1.0 / 1.1 (1.0) | 2.1 / 2.2 (1.25) | 1.5 / 4.1 (5.4) | 0.04 / 1.5 s | 39 / 438 |
| Yee dx/4, staircase | 3.4 / 4.0 (3.3) | 0.46 / 0.49 (0.46) | 0.51 / 0.39 (0.38) | 0.72 / 2.0 (2.7) | 0.18 / 12 s | 156 / 1735 |
| Yee dx, averaged | 1.5 / 2.7 (1.2) | 0.74 / 1.9 (0.79) | 11.5 / 11.4 (5.3) | 1.0 / 2.0 (1.1) | 0.01 / 0.1 s | 10 / 110 |
| Yee dx/2, averaged | 0.34 / 0.60 (0.27) | 0.20 / 0.46 (0.18) | 1.9 / 1.9 (0.89) | 0.22 / 0.45 (0.23) | 0.04 / 1.5 s | 39 / 438 |
| full DGTD P1 | 0.31 / 0.66 (0.19) | 0.24 / 1.20 (0.27) | 3.0 / 2.8 (1.3) | 0.31 / 0.55 (0.18) | 2.2–2.9 / 72–95 s | 151 / 1976 |
| full DGTD P2 | 0.43 / 0.19 (0.38) | 0.39 / 0.29 (0.59) | 3.2 / 2.6 (1.4) | 1.05 / 0.17 (0.52) | 1.7 / 22–26 s | 65 / 769 |
| hybrid P1, Yee dx | 1.3 / 2.0 (1.0) | 0.82 / 4.4 (0.82) | 8.2 / 8.6 (3.8) | 0.99 / 1.15 (0.97) | 4.6–5.4 / 14–20 s | 42 / 498 |
| **hybrid P1, Yee dx/3** | **0.36 / 0.30 (0.30)** | **0.30 / 1.19 (0.25)** | **1.18 / 0.95 (0.57)** | **0.29 / 0.35 (0.17)** | 5.4–6.3 / 22–29 s | 86–94 / 815–879 |
| hybrid P2, Yee dx/3 | 1.3 / 0.33 (0.40) | 0.54 / 0.57 (0.27) | 1.6 / 1.2 (0.76) | 0.54 / 0.18 (0.39) | 5.2–5.4 / 11–13 s | 63 / 460 |

Values are gather L2 / gather median per trace (B-scan L2), in %. Unknowns per
method: Yee dx 0.22 M, dx/2 0.90 M, dx/4 3.6 M; full DGTD P1 1.55 M, P2 0.77 M;
hybrid P1 with Yee dx/3 1.9 M (0.38 M DG + 1.5 M Yee); reference 14.4 M. Time
steps over 40 ns: 960 (Yee dx), 3600 (Yee dx/4), 2560–3200 (DGTD),
3200–3760 (hybrid), 7200 (reference). Peak GPU memory is the PyTorch
allocation of one gather / the 26-shot B-scan, without the CUDA context. Full
per-case tables: `results/phase2_tables.md`; figures:
`results/phase2_*_{snapshots_field,snapshots_error,bscan}.png`.

**Findings.**

* A body-fitted surface removes the staircase error. At the same 2 cm model
  resolution, the hybrid (Yee part at `dx/3`) is 38× more accurate than
  staircase Yee for the flat model, 10× for C and 64× for the slope B-scan.
  It beats staircase Yee at `dx/4` for A and the slope, and lies between
  staircase `dx/2` and `dx/4` for B and C.
* Two settings decide the hybrid's accuracy. (1) Yee refinement:
  `fdtd_refinement=3` reduces the artificial interface reflection and the Yee
  dispersion by 9 at 2–20 % more time steps (A 1.3→0.36 %, B median
  4.4→1.2 %, C 8.2→1.2 %). (2) Matched CPML profiles in the side columns:
  slope 0.95→0.29 %, A 0.52→0.36 %.
* Model C shows the division of labour the hybrid is meant for. DGTD carries
  the surface, the refined Yee grid carries the high-permittivity anomaly. Full
  DGTD at `h = 2 cm` under-resolves the `eps_r = 15` anomaly (about 5 elements
  per wavelength at the top frequency, 3.0 %), and with the anomalies inside the
  DG band the hybrid shows the same limitation (3.5 %).
* In 2D TM, sub-cell averaging is a strong baseline. Yee with cell-averaged
  `eps`/`sigma` at `dx/2` is as accurate as the hybrid for A, B and the slope,
  at a hundredth of the run time, because Ez is tangential to every surface.
  The hybrid is ahead where the averaging does not help (C, anomalies). The
  body-fitted advantage is expected to grow for field components normal to
  interfaces (3D, TE), where averaging degrades. Phase 4 has to measure this.
* Artificial reflection. The interface reflection measured in Phase 1 (about
  2 % at 15 points per wavelength, ∝ `dx²`) scales with the Yee spacing, so with
  `r = 3` it is about 0.1 % at the ground-side interface of these models (61
  cells per centre wavelength). The bottom interface no longer needs to sit far
  from the antennas: bands of 0.7–1.8 and 0.7–2.6 m gave nearly the same
  errors with `r = 3` (sine median 1.1 vs 1.0 %), whereas with the Yee part at `dx` the band depth changed the sine
  median from 4.4 % to 2.0 %.
* Cost. Native CUDA Yee runs 30–500× faster than the PyTorch DGTD paths.
  Full DGTD P1 is faster per shot than the hybrid (2.5 vs 5.5 s), because the
  hybrid issues more small kernels (the step time is launch-bound: it hardly
  changes between 43k and 86k DG elements). With batched shots the hybrid is
  3× faster than full DGTD and needs 2.3× less memory. Fused kernels (Phase 4)
  are needed before cost comparisons say anything about the method rather
  than the implementation.

**Absorbing boundaries and long runs.** See the CPML table above. Full DGTD
with CPML and topography decays to below 1e-6 of its peak energy in 12 000
steps (`FullDGTDStabilityTest`), and the hybrid long runs of Phase 1 cover the
side-column CPML.

## 8. Discrete adjoint and FWI (Phase 3)

### 8.1 Discrete forward model

All operators of one leapfrog step live in one side-effect-free function,
`functional.step` (`functional.py`). The forward solver (`HybridTM2D.run`), the
CFL estimate and the adjoint all call it. It advances the full coupled state

```
u = (Ez, Hx, Hy, ψ_ex, ψ_ey, ψ_hx, ψ_hy,      Yee fields and Yee CPML
     E, Hx, Hy, q_ex, q_ey, q_hx, q_hy)       DG fields and DG CPML
```

as `u^(n+1) = S(u^n, c(m), w_n)`, where `S` is linear in `u` and `w`:

* Yee H update with CPML (Yee `ch`, fixed);
* DG `E*` from the projected Yee interface rows (`C`);
* DG H update with fluxes and DG CPML;
* `H_b = Cᵀ Hy_DG / dy` and the ghost rows;
* Yee E update `ca Ez + cb (curl H + CPML)`;
* DG E update `ca E + cb (D_x Hy − D_y Hx)` with upwind penalties;
* source injection `−cb w_n δ`.

Receivers are `d_n = R u^(n+1)`. The material coefficients `c(m)` are

* Yee: `ca = (ε/dt − σ/2)/(ε/dt + σ/2)`, `cb = 1/(ε/dt + σ/2)` per Ez node;
* DG: the same per element (this is where the mass matrix `ε_K J_K M` enters);
* the face admittances `Y = sqrt(ε/μ)` of the upwind penalty, on both sides
  of every face.

They come from the model-grid `(εr, σ)` through
`HybridTM2D.material_fields` and `functional.build_coefficients`. The chain is:
model pixels → DG elements and the refined Yee grid → air above the surface
→ ground continued over the surface for DG sampling → CPML replication
(`_ExtendModel`). All of these are PyTorch operations; `μr` is fixed.

### 8.2 Adjoint

For `J(d)` the discrete adjoint recursion is (`adjoint.py`)

```
g_n       = λ^(n+1) + Rᵀ ∂J/∂d_n
λ^n       = (∂S/∂u)ᵀ g_n
∂J/∂c    += (∂S/∂c)ᵀ g_n ,      ∂J/∂w_n = (∂S/∂w)ᵀ g_n ,      λ^nt = 0
```

Each transpose product is a vector–Jacobian product of `S` itself, computed by
reverse-mode differentiation of the step operator once per step. It is
therefore the exact transpose of the implemented discrete step: the interface
mortar `C`/`Cᵀ`, the ghost rows, the DG lift and trace maps, the upwind
penalties, the CPML recursions, conductivity and source scaling. No
continuous adjoint equations are involved. `∂J/∂c` is then carried to
`(εr, σ)` by PyTorch through `build_coefficients` and `material_fields`.

The reverse sweep needs `u^n` for every `n`. The forward pass stores a
checkpoint every `k = ceil(sqrt(nt))` steps, and the backward pass recomputes
one segment at a time, so memory holds about `2 sqrt(nt)` states. The cost of
a gradient is one forward pass, one recomputation and one VJP sweep.

`HybridPropagator(solver, sources, receivers)` wraps this in a
`torch.autograd.Function`: `data = prop(eps_r, sigma, wavelet)`, and
`loss.backward()` returns `eps_r.grad`, `sigma.grad` (and the wavelet
gradient).

### 8.3 Unified solver selection

```python
from DeepGPR.hybrid import make_propagator
prop = make_propagator("hybrid_dgtd", eps0, sig0, dx, src, rec, surface=surface,
                       dg_band=(0.25, 1.05), fdtd_refinement=3, mesh="delaunay")
# or: make_propagator("fdtd", eps0, sig0, dx, src, rec, surface=surface)
data = prop(eps_r, sigma, wavelet)        # (nshot, nt, nrx), differentiable
loss = ((data - d_obs) ** 2).sum(); loss.backward()
```

Both propagators take the ground model on the model grid plus the surface; the
inversion variables, losses (MSE/L1/M-SSIM or any PyTorch loss), optimizers
(Adam, L-BFGS), bounds and multi-shot batching are identical. `"fdtd"` calls
`DeepGPR.compute` (native discrete adjoint) with the surface as a node
staircase; `resample` maps observed data onto each propagator's time axis.

### 8.4 Gradient verification (`examples/hybrid/phase3_gradient_check.py`, `tests/test_hybrid_adjoint.py`)

Small topography model (30 × 40 cells of 2 cm, sine surface, DGTD band with a
non-matching interface, Yee part at `dx/3`, CPML, `sigma > 0`, upwind penalty
`alpha = 0.1`, two shots, 250 steps):

* Dot-product test of the linear source-to-data map
  `<F δw, r> = <δw, Fᵀ r>`: relative difference 5e-14 (float64).
* Checkpoint interval `k = 1, 17, 250`: identical gradients (difference 0).
* Sensitivity above the surface: exactly 0 (air is not a parameter).
* Pure-Yee mode vs DeepGPR's native discrete adjoint (float32, random model,
  two shots): data 4e-7, `∂J/∂εr` 6e-6, `∂J/∂σ` 1e-6 relative difference.
* Finite-difference directional derivatives of `J = ½‖d(m) − d_obs‖²`,
  central differences, relative step `h` (perturbations confined to the region
  named; values are `|g_FD − g_adj| / max(|g_FD|, |g_adj|)`):

| direction | h = 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|
| DG band, εr | 1.3e-1 | 1.5e-3 | 1.5e-5 | 1.5e-7 | 1.9e-9 | 3.6e-9 | 2.3e-8 |
| DG band, σ | 3.2e-5 | 3.2e-7 | 3.2e-9 | 1.8e-10 | 1.1e-9 | 4.2e-9 | 1.9e-7 |
| Yee region, εr | 1.3e-1 | 1.5e-3 | 1.5e-5 | 1.5e-7 | 1.5e-9 | 5.7e-10 | 1.3e-8 |
| Yee region, σ | 3.3e-5 | 3.3e-7 | 3.3e-9 | 1.0e-9 | 4.6e-10 | 1.1e-7 | 4.3e-7 |
| interface rows, εr | 2.3e-2 | 2.3e-4 | 2.3e-6 | 2.3e-8 | 2.2e-10 | 3.7e-10 | 1.7e-9 |
| interface rows, σ | 1.5e-5 | 1.5e-7 | 1.4e-9 | 1.2e-10 | 8.3e-10 | 5.5e-8 | 4.1e-8 |
| everywhere, εr | 1.1e-2 | 4.9e-5 | 4.8e-7 | 4.8e-9 | 1.7e-10 | 2.3e-10 | 1.2e-8 |
| everywhere, σ | 4.7e-5 | 4.7e-7 | 4.8e-9 | 1.2e-9 | 6.2e-9 | 4.0e-8 | 2.4e-7 |

The error falls by 100 per decade of `h` (the `O(h²)` truncation of the
central difference) down to 1e-10–2e-9, where float64 round-off takes over.
That is the signature of an exact gradient. In float32 on the GPU the same sweep
reaches 4e-5–3e-4 at `h = 1e-2…1e-3` before round-off dominates
(`results/phase3_gradient_check.json`).

### 8.5 FWI on undulating ground (`examples/hybrid/fwi_topography.py`)

**Set-up.**
* Model: 4 × 2.4 m on a 2.5 cm model grid; irregular surface from elevation
  samples; ground `εr = 5`, `σ = 2 mS/m`; a known reflector at 2 m depth
  (`εr` 5 → 9, also in the starting model); an `εr = 6.5` disc and a
  `σ = 20 mS/m` disc, radius 0.25 m, at 1.4 m depth.
* Acquisition: 14 sources and 33 receivers about 0.1 m above the ground,
  300 MHz Ricker, 35 ns; offsets below 0.3 m are muted.
* Observed data: an independent solver, native FDTD at `dx/4` with
  cell-averaged surface materials and the discs on that grid (no inverse crime).
* Inversion: L-BFGS (strong Wolfe) on normalised parameters. Stage 1: `εr`
  only, data low-passed below 180 MHz, 15 iterations. Stage 2: `εr` and `σ`,
  full band, 35 iterations. Bounds `εr ≥ 1`, `σ ≥ 0`. The same script, data
  and optimiser run with `solver="hybrid_dgtd"` (DG band 0.25–1.05 m, P1,
  Yee at `dx/3`) and `solver="fdtd"` (native DeepGPR, staircase surface).

**Results.** Three experiments covering the Phase 3 list: (1) flat ground, one
`εr` disc; (2) irregular ground, one `εr` disc; (3–4) irregular ground, `εr` and
`σ` discs, joint inversion; (5) every case inverted with the staircase Yee-FDTD
and with the hybrid, on the same data. (1) and (2) invert `εr` only, with 15 + 15
iterations; (3) runs 15 + 35. GIFs (one frame per iteration):
`results/fwi_<case>_{hybrid_dgtd,fdtd,compare}.gif`.

Model error is `‖m − m_true‖ / ‖m_true − m_0‖` over the ground. "Near-surface"
is the RMS `εr` error 0–0.3 m below the surface (artifacts); "disc" is the mean
`εr` increase recovered inside the `εr` disc (true +1.5).

| case | solver | data misfit | εr error | σ error | near-surface εr RMS | εr in disc | time / iteration | peak GPU |
|---|---|---|---|---|---|---|---|---|
| (1) flat, εr disc | Yee staircase | 2e-4 | 0.83 | – | 0.335 | +0.96 | 0.11 s | 0.7 GB |
| | hybrid | 1e-4 | **0.53** | – | **0.036** | +1.07 | 46 s | 15.5 GB |
| (2) irregular, εr disc | Yee staircase | 2e-4 | 0.70 | – | 0.131 | +0.85 | 0.11 s | 0.7 GB |
| | hybrid | 2e-4 | **0.59** | – | **0.048** | +0.95 | 63 s | 17.5 GB |
| (3–4) irregular, εr + σ discs | Yee staircase | 1.0e-4 | 0.90 | 0.90 | 0.168 | +0.92 | 0.10 s | 0.7 GB |
| | hybrid | 1.3e-4 | **0.79** | 0.91 | **0.073** | +0.99 | 60 s | 17.5 GB |

About 2.2 forward+adjoint evaluations per L-BFGS iteration; one hybrid evaluation
(14 shots, 3004 steps) takes 21–28 s, a staircase-Yee evaluation 0.05 s
(625 steps of 5.6e-11 s).

Findings:

* Interface artifacts. The staircase surface makes the inversion put spurious
  `εr` structure just below the surface: stripes along the steps and patches under
  slopes. These fit the data as well as or better than the hybrid while
  degrading the model. Near-surface error is 2.3–9× lower with the body-fitted
  surface, and the recovered disc is closer to the truth in every case. Even the
  flat case suffers with the staircase: the surface at 0.62 m lies between Yee
  rows, so the staircase shifts it by 7.5 mm.
* Crosstalk (case 3–4). `εr` at the `σ` disc: +0.04 (staircase) / +0.07
  (hybrid); `σ` at the `εr` disc: 0.63 / 0.34 mS/m (true 0).
* `σ` is weakly recovered in this geometry by both solvers: 2.1 of the 18 mS/m
  contrast inside the disc after 35 joint iterations. Reflection data constrain
  `σ` mainly through the amplitude of waves crossing the disc. Better `σ`
  recovery would need a stronger deep reflector, longer offsets or more
  iterations; it is not a solver limitation (both behave alike).
* Cost. In 2D the native staircase Yee iteration is about 500× faster. The
  hybrid's per-step adjoint is launch-bound PyTorch (Phase 4: fused kernels),
  and its checkpoints (`2 sqrt(nt)` states of 14 shots) take the 15–17.5 GB.

**Why these settings.** With a homogeneous ground and no deeper reflector,
FWI from the background only recovers the edges of the discs (a ring), even
without any modelling error. That is a property of reflection data, not of the
solver: an inverse-crime run (FDTD data inverted with FDTD) behaves the same.
The known deep reflector supplies waves that cross the discs. The contrasts
keep the two-way delay below half a period (`εr` 8 would delay the deep
reflection by 2 ns, cycle-skipped at 300 MHz). In the inverse-crime test
Adam fitted the data while the model error grew (1.0 → 1.2), whereas L-BFGS and
normalised steepest descent reduced it. Adam's per-parameter step
normalisation amplifies updates in low-sensitivity regions.

### 8.6 CUDA library (2D): fused kernels, hand-written adjoint, CUDA Graphs

Files: `csrc/dg2d.cu`, `csrc/yee2d.cu` (compiled with the 3D kernels into one
library, `cuda3d.load_library`), wrappers `cuda2d.py`, transpose `vjp2d.py`.
`HybridTM2D(..., cuda_kernels=True)` (default) attaches them on CUDA devices.
Same design as the 3D library (Section 11.4):

* **DG** (`terms2_fwd`, one thread per shot and triangle): the two curl terms
  with their fluxes, written as linear combinations of the traces with per-face-
  node coefficients.
  - H step: `(D_x E, D_y E)` with the E-flux (`Ym`, `Yp`, compact interface
    trace `X`).
  - E step: `(D_x Hy, D_y Hx)` with the H-flux (`Zm`, `Zp`).

  The transpose (`terms2_bwd1/2`) works in two passes:
  1. per element: the volume transpose, the cotangent of the flux, and from it
     the contributions to the own node and to the mirror node (face nodes run
     in opposite directions on the neighbour), plus the cotangents of
     `Ym/Yp/Zm/Zp` and of `X`;
  2. per element: gather through the mirror-slot map, with optional addends.

  The updates (`upd2_fwd/bwd`) include the stretched-coordinate DG CPML; their
  transposes also return the `ca`/`cb` cotangents. P1–P3, float32/float64.
* **Yee** (`yee2_*`, one thread per shot and node): H and E updates with the
  full-grid CPML coefficient arrays. Each transpose is a single kernel: every
  node evaluates the difference cotangents of itself and of its neighbour, so
  no temporaries are needed. It also returns the `ca`/`cb` cotangents. No
  atomics anywhere in the kernels.
* **Step** (`functional.step`): no host-device synchronisation.
  - The ghost-row index is cached.
  - The flux coefficients are built without `repeat_interleave` and kept
    contiguous, since the kernels read them flat.
  - Source injection uses `index_add` instead of `index_put(accumulate=True)`.

  This makes the step capturable as a CUDA Graph (`graphs.py`, used
  automatically for states ≤ 32 MB).
* **Adjoint**: `vjp2d.step_vjp` applies the transposes in reverse order. It
  uses `u^n` and `u^{n+1}` and recomputes only the compact interface trace and
  the E-step terms (for the `cb` cotangent). `HybridPropagator` uses it
  whenever Yee and DG both run on the kernels (`DEEPGPR_HYBRID_AUTOGRAD_VJP=1`
  selects autograd).

Verification (`tests/test_hybrid_adjoint.py`, `tests/test_hybrid_dgtd.py`, float64).
Configurations: P1/P2/P3, terrain-following and Delaunay meshes, PEC/ABC
sides, upwind-penalised/central flux, r = 1/3, pure DG and pure Yee.
* fused step against the PyTorch step: forward ≤ 3.6e-16, VJP including every
  coefficient cotangent ≤ 5.6e-16;
* hand-written transpose against autograd: ≤ 3.5e-16 for every state,
  coefficient and source cotangent;
* the whole reverse sweep against the autograd sweep: ≤ 1e-12;
* the CPU adjoint checks also run on CUDA through the hand-written path:
  - source dot product;
  - independence of the checkpoint interval;
  - zero gradient above the surface;
  - finite differences to < 1e-7;
* CUDA Graph replay against the eager step: ≤ 1e-13. As in 3D the coupled
  step is reproducible only to round-off, because of atomics in the coupling's
  `index_add`;
* energy conservation of the coupled scheme on the kernels (central flux, no
  CPML, 1500 steps): < 1e-12.

**Timings** (`examples/hybrid/cuda2d_benchmark.py`). Model and acquisition of
the FWI example (8.5): 13 shots batched, Delaunay P1 band, Yee at dx/3,
3004 steps, float32, RTX 4090. "Gradient" means the whole gradient of
`J = ½‖d‖²`.

| path | forward | gradient | gradient memory |
|---|---|---|---|
| PyTorch step, autograd VJP | 4.12 s (1.37 ms/step) | 27.1 s (9.01 ms/step) | 8.7 GB |
| fused kernels, autograd VJP of the step | 1.47 s (0.49 ms/step) | 7.45 s (2.48 ms/step) | 8.6 GB |
| fused kernels, hand-written transpose (default) | **1.46 s (0.49 ms/step)** | **5.89 s (1.96 ms/step)** | 8.5 GB |
| native DeepGPR, staircase on dx (625 steps) | 0.02 s | 0.04 s | 0.7 GB |

* **Against the PyTorch step**: the forward is 2.8× faster, the gradient 4.6×.
  The three hybrid paths give the same gradients to 1.4e-6 (float32
  round-off).
* **Single-shot forward** (state 5.9 MB, so graphs are on): 0.103 ms per step
  with CUDA Graph replay against 0.365 ms eager.
* **Against native DeepGPR**: the native code is still about 70× faster for a
  forward run and 150× for a gradient. It runs Yee only, on the coarse
  staircase grid, with a 4.8× larger `dt`. The comparison at equal accuracy is
  in Section 7 (Phase 2).

### 8.7 Rugged-topography study (2D): accuracy, cost, eps_r-only FWI (`examples/hybrid/rugged2d_study.py`)

**Model** (2.5 cm model grid, 1.6 m × 3.2 m):
* surface: datum 0.55 m with a 0.24 m ridge, a 0.17 m V-shaped gully and 2 cm
  roughness: 0.43 m of relief, slopes up to 61°, steeper than 30° on half of
  the profile;
* ground: `εr` 9 and 5 mS/m, with a known reflector (`εr` 16 below 1.3 m) and
  three near-surface `εr` anomalies (14, 5, 13);
* acquisition: 14 sources and 29 receivers 5 cm above the local surface, all
  on 2.5 cm nodes; 400 MHz, 24 ns.

**Reference**: native FDTD with cell-averaged surface materials on 2.5/9 cm.
Its difference to 2.5/15 cm is 0.18 %. With an even factor (2.5/8 cm) its
boundary nodes coincide with those of even FDTD grids and flatter them: dx/4
then scored better than dx/5.

**Forward**: relative L2 of all gathers (offsets ≥ 0.2 m), 14 shots batched, RTX 4090.

| solver | error | time | peak memory |
|---|---|---|---|
| FDTD staircase, 2.5 cm (model grid) | 21.8 % | 0.02 s | 0.01 GB |
| FDTD staircase, 0.83 cm | 2.73 % | 0.09 s | 0.10 GB |
| FDTD cell-averaged, 0.83 cm | 2.25 % | 0.09 s | 0.10 GB |
| FDTD staircase, 0.5 cm | 1.13 % | 1.14 s | 0.29 GB |
| hybrid: DG P1 h = 1.25 cm + Yee 0.83 cm | 2.12 % | 2.21 s | 0.10 GB |

The DG band needs `h = 1.25 cm` here. At 2.5 cm it is the accuracy
bottleneck (6.3 %) for two reasons:
* 10 elements per ground wavelength;
* elements that straddle the 2.5 cm pixels of the anomalies, since each
  element takes the material at its centroid.

P2 elements do not remove the pixel effect.

**`εr`-only FWI**: observed data from the reference; σ known; L-BFGS; 20
iterations low-passed to 250 MHz, then 40 full band. Model error
`‖m − m_true‖ / ‖m_true − m0‖` over the ground:

| solver | best | after 60 iterations | per misfit + gradient | peak memory |
|---|---|---|---|---|
| FDTD staircase, 2.5 cm | 0.590 (iteration 20) | 0.752 | 0.03 s | 0.31 GB |
| FDTD staircase, 0.83 cm | 0.348 | 0.348 | 0.21 s | 1.58 GB |
| hybrid (as above) | 0.357 | 0.357 | 10.4 s | 12.6 GB |

Reading, stated plainly:

* **Against FDTD on the model grid** the gap is large.
  - Forward: 21.8 % against 2.1 %.
  - FWI: the 2.5 cm staircase stops improving after the low-pass stage and
    then fits its modelling error (model error 0.59 → 0.75, striped
    anomalies), while the hybrid keeps improving.
* **Against FDTD on the hybrid's Yee spacing** there is no meaningful gap.
  - Forward: 2.1 % against 2.7 % (2.25 % cell-averaged).
  - FWI: the same model error (0.357 against 0.348).
  - Cost: the refined staircase is 50× cheaper per gradient and needs 8× less
    memory. FDTD at 0.5 cm is more accurate than the hybrid at half its cost.
* **Why**: in 2D TM, Ez is tangential to the surface everywhere. The staircase
  error of a tangential field is mild and falls quickly with refinement, so at
  0.83 cm the remaining error is mostly the Yee dispersion that the hybrid's
  Yee part shares. A larger benefit is expected where the electric field has a
  component normal to the surface (TE, 3D), or where a surface lies on a grid
  plane (half-cell bias of the staircase; 3D benchmark A3 in Section 11.5:
  0.43 % against 5.5 % at equal Yee spacing). Neither is covered by this 2D TM
  study.

Outputs (`results/rugged2d/`):
* `forward.gif`: three columns (FDTD 2.5 cm, FDTD 0.83 cm, hybrid), each with
  the full wavefield, a zoom with the Yee cells or triangles, and the error;
* `mesh.png`, `traces.png`, `forward_summary.png`;
* `fwi.gif` (60 iterations) and `24it_fwi.gif` (24 iterations);
* `fwi_summary.png`, `forward.json`, `fwi.json`.

### 8.8 Staircase DGTD-FDTD interface following the surface (2D prototype, `staircase.py`)

`StaircaseHybridTM2D(..., layer=(above, below))` replaces the planar band by
a DGTD layer around the surface. Per model column (2.5 cm, `r × r` Yee cells)
the layer covers the model rows from at least `above` over the surface to at
least `below` under it. The DGTD-FDTD interface `Γ` is therefore a staircase
of horizontal and vertical Yee grid lines that follows the surface.

* **Why a staircase and not a smooth curve.** `Γ` is artificial, with the same
  medium on both sides, so staircasing it costs no geometric accuracy. The
  ground surface stays inside the layer as a chain of triangle edges. A smooth
  artificial interface would cut Yee cells and bring back the cut-cell
  problem.
* **Yee side: summation by parts on an arbitrary union of cells.**
  - The Ez update is the circulation of H around the part of the node's dual
    cell that lies in the Yee region `R`.
  - Node weights: 1 inside, ½ on straight pieces, ¼ at convex and ¾ at
    concave corners of `R`. H-edge weights: (adjacent cells in `R`)/2. Every
    dual side has a fraction 0, ½ or 1 inside `R`.
  - The missing circulation is the tangential DG field (Hy on horizontal
    pieces, Hx on vertical ones), through a hat-function mortar per straight
    piece. End nodes get half hats, and a corner node collects both pieces.
  - Nodes on straight pieces and at convex corners keep the regular (fused)
    Yee update. The side that faces the layer is an inactive edge used by
    that node only, and receives a ghost value, as the planar ghost rows do.
    Concave corners are updated explicitly after the Yee update.
  - The layer is flat in the side-CPML columns and two columns beyond, so no
    corner lies in a CPML.
* **DG side**: unchanged. `E* = I(E_b)` is continuous piecewise linear along
  `Γ`, and `H* = H⁻`. All pieces are combined into one sparse operator per
  direction, so the coupling costs a few kernels per step: with one operator
  per piece the step was 25× slower.
* **Mesh**: Delaunay of the resampled staircase boundary, the surface and
  graded interior points. It is checked to tile the layer exactly and not to
  cross the surface.

**Verification** (`tests/test_hybrid_staircase.py`):
* the coupled energy (central fluxes, no absorbers, sine surface with 146
  pieces and 72 concave corners, 1500 steps) is conserved to 5e-16, for r = 1
  and 3, on the CPU and through the fused CUDA kernels;
* discrete adjoint (autograd VJP of the step; the hand-written transpose of
  8.6 is not yet extended to the staircase): source dot product < 1e-12,
  finite differences for `εr` and `σ` < 1e-7;
* weights and corner types on a test step; the layer mesh tiles the layer.

**Rugged model of 8.7**, layer 10 cm above and 7.5 cm below the surface:
* 216 interface pieces and 107 concave corners;
* 11 111 triangles against 44 588 for the planar band;
* the three anomalies now lie in the Yee region.

| solver | gather error | time (14 shots) | peak memory |
|---|---|---|---|
| FDTD staircase, 0.83 cm | 2.73 % | 0.10 s | 0.10 GB |
| hybrid, planar band 0.15–0.95 m | 2.12 % | 2.13 s | 0.10 GB |
| **hybrid, staircase interface** | **1.95 %** | **1.55 s** | 0.07 GB |
| FDTD staircase, 0.5 cm | 1.13 % | 1.05 s | 0.29 GB |

Thinner (7.5 / 5 cm) or thicker (10 / 15 cm) layers give 2.11 % and 1.92 %.
The DG time step is unchanged (smallest element), so the gain is the 4× smaller
DG part per step; the remaining error is again mostly the Yee dispersion at
0.83 cm.

`εr`-only FWI with the same schedule as 8.7 (60 iterations):

| solver | model error after 60 iterations | per misfit + gradient | peak memory |
|---|---|---|---|
| FDTD staircase, 2.5 cm | 0.752 (best 0.590) | 0.03 s | 0.31 GB |
| FDTD staircase, 0.83 cm | 0.348 | 0.21 s | 1.58 GB |
| hybrid, planar band | 0.357 | 10.4 s | 12.6 GB |
| hybrid, staircase interface | **0.342** | 7.8 s (autograd VJP) / 6.0 s (hand-written) | 9.1 GB |

The staircase FWI ran with the autograd VJP of the step. The hand-written
transpose (verified afterwards, `HandWrittenAdjointTest`) brings one 14-shot
gradient from 7.7 s to 6.0 s, against 10.4 s for the planar band.

Reading:
* **Against the planar band**: the staircase interface gives the same or
  slightly better accuracy at 4× fewer DG elements and 1.7× less time per
  gradient. The anomalies now lie in the Yee region, so the DG pixel-sampling
  error of 8.7 disappears.
* **Against the refined staircase FDTD** the conclusion of 8.7 stands.
  Accuracy is equal within noise (0.342 against 0.348 model error, 1.95 %
  against 2.73 % gather error), and the FDTD is still 28× cheaper per
  gradient. In 2D TM the remaining error is set by the Yee dispersion that
  all three refined solvers share.

Figures: `results/rugged2d/interfaces_staircase.png` (where the interface is,
zooms on the ridge and the gully) and the fourth column of
`results/rugged2d/forward.gif`.

## 9. Usage

```python
import torch, DeepGPR
from DeepGPR.hybrid import HybridTM2D, Surface, FieldSampler

eps = torch.full((180, 300), 6.0)          # ground model, dx = 2 cm, axis 0 = depth
sig = torch.full((180, 300), 0.003)
surface = Surface.sine(1.2, 0.25, 2.4)     # depth(y) [m]
sim = HybridTM2D(eps, sig, 0.02, surface=surface, dg_band=(0.7, 1.8),
                 pmlthick=20, mesh="delaunay", fdtd_refinement=3,
                 device="cuda", dtype=torch.float32)
src = [[surface.point_at_height(3.0, 0.1)]]                 # (nshot, nsrc, 2) [m]
rec = [[surface.point_at_height(y, 0.1) for y in (2.0, 4.0)]]
nt = int(40e-9 / sim.dt)
w = DeepGPR.wavelet.ricker(3e8, nt, sim.dt, 1 / 3e8 * 1.2)
data = sim.run(w, src, rec)["receivers"]                    # (nshot, nt, nrx)
```

`dg_band=None` gives the pure Yee scheme (same as `DeepGPR.compute`,
order 2) and `dg_band=(None, None)` full DGTD. On CUDA the step runs on the
fused kernels of Section 8.6 (`cuda_kernels=False` selects the PyTorch step;
same results to round-off). The environment switches of Section 11.8
(`DEEPGPR_HYBRID_DISABLE_GRAPHS`, `DEEPGPR_HYBRID_AUTOGRAD_VJP`,
`DEEPGPR_HYBRID_DISABLE_CUDA`) apply in 2D too.

## 10. Open issues

* Adjoint memory: the checkpoints hold `2 sqrt(nt)` full states (8.5 GB for the
  13 shots of the FWI example in float32). Storing the Yee CPML auxiliaries
  only on the CPML strips (as in 3D) would cut it. The adjoint cost itself is
  now 4× a forward step (fused kernels and hand-written transpose, 8.6).
* The FWI examples recover `εr` well and `σ` weakly (Section 8.5); the
  optimiser settings are tuned for the examples, not general defaults.
* Artificial interface reflection: about 2 % at 15 Yee cells per wavelength,
  falling as `dx²`. It comes from the second-order SBP closure of the Yee
  interface row. `fdtd_refinement=3` brings it to about 0.1 % in the Phase 2
  models; a higher-order staggered SBP closure, or an interface on an Hy row,
  has not been tried.
* Pure central fluxes need structured-like meshes; the default `alpha = 0.1`
  is recommended for Delaunay meshes and point sources. Full upwind with
  leapfrog needs `dt ≲ 0.3` of the central limit.
* The DG CPML reflects 1e-4–5e-4 (P1, 20–30 layers, strength 0.4), against
  about 1e-7 for the Yee CPML.
* DGTD elements take the material of the model pixel containing their
  centroid: anomalies inside the band keep the 2 cm pixel shape, and
  under-resolved high-permittivity bodies should stay in the (refined) Yee
  region.
* The terrain-following mesh shrinks `dt` under steep relief (thin layers); the
  Delaunay mesh keeps the element size uniform.
* Cost: with the fused kernels (8.6) the FWI-size model takes 0.49 ms per
  forward step and 1.96 ms per gradient step, against 1.37 / 9.01 ms with the
  PyTorch step. That is still far slower than DeepGPR's native Yee, which runs
  on the coarse grid with a larger `dt`. Local time stepping is outside this
  plan's scope.
* In 2D TM, cell-averaged Yee at `dx/2` is a cheap and accurate alternative
  for smooth surfaces without near-surface heterogeneity; the hybrid's case
  rests on exact geometry, near-surface heterogeneity and on 3D/TE behaviour,
  which Phase 4 has to measure.

## 11. Phase 4: 3D (tetrahedral DGTD band + Yee-FDTD, CUDA)

Modules: `reference3d.py`, `mesh3d.py`, `dgtd3d.py`, `coupling3d.py`,
`functional3d.py`, `solver3d.py` (`HybridMaxwell3D`, `FieldSampler3D`),
`cuda3d.py` + `csrc/dgtd3d.cu`, `csrc/yee3d.cu` (fused kernels), `vjp3d.py`
(hand-written step transpose), `graphs.py` (CUDA Graph replay). Examples:
`examples/hybrid/models3d.py`, `phase4_benchmark.py`, `phase4_gradient_check.py`,
`phase4_cuda_benchmark.py`, `topography3d_figures.py`, `fwi3d_topography.py`.
Tests: `tests/test_hybrid_3d.py`.

### 11.1 Discretisation

* **Layout.** DeepGPR's 3D layout: `x` (axis 0) depth, `y`/`z` horizontal,
  extended grid with a one-cell halo. Yee positions `Ex(i+½,j,k)`,
  `Ey(i,j+½,k)`, `Ez(i,j,k+½)`, `Hx(i,j+½,k+½)`, `Hy(i+½,j,k+½)`,
  `Hz(i+½,j+½,k)`.
* **Yee part.** `functional3d.step` reproduces `update_h_cpu`, `cpml_h_cpu`,
  `update_e_cpu`, `cpml_e_cpu` and `inject_sources_cpu` on full-grid arrays,
  with per-component activity masks and the CPML as stretched curl terms
  (`d → d + (RA−1)d + RB ψ`). Pure-Yee mode against `DeepGPR.compute` (random
  `εr`/`σ`, CPML on six faces, two shots): receivers agree to 3.0e-7 / 3.3e-7 /
  2.9e-7 for the `Ez`/`Ex`/`Ey` source-receiver components (float32 round-off).
* **DG part.** Nodal P1/P2 tetrahedra (equidistant nodes, which coincide with
  the warp-and-blend nodes for N ≤ 2), orthonormal simplex basis, `Dr, Ds, Dt`,
  `LIFT`, affine geometric factors, normals and face Jacobians as in Hesthaven
  & Warburton ch. 10 (doi:10.1007/978-0-387-72067-8). Checked: derivatives exact
  for degree-N polynomials (≤ 4e-14), `∫ ∂_x f = ∮ f n_x` through `LIFT` to
  1e-17, face areas `= 2 sJ`, outward normals. The strong-form DG derivative
  `D_a f = ∂_a f + LIFT(Fscale n_a (f* − f⁻))` is applied per component, so
  `curl_D H = curl H + LIFT(n × (H* − H⁻))`, with
  `H* − H⁻ = ½[H] + α(Z⁺/(Z⁻+Z⁺) − ½)[H] − α/(Z⁻+Z⁺) n×[E]` and
  `E* − E⁻ = ½[E] + α(Y⁺/(Y⁻+Y⁺) − ½)[E] + α/(Y⁻+Y⁺) n×[H]` (α = 0.1 by
  default; PEC `E⁺ = −E⁻`; Silver-Müller faces full upwind against a zero
  exterior). In 2D these reduce to the formulas of Section 3.
* **Mesh.** Two body-fitted tetrahedral meshes of the band, both with vertex
  columns on the horizontal `(y, z)` grid and the surface as a vertex sheet (so
  the surface is a set of tetrahedron faces), each hexahedron of the index
  lattice split into six tetrahedra along its diagonal (Kuhn subdivision,
  conforming between neighbours):
  * `mesh="adaptive"` (default, `adaptive_column_tet_mesh`): every column has
    its own number of layers, `n = clip(round(gap/h), 1, n_max)` uniform layers
    between each band plane and the surface (air and ground separately). The
    lattice sheets a column does not need collapse onto its surface vertex; the
    Kuhn tetrahedra with a collapsed vertical edge have zero volume and are
    removed. A Kuhn tetrahedron's volume is `h_y h_z / 6` times the height
    difference along its single vertical step, so column-wise monotone sheets
    never invert an element, and collapse only identifies consecutive vertices
    of a column, so the mesh stays conforming. Layers are 2/3 h to 2 h thick
    (thinner only where the air or ground gap itself is thinner), instead of
    `gap_min / n` everywhere. This is the "mesh without thin layers" variant:
    a fully unstructured Delaunay mesher was prototyped but discarded (slivers,
    and boundary recovery failed once boundary points were moved; no new
    dependencies were allowed).
  * `mesh="terrain"` (`terrain_following_tet_mesh`): the same number of layers
    in every column, stretched between the band planes and the surface,
    optional interior jitter. Relief makes layers thin and shrinks `dt`.

  Checked: conformity (no face shared by more than two tetrahedra; faces of a
  single tetrahedron lie on the band boundary only), total volume, positive
  volumes, point location (a CSR column
  locator, `locate_points_columns`, agrees with brute force). B3 (±6 cm relief,
  0.24 m band, h = 2 cm, physical part of the band): adaptive 251k tetrahedra,
  shortest edge 1.7 cm, smallest inradius 3.1 mm; terrain-following 376k,
  0.50 cm, 1.9 mm (C3: 251k / 1.7 cm / 3.0 mm against 376k / 0.50 cm /
  1.9 mm). Stable `dt` for B3: 7.8e-12 s adaptive against 3.5e-12 s
  terrain-following (flat ground: 1.0e-11 s). Materials: air elements above the
  surface, otherwise the model pixel containing the centroid (ground continued
  over the surface), as in 2D.

### 11.2 Coupling on horizontal planes, interface penalty

* The interface is an E plane `i_T`/`i_B` of the Yee grid. It carries the
  tangential `Ey` lattice `((j+½)dy, k dz)` and `Ez` lattice
  `(j dy, (k+½)dz)`; `Ey`, `Ez` and the normal `Hx` on the plane have weight ½.
  The x-derivatives of `Hy`/`Hz` at the plane use ghost rows
  `2 H_b − H_inner` (rows `i_T` and `i_B − 1`), exactly the 2D SBP closure.
* Mortar per tangential component: `C_im = ∫ L_i φ_m dA` with the bilinear
  hat functions `φ_m` of the staggered lattice. Every triangle is clipped
  against every lattice cell (vectorised Sutherland–Hodgman), the convex pieces
  fan-triangulated and integrated with a collapsed Gauss rule exact for
  `deg(L_i φ_m) ≤ 4`. DG side: `E*_c = (sJ M_f)⁻¹ C E_c`, `H* = H⁻`; Yee
  side: `H_b = Cᵀ H_t,DG / (dy dz)` (`Hz` for `Ey`, `Hy` for `Ez`). The
  projection reproduces linear fields to 1e-14 (P1, P2, matching and
  non-matching lattices).
* **Energy.** With central fluxes, no CPML and no penalty, the coupled 3D
  scheme conserves the discrete leapfrog energy to 2e-15 over 400 steps of
  random initial data (sine topography, jittered mesh; P1, P2 and
  `fdtd_refinement=3`; `Energy3DTest`).
* **Interface penalty (new in 3D).** In the first benchmark runs the surface
  receivers showed a slow late-time oscillation. Snapshots located it: a
  grid-scale, sign-alternating `Ey`/`Ez` mode trapped on the top interface
  plane (amplitude 50 at 11 ns, decaying 6× per Yee row into the air), which no
  dissipation reaches because the α-penalty acts only on interior DG faces.
  `interface_penalty` adds a dissipative (SAT-type) penalty that relaxes the DG
  trace and the Yee plane towards each other: DG side
  `H* − H⁻ = −γ n × (P E_FD − E⁻)` with `γ = α_if Y⁻/2` (it enters the CUDA
  kernel through `b3` and the external input `X`); Yee side
  `rhs_c += (2/dx)[Cᵀ(γ E⁻_c)/dA − Cᵀ(γ)/dA · E_c]`. Because the lumped norm of
  the hat functions bounds the consistent one, the two terms together
  contribute `−‖√γ (I E_FD − E⁻)‖² ≤ 0` to the energy rate; the discrete
  energy decreases monotonically (`test_interface_penalty_is_dissipative`).
  The penalty is explicit: measured stability limits are `α_if` 1–3 for
  `fdtd_refinement=1` and 0.3–1 for 3. `α_if = 0.1` (default) already removes
  the mode (flat case, surface-line error 6.4 % → 1.5 %; 0.3 and 1.0 give
  1.46 / 1.44 %).

### 11.3 Side absorbers: why a UPML

The 2D design (CPML stretching of the DG derivatives in the side columns,
matched to the Yee CPML) is **unstable in 3D**:

* Long runs with random initial data grow exponentially (×10³ per 500 steps)
  with central, α = 0.1 and full-upwind fluxes alike; CPML on the x faces only
  (away from the band) is stable. The same happens in a DG-only box, so the
  coupling is not the cause.
* Spectrum of the semi-discrete DG + CPML operator (216 tetrahedra, y-PML):
  max Re λ = 2.5e9 s⁻¹ at the solver's strength (5.8e8 even at 1/24 of it),
  against max |Im λ| = 1.9e11 s⁻¹. A complex frequency shift does not help, nor
  does a multi-axial PML (M-PML; Meza-Fajardo & Papageorgiou, BSSA 98(4) 2008,
  doi:10.1785/0120070223) for any ratio p < 1; only p = 1, an isotropic stretch,
  i.e. a matched lossy layer, is stable. This is consistent with the stability
  analysis of PMLs (Bécache, Fauqueux & Joly, J. Comput. Phys. 188(2) 2003,
  doi:10.1016/S0021-9991(03)00184-0): a PML amplifies modes whose group and
  phase velocities have opposite signs along its axis, and DG on tetrahedra
  has such spurious branches; stretching the lifted flux terms component-wise
  also breaks the upwind dissipation.
* The matched lossy layer is stable but absorbs poorly (pure Yee, 10 cells:
  late-window error 0.78 against 1.2e-3 for the CPML and 10.8 for PEC sides),
  so it was rejected.
* **UPML in auxiliary-equation form** (Gedney, IEEE Trans. Antennas Propag.
  44(12) 1996, doi:10.1109/8.546249) keeps the DG curl, with its fluxes,
  unchanged and adds local ODEs per component:
  `∂D_c/∂t + s_{c+1} D_c = (curl H)_c`,
  `∂(εE_c)/∂t + s_{c+2} εE_c + σE_c = ∂D_c/∂t + s_c D_c` (same for `B`/`H`),
  Crank–Nicolson inside the leapfrog, `s_a = σ_a/ε0` from DeepGPR's quartic
  profile times `dg_pml_strength`. Its semi-discrete spectrum is stable
  (max Re λ ≈ 2e3 s⁻¹ y-PML, 2e6 s⁻¹ in a corner, i.e. at most e^0.2 over
  100 ns, against |Im λ| = 1.9e11 s⁻¹). In the hybrid the Yee **CPML next to the
  band** also destabilised the coupled scheme (DG UPML + Yee side CPML grows;
  DG UPML without Yee side PML is stable). The hybrid therefore uses a UPML in
  the Yee side columns too (compact index sets per component, staggered
  positions), and DeepGPR's CPML on the x faces. Pure-Yee mode is unchanged
  (DeepGPR's CPML everywhere). With this, all tested configurations are stable
  (6000 steps of random data, PML on all faces, central / α = 0.1 / upwind;
  `test_side_upml_long_run_stable`).

### 11.4 CUDA library (forward and adjoint)

`csrc/dgtd3d.cu` and `csrc/yee3d.cu` are compiled together on first use with
`nvcc` for the current GPU, cached in `~/.cache/deepgpr/hybrid` (keyed by the
source hash), loaded with `ctypes` and launched on PyTorch's current stream.
Without `nvcc` or a GPU the PyTorch implementation of the same step is used.
Every kernel has an exact transpose; no kernel uses atomics, so the kernels
themselves are deterministic.

* **DG flux and curl terms** (`terms_fwd`): one thread per shot and element.
  It gathers the traces of `F` and of the penalty partner `G`, evaluates
  `dF = b0[F] + b1 n×[G] + b2 F⁻ + b3 n×G⁻ + X` on the four faces with per-face
  coefficients (affine elements), lifts the flux and adds the volume
  derivatives of the six curl terms (P1/P2, float32/float64, int32 maps). The
  external input `X` (projected Yee trace, penalty term) is compact on the
  interface face slots, addressed through a slot map; its cotangent is written
  in the same compact layout. No dense `K × 4 Nfp` array is formed.
* **Its transpose** (`terms_bwd1`, `terms_bwd2`): pass 1 forms, per element,
  - the volume transpose;
  - the lifted face cotangent `h = fscale Σ_j n_{a_j} e_{f_j} (LIFTᵀ g_j)` (three
    values per face node, stored slot-major for coalescing);
  - the cotangents of the per-face coefficients `b0..b3`, which carry `εr`
    through the impedances.

  Pass 2 (one thread per element) rebuilds the cotangents of `F⁻`, `F⁺`, `G⁻`
  and `G⁺` from `h`, using the element's own face coefficients and those of the
  mirror face (mirror-slot map), and gathers them onto the nodes. Storing only
  `h` instead of the four face cotangents cuts the temporaries 4×. Pass 2 can
  add the cotangents of the other uses of `F` and `G` (optional addends) in the
  same pass, which saves the separate full-field additions in the reverse sweep.
* **DG update** (`dgupd_fwd`/`dgupd_bwd`): `curl = T_even − T_odd`, the plain
  update (`H − ch curl`, `ca E + cb curl`) or the UPML recursion in the PML
  elements, with the cotangents of `ca`, `cb` and the material UPML factors.
  The forward uses one thread per node (coalesced). The transpose uses one
  thread per element, which sums the `ca`/`cb` cotangents of the element
  without atomics.
* **Yee H and E updates** (`yee_fwd`, `yee_bwd1`/`yee_bwd2`): one thread per
  node and shot covers all three components:
  - the six differences with the CPML stretching on the slab planes;
  - the auxiliary update (ψ stored on the slabs only);
  - the activity masks and the side UPML (compact index sets);
  - the interface-row penalty.

  The transpose is in two passes: per-node cotangents with the difference
  cotangents in a temporary, then the transposed differences gathered onto
  the differentiated field. The coefficient cotangents are those of `ca`, `cb`
  and the UPML factors. The six ψ pointers and the profile tables are passed
  by value in a struct, so a launch needs no host-to-device copy.
* **Coupling, penalty, sources and receivers** stay small PyTorch index
  operations (`index_select`/`index_add` of the mortar matrices). The step
  issues no host-device synchronisation: cached index tensors, no
  `repeat_interleave`, and `index_add` instead of `index_put(accumulate=True)`.
  Because `index_add` accumulates with atomics, the coupled step as a whole is
  reproducible only to round-off: run-to-run differences are 4e-16 in float64
  and 2e-7 in float32.
* **CUDA Graph** (`graphs.py`): `HybridMaxwell3D.run` and the adjoint's
  forward and recompute passes can capture one step, including the receiver
  sampling and the copy into static state buffers, as a CUDA Graph and replay
  it. This removes the host overhead of about a hundred launches per step, but
  the replay copies the state once per step.
  - Test model (state < 1 MB): 0.90 ms eager against 0.24 ms replayed per step.
  - B3 at r = 1 (state 75 MB): the step is GPU-bound and replay is slower,
    1.87 ms against 1.75 ms.

  Graphs are therefore used automatically only for states up to
  `DEEPGPR_HYBRID_GRAPH_MAX_MB` (default 32 MB); `DEEPGPR_HYBRID_DISABLE_GRAPHS=1`
  turns them off. Graph and eager agree to round-off
  (`test_cuda_graph_replay_matches_eager_step`).
* **Hand-written step transpose** (`vjp3d.step_vjp`, Section 11.6): the reverse
  sweep calls the transpose kernels directly instead of differentiating the
  step with autograd.

Verification:
* the fused forward step against the PyTorch step: ≤ 2.4e-14 (float64; P1, P2,
  UPML or ABC sides, r = 1 and 3);
* the autograd VJP through the fused kernels against the PyTorch step's
  autograd VJP: ≤ 1.4e-11. The largest difference is in the cotangent of the
  DG `cb`, a sum over many terms that cancel;
* the hand-written transpose against autograd: ≤ 2e-15 for every state,
  coefficient and source cotangent.

**Timings** (`examples/hybrid/phase4_cuda_benchmark.py`). B3 model, P1,
adaptive mesh, one shot, float32, RTX 4090:
* r = 1: Yee 61 × 81 × 81 including the absorbers; 460 800 tetrahedra
  (11.1M DG unknowns); state 75 MB;
* r = 3: Yee 181 × 241 × 241; state 436 MB;
* hybrid `dt` = 7.8e-12 s (DG-limited) for both.

"Gradient" means the whole gradient of `J = ½‖d‖²` (forward with checkpoints,
recompute and reverse sweep), per time step of the model.

| path | r = 1 forward | r = 1 gradient | r = 3 forward | r = 3 gradient |
|---|---|---|---|---|
| PyTorch step (`cuda_kernels=False`), autograd VJP | 17.3 ms | 135.5 ms | – | out of memory |
| fused kernels, autograd VJP of the step | 1.66 ms | 9.57 ms | 4.23 ms | 27.3 ms |
| fused kernels, hand-written transpose (default) | **1.67 ms** | **7.17 ms** | **4.22 ms** | **18.6 ms** |
| native DeepGPR, staircase on dx/r (its own `dt`) | 0.07 ms | 0.12 ms | 2.33 ms | 4.79 ms |

Run lengths and memory:
* r = 1: the 16 ns run, 2056 hybrid steps and 439 native steps.
* r = 3: the hybrid over 1.5 ns (194 steps). The full 16 ns r = 3 gradient
  keeps about 2√nt states of 436 MB, roughly 40 GB, which is more than the
  GPU's 24 GB. The native r = 3 run is the 16 ns run (1313 steps, history
  sub-sampled).
* The three hybrid paths give the same gradients to 2e-6 (float32 round-off).
* Peak memory of the r = 1 gradient: 7.5 GB hybrid, 1.9 GB native.

What the numbers mean:

* **Fused kernels against the PyTorch step** (r = 1): 10× faster forward,
  19× faster gradient.
* **Hand-written transpose against autograd of the step**: the gradient is
  1.33× faster at r = 1 and 1.47× at r = 3.
* **Against the code before this work**: B3 forward runs now take 3.4 s
  (r = 1) and 8.7 s (r = 3), against 32 s and 100 s with the earlier code
  (terrain-following mesh, PyTorch Yee part). About 2.2× of that comes from
  the adaptive mesh (2056 instead of 4557 steps), the rest per step.
* **Where the time goes** (r = 1, GPU time per step).
  - Forward 1.62 ms: DG flux/curl 0.77, DG updates 0.39, coupling/penalty
    glue 0.37, Yee kernels 0.09.
  - Hand-written transpose 3.85 ms: DG flux transpose 1.59, glue 1.10, DG
    update transposes 0.57, flux recompute 0.41, Yee 0.19.

  The forward step is GPU-bound: wall time equals GPU time.
* **Against native DeepGPR at comparable accuracy** (B3: hybrid r = 3
  0.62 %, native 2/3 cm staircase 0.81 %, averaged 1.05 %; Section 11.5):
  - forward 8.7 s against 2.8–2.9 s, about 3×;
  - gradient about 38 s (2056 × 18.6 ms) against 6.3 s, about 6×;
  - the 16 ns r = 3 gradient does not fit in 24 GB with √nt checkpointing,
    while the native one needs 18.9 GB.

  The native code is faster per step (Yee only) and takes larger steps (`dt`
  1.6× the hybrid's at r = 3, 4.7× at r = 1).

### 11.5 Benchmark (`examples/hybrid/phase4_benchmark.py`)

Models (`models3d.py`): 2 cm grid, 0.8 m × 1.2 m × 1.2 m, 10-cell absorbers,
ground `εr` 6, `σ` 3 mS/m, 400 MHz Ricker, horizontal `Ez` dipole 0.16 m deep
(≥ 8 cm above the ground), receivers on a line at the antenna depth and on a
buried line at 0.62 m. A3 flat ground at 0.30 m (on a node plane); B3
`d = 0.30 + 0.06 sin(2πy/0.6) cos(2πz/0.8)`; C3 irregular surface from gridded
elevations (PCHIP) with an `εr` = 14 and a 50 mS/m sphere. DG band 0.20–0.44 m,
P1, `h` = 2 cm, adaptive column mesh unless noted. Reference: native DeepGPR on 2.9 mm (dx/7) with cell-averaged
surface materials. The averaged references converge at second order
(ref_3→ref_5 0.68 %, ref_5→ref_7 0.19 % for A3: ratio 3.6 = (1/9−1/25)/(1/25−1/49)),
so ref_7 is accurate to about 0.2 %. Error = relative L2 of the gathers
(surface line offsets ≥ 0.3 m and buried line).

| solver (Yee spacing) | A3 flat | B3 sine | C3 irregular | time A3 / B3 / C3 |
|---|---|---|---|---|
| staircase, native (2 cm) | 23.0 % | 10.4 % | 8.2 % | 1.4 s |
| staircase, native (2/3 cm) | 5.5 % | 0.81 % | 0.79 % | 3–5 s |
| averaged, native (2/3 cm) | 0.87 % | 1.05 % | 0.96 % | 3 s |
| hybrid P1, Yee 2 cm | 6.0 % | 5.6 % | 6.7 % | 2.8 / 3.6 / 3.7 s |
| hybrid P1, Yee 2/3 cm | **0.43 %** | **0.62 %** | 0.85 % | 7.1 / 9.0 / 9.2 s |
| hybrid P1, Yee 2/3 cm, terrain-following mesh | 0.43 % | 0.61 % | 0.88 % | 7.1 / 22.3 / 22.4 s |
| averaged, native (4 mm) | 0.19 % | 0.26 % | 0.24 % | 21–23 s |

Hybrid times are those of the current CUDA library (Section 11.4), measured
before its last three kernel changes, which took another 5 % off. With the
earlier code (terrain-following mesh, PyTorch Yee part) the hybrid runs took
8 / 32 / 32 s (r = 1) and 32 / 100 / 100 s (r = 3) at the same errors; those
results are kept in `results/phase4/old_code_terrain_mesh/`. For flat ground
both meshes are the same; for B3/C3 the adaptive mesh needs 2056/2101 steps
instead of 4557/4569 (`dt` 7.8e-12 s against 3.5e-12 s), at equal accuracy.

Surface / buried lines separately: A3 hybrid r3 0.14 / 0.56 %, staircase 2 cm
8.7 / 29 %; B3 0.28 / 0.79 % vs 1.5 / 14 %; C3 0.18 / 1.3 % vs 1.5 / 12 %.
Peak GPU memory of the hybrid forward run: 1.0–2.7 GB (r = 1), 4.1–5.8 GB
(r = 3).

Findings, stated plainly:

* With the Yee part at the model spacing (2 cm) the hybrid error (6 %) is the
  dispersion of the coarse Yee grid below the band: it does not change with P2,
  `h` = 1 cm or α = 0.3, and drops to 0.4–0.9 % with `fdtd_refinement=3`.
* At equal Yee spacing (2/3 cm) the body-fitted band wins clearly when the
  surface is a grid plane (A3: 0.43 % vs 5.5 % for the staircase, which shifts a
  surface on a node plane by half a cell). For the smooth undulating surfaces
  B3 and C3 the fine staircase is as accurate as the hybrid in these gathers
  (0.8 % vs 0.6 / 0.9 %): its sub-cell surface errors average out, and the
  DG band itself only resolves the surface at its 2 cm column spacing.
* Cost: at comparable accuracy (hybrid r = 3 against the native 2/3 cm
  staircase or averaged Yee) the native CUDA code is about 3× faster for a
  forward run and about 6× for a gradient (Section 11.4). With the earlier code
  it was 10–30× faster. Remaining reasons:
  - the DG band sets the time step: 7.8e-12 s against 1.2e-11 s for the 2/3 cm
    Yee grid; at the model spacing it is 0.20 of the Yee limit;
  - a DG unknown costs much more than a Yee unknown (gathered traces, lifting):
    at r = 1 the band's 11.1M unknowns take 1.2 of 1.6 ms per step against
    2.4M Yee unknowns; at r = 3 the 63M Yee unknowns add about 2.6 ms.

  To reach the hybrid's accuracy on B3/C3 natively takes the 4 mm averaged
  grid (0.26 / 0.24 %, 21–23 s).
* Near field: with the antenna 4 cm above the band, the 2 cm P1 elements leave a
  residual of the dipole near field at the band top directly below it (2–3 % of
  the near-field peak after the pulse, decaying with a 1–2 ns time constant
  through the interface penalty; it does not radiate: elsewhere on the section
  |hybrid − reference| ≤ 0.4 against a wavefield of about 25). Keep antennas
  several cells away from the band or inside it.

Figures: `results/phase4_mesh_<case>.png` (voxel staircase vs body-fitted
triangulated surface; vertical section through the tetrahedra with the Yee
grid, true surface and staircase), `results/phase4_<case>.gif` (`Ez` on the
section through the source for staircase 2 cm, hybrid r3 and the 4 mm
reference, with differences).

### 11.6 3D discrete adjoint (`examples/hybrid/phase4_gradient_check.py`, `tests/test_hybrid_3d.py`)

`HybridPropagator` serves `HybridMaxwell3D` (Section 8: one VJP of the step per
time step, √nt checkpoints of Phase 3). The fused kernels' backward is their
exact transpose.

**Hand-written step transpose** (`vjp3d.py`, used when Yee and DG both run on
the fused kernels). The reverse sweep applies the transpose of the fused step
operation by operation, in reverse order:
* the transpose kernels for the Yee E and H updates, the DG E and H updates
  and the two flux/curl evaluations;
* explicit transposes for the linear coupling operations: interface projection
  `E* = P fe`, ghost rows, interface penalty, `X_h = g n_x (E*_z, −E*_y)`,
  source injection and receiver sampling.

It takes `u^n` and `u^{n+1}` from the recomputed segment (the next checkpoint
for the last step of a segment) and recomputes only `E*`, `X_h`, the penalty
rows and one flux evaluation `T_e(H^{n+1}, E^n)`. Compared with autograd
differentiation of the step, it skips the second forward evaluation and the
zero-filled cotangents of every gather (slice/select backward, accumulation).
Coefficient cotangents accumulate in place over the sweep; autograd then
carries them once to `εr`/`σ` through the coefficient construction.

Checks:
* step level (`test_hand_written_step_vjp_matches_autograd`): against autograd
  of the same step for every state, coefficient and source cotangent, ≤ 2e-15
  (float64; P1/P2, UPML/ABC, r = 1/3, with and without penalty, both meshes);
* sweep level (`test_hand_written_sweep_matches_autograd_sweep`): `εr`, `σ` and
  source gradients ≤ 1e-11 against the autograd sweep
  (`DEEPGPR_HYBRID_AUTOGRAD_VJP=1` selects it);
* the dot-product and finite-difference checks below run through it on the GPU.

`examples/hybrid/phase4_gradient_check.py`, current code (adaptive mesh, fused
kernels, hand-written transpose on the GPU). Model: 14 × 16 × 16, sine surface,
band in the DG, `εr` anomaly in the Yee region, two shots, 320 steps
(`dt` 8.0e-12 s), float64. The results of the earlier run (terrain-following
mesh, autograd VJP) are kept in `results/phase4_gradient_check_autograd_vjp.*`
and are of the same quality: dot product 6.2e-16, finite differences to 5e-11.

| check | result |
|---|---|
| source dot product `<F dw, r>` vs `<dw, Fᵀ r>` | 2.5e-15 |
| checkpoint interval 23 / 1 / 1000 | max rel. difference 2.3e-15 |
| gradient above the surface | exactly 0 |
| pure-Yee 3D vs DeepGPR native adjoint (float32) | data 4.3e-7, `εr` 9.8e-7, `σ` 8.4e-7 |

Central finite differences of `J = ½‖d − d_obs‖²` along random directions,
relative error for relative steps h = 1e-2 … 1e-6 (second-order decay to
round-off):

| region / parameter | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| DG band / `εr` | 6.3e-5 | 6.3e-7 | 6.3e-9 | 3.5e-11 | 1.1e-10 |
| DG band / `σ` | 4.8e-7 | 4.1e-9 | 1.7e-9 | 6.0e-9 | 1.5e-7 |
| Yee region / `εr` | 1.7e-5 | 1.6e-7 | 1.6e-9 | 4.5e-13 | 1.2e-10 |
| Yee region / `σ` | 5.8e-7 | 5.9e-9 | 2.5e-10 | 2.9e-9 | 5.0e-8 |
| bottom interface plane / `εr` | 8.8e-3 | 8.8e-5 | 8.8e-7 | 8.4e-9 | 1.9e-9 |
| bottom interface plane / `σ` | 5.4e-8 | 5.4e-10 | 6.9e-10 | 7.7e-10 | 5.2e-9 |
| side UPML columns / `εr` | 2.0e-5 | 2.0e-7 | 2.0e-9 | 1.3e-10 | 1.3e-10 |
| side UPML columns / `σ` | 8.6e-8 | 1.5e-9 | 1.5e-9 | 1.4e-7 | 4.8e-7 |

A bug found by this check: the Yee halo was zero-padded (as in 2D), so `ca`/`cb`
were NaN/inf on the never-updated halo nodes. The forward pass masks them, but
the 3D step masks with `torch.where`, whose backward forms `0 · NaN`. The halo
is now replicate-padded.

The same checks with the PyTorch step on the CPU (no kernels, autograd VJP;
160 steps): dot product 1.1e-15, checkpoint intervals identical, gradient
above the surface 0. Finite differences for h = 1e-3, 1e-4, 1e-5:
* DG band: `εr` 1.8e-5 / 1.8e-7 / 2.5e-10, `σ` 1.5e-10 / 1.6e-9 / 1.8e-8;
* bottom interface plane: `εr` 3.3e-6 / 3.3e-8 / 3.3e-9, `σ` 2.8e-10 / 1.8e-9
  / 1.2e-8.

### 11.7 Small 3D FWI (`examples/hybrid/fwi3d_topography.py`)

Model (2 cm grid, 0.5 m × 0.64 m × 0.64 m, 8-cell absorbers): irregular
surface around 0.16 m (±4 cm), ground `εr` 5, `σ` 3 mS/m, an `εr` = 7 and a
30 mS/m sphere (radius 10 cm, centres 0.34 m deep), known reflector `εr` 8 below
0.46 m (also in the initial model). 4 `Ez` sources (2 × 2) and 7 × 7 receivers
4 cm below the model top (≥ 8 cm above the ground), 400 MHz Ricker, offsets
< 0.15 m muted. Observed data: native DeepGPR on 4 mm with cell-averaged
surface materials and the 2 cm ground model upsampled. Inversion on the 2 cm
grid from the background, L-BFGS (strong Wolfe), stage 1 `εr` only with data
low-passed to 300 MHz (6 iterations), stage 2 `εr` and `σ` full band (12);
within a stage the objective is divided by its value at the stage start.
Model error `‖m − m_true‖ / ‖m_true − m0‖` over the ground.

**Modelling error decides the outcome.** `--diagnose` compares each solver's
data at the true model with the observed data, against the anomaly signal
`‖d(m_true) − d(m0)‖ / ‖d(m_true)‖ = 2.33 %` of the observed data:

| solver | modelling error at the true model |
|---|---|
| staircase FDTD, 2 cm (native) | 2.19 % |
| hybrid, Yee 2 cm | 2.37 % |
| staircase Yee, 2/3 cm (hybrid code, pure Yee) | 0.38 % |
| hybrid, Yee 2/3 cm | **0.29 %** |

(Before 5 ns: staircase 2 cm 1.9 %, hybrid 2/3 cm 0.24 %, staircase 2/3 cm
0.37 %.) At 2 cm both solvers are as far from the data as the anomalies are
from the background, so an inversion fits modelling error: the first attempt
(600 MHz, 8 cm spheres, signal 1.7 %) ended with model error 1.47 (hybrid,
Yee 2 cm) and 1.48 (staircase, 2 cm) after an 83–84 % misfit reduction. An
inverse-crime check (staircase data inverted with the staircase solver) on that
setup reduced the model error monotonically to 0.86, so the acquisition and
optimiser are adequate; the failure was the modelling error. Two lessons from
that attempt are built into the example: the observed data must use the
inversion grid's pixel convention (sampling the spheres and the reflector
directly on 4 mm shifted the reflector by half a model cell and caused a 14 %
late-time data error for every solver), and the raw normalised misfit (order
1e-4) stalls PyTorch's L-BFGS (absolute tolerances, first step `|g|`).

Results (final iteration):

| solver | misfit reduction | `εr` error | `σ` error | time |
|---|---|---|---|---|
| staircase FDTD, 2 cm (native) | 74 % (4.6e-4 → 1.2e-4) | 1.13 (min 0.975 at iteration 2) | 1.003 | 3 s |
| staircase FDTD, 2/3 cm (native, sub-sampled history) | 98 % (2.8e-4 → 4.9e-6) | 0.896 | 0.992 | 258 s |
| hybrid, Yee 2/3 cm (current code) | 98.8 % (2.8e-4 → 3.3e-6) | **0.886** | 0.993 | 1 758 s |
| hybrid, Yee 2/3 cm (earlier code) | 98.8 % (2.8e-4 → 3.3e-6) | 0.885 | 0.993 | 13 927 s |

Figure: `results/fwi3d_compare.gif`. Each frame is one iteration. It shows x-y
and y-z slices of `εr` and `σ` for the true model and the three inversions,
with the misfit and model-error curves.

* **2/3 cm Yee grid, hybrid vs staircase.** Both fit the data to about 1 % of
  the initial misfit, and their model errors are nearly the same: `εr` 0.886 vs
  0.896, `σ` 0.993 vs 0.992. The hybrid's lower modelling error at the true model
  (0.29 % vs 0.38 %) brings only a small model improvement.
* **What limits the inversion here is not the forward solver.** With either
  solver the `εr` sphere is located (y-z slice) but smeared and with a fraction
  of its contrast, and `σ` is essentially not recovered. Plausible causes:
  - 4 sources, 7 × 7 receivers above the ground;
  - near offsets muted, reflection geometry;
  - one 400 MHz band;
  - only 18 iterations.
* **2 cm staircase.** Its 2.2 % modelling error is as large as the anomaly
  signal, so after iteration 2 it fits modelling error: `εr` error rises to 1.13,
  with artefacts along the surface.
* **Cost.** The same inversion with the current code (adaptive mesh, CUDA
  library, hand-written transpose) took 1 758 s against 13 927 s for the
  earlier code (terrain-following mesh, first DG kernel, PyTorch Yee part,
  autograd VJP), 7.9× less, with the same result. Caveats on both timings:
  - the earlier run shared the GPU with test jobs for part of the time;
  - the current run shared the CPU with the unit-test suite for its first half.

  It is still 6.8× the native 2/3 cm staircase (258 s). Shots are processed
  one at a time in both. The earlier run's files are kept in
  `results/fwi3d_old_code/`.
* **Conclusion for this small 3D test.** The hybrid gives a correctly
  differentiated inversion with the lowest modelling error, but at 2/3 cm no
  meaningful model improvement over the fine staircase, at much higher cost. The
  2D topography case (Section 8.5) showed a clearer benefit (`εr` error 0.79
  vs 0.90).

### 11.8 Usage (3D)

```python
import numpy as np, torch, DeepGPR
from DeepGPR.hybrid import HybridMaxwell3D, Surface3D, HybridPropagator

eps = torch.full((40, 60, 60), 6.0)                    # dx = 2 cm, axis 0 = depth
sig = torch.full((40, 60, 60), 0.003)
surface = Surface3D.sine(0.30, 0.06, 0.6, 0.8)         # depth(y, z) [m]
sim = HybridMaxwell3D(eps, sig, 0.02, surface=surface, dg_band=(0.20, 0.44),
                      pmlthick=10, fdtd_refinement=3, device="cuda",
                      dtype=torch.float32)             # Ez dipoles (component=2)
src = [[[0.16, 0.60, 0.61]]]                           # (nshot, nsrc, 3) (x, y, z) [m]
rec = [[[0.16, y, 0.61] for y in np.arange(0.1, 1.11, 0.04)]]
nt = int(16e-9 / sim.dt)
w = DeepGPR.wavelet.ricker(4e8, nt, sim.dt, 3e-9)
data = sim.run(w, src, rec)["receivers"]               # (nshot, nt, nrx)
prop = HybridPropagator(sim, src, rec)                 # differentiable, discrete adjoint
```

`dg_band=None` gives the pure 3D Yee scheme (DeepGPR's, CPML on all faces).
`mesh="adaptive"` (default) or `"terrain"` selects the band mesh (11.1), or pass
a `TetMesh`. Without `nvcc` or a GPU, or with `cuda_kernels=False`, the step runs
in PyTorch (same results to round-off). Environment switches for comparisons:
`DEEPGPR_HYBRID_DISABLE_GRAPHS=1` (eager step instead of CUDA Graph replay),
`DEEPGPR_HYBRID_AUTOGRAD_VJP=1` (autograd VJP of the step instead of the
hand-written transpose), `DEEPGPR_HYBRID_DISABLE_CUDA=1` (no kernels).

### 11.9 Not implemented in Phase 4, and open issues (3D)

Not implemented, as specified for this phase: no new checkpointing or
recomputation scheme (the 3D adjoint reuses the √nt checkpointing of Phase 3),
no asynchronous offload, local time stepping, subcycling, SIREN, IFWI,
orders above P2 or multi-GPU. Also not done:

* No full-DGTD 3D mode (the band always has Yee above and below; it may not
  reach the x faces).
* The coupling, penalty, source and receiver operations remain PyTorch index
  operations (atomics in `index_add`: the step is reproducible to round-off,
  not bitwise). The DG kernels use one thread per element; the flux kernel is
  close to the memory bound of its gathers. No local time stepping (the DG band
  holds most of the unknowns, so it would gain little; Section 11.4).
* Adjoint memory: √nt checkpoints of the full state plus one recomputed
  segment, about 2√nt states. With `fdtd_refinement=3` one shot of the FWI
  model (11.7) needs about 18 GB, so the FWI example processes the shots one at
  a time. The B3 benchmark model at r = 3 (state 436 MB, 2056 steps) would need
  about 40 GB and does not fit on a 24 GB GPU (native DeepGPR: 18.9 GB with its
  sub-sampled history). Fixing this needs a multi-level checkpointing or
  offload scheme; none was developed, as specified for this phase.
* The adaptive column mesh (default) avoids the thin layers of the
  terrain-following mesh, but it is still column-structured: the horizontal
  resolution is the column spacing everywhere, and elements thin out where the
  surface comes within `h/2` of a band plane. A general unstructured
  tetrahedral mesher (graded, surface-conforming, quality-optimised) is not
  available without new dependencies; a Delaunay prototype was discarded
  (slivers, failed boundary recovery).
* The side UPML shows a weak, linear-in-time drift for initial data with a
  static (zero-frequency) component, e.g. random fields. This is the known
  behaviour of the standard auxiliary-equation UPML for static fields, and the
  PyTorch and CUDA steps show it equally. Physical sources (zero-mean
  wavelets) do not excite it: with a Ricker source the energy decays to 3e-11
  of its peak and stays there up to 240 ns (`test_side_upml_long_run_stable`).
  A complex-frequency-shifted UPML would remove the drift; not implemented.
* The DG band resolves the surface only at its column spacing (`dg_h`); P1
  faces are planar. For smooth surfaces the fine staircase is equally
  accurate in the benchmark gathers (11.5).
* The interface penalty is explicit (stability limit `α_if` 0.3–1 for
  `fdtd_refinement=3`); an implicit treatment of its diagonal part was not
  needed for the defaults.
* Antennas close to the band leave a slowly decaying near-field residual at the
  band top (11.5).
* In the UPML the material conductivity enters the E update as an ordinary
  loss term rather than through Gedney's lossy-medium UPML equations; for
  ground losses `σ/(ωε) ≈ 0.02` this changes the absorption slightly. Not
  measured separately.
