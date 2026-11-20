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

* **2D, refined Yee grid on the flat sand-pit geometry (open, 2026-11-07).** On the
  field geometry of the sand-pit profile (flat surface, band 1 cm above to 5 cm
  below, antennas 2 cm above the sand, 400 MHz, 32 ns), compared with the native
  solver on 0.5 cm: `fdtd_refinement = 1` with the 0.1 penalty is as accurate as the
  native 1 cm solver (background 5.1 / 7.8 % in 0–12 / 12–20 ns against 6.2 / 8.4 %;
  target response 13.8 % against 7.2 %), but `fdtd_refinement = 2` or 3 produce
  spurious late-time energy (12–20 ns: 17–30 %, after 20 ns several times the
  reference) with either flux, and central fluxes do so already at `r = 1` (46 %).
  The 3D hybrid at `r = 3` does not show this (12.6). Not investigated further.

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
* **Coupling, penalty, sources and receivers** were small PyTorch index
  operations (`index_select`/`index_add` of the mortar matrices); since 12.11 they
  are fused kernels too (`csrc/glue3d.cu`), and the PyTorch version is kept for
  autograd. The step issues no host-device synchronisation. Accumulation into
  shared nodes and coefficients uses atomics, so the coupled step as a whole is
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
  (`test_cuda_graph_replay_matches_eager_step`). Superseded in 3D by the
  whole-sweep graphs of 12.11, which need no per-step copy and also cover the
  reverse sweep.
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

## 12. Local time stepping (LTS): the band at its own time step

Modules: `functional3d.lts_coefficients`, `lts_slabs`, `lts_h_half_step`,
`h_half_step` (3D); `functional.lts_h_half_step`, `lts_slabs` (2D); solver options
`lts`, `lts_cfl`, `lts_overlap`, `lts_nu` of `HybridMaxwell3D` and `HybridTM2D`
(and of `HybridSubgridMaxwell3D` through `**hybrid_kw`). Tests:
`tests/test_hybrid_lts.py`. Example: `examples/hybrid/lts_benchmark3d.py`.

### 12.1 Why

The coupled leapfrog of Sections 4 and 11 takes one global step set by the
stiffest part, the DG band (`dt_DG ≈ 1/3 … 1/6` of the Yee limit of the model
grid, Section 6). The Yee grid, which holds most of the volume, is therefore
stepped 3–6 times more often than its own stability limit requires, and any
refinement of the band (thinner layers, `dg_h < dx`, P2) shrinks the time step of
the whole domain. With local time stepping the Yee grid keeps its own step `dt`
and the band takes `p` sub-steps `dt / p`. Together with the Huygens subgrids
(`DeepGPR.subgrid`, time ratio `r`) every part of a layout then runs at its own
rate: Yee at `dt`, DG at `dt / p`, subgrids at `dt / r` (the *time-step ladder*,
`DeepGPR.subgrid.schedule.TimeStepLadder`).

### 12.2 Scheme

The scheme is the leapfrog-based explicit LTS of Diaz & Grote (SIAM J. Sci.
Comput. 31(3), 2009, doi:10.1137/070709414) in its stabilised form (Grote,
Michel & Sauter, Math. Comp. 90, 2021, doi:10.1090/mcom/3650), written in the
first-order variables of the coupled system `M_e de/dt = K h`, `M_h dh/dt = −Kᵀ e`
of Section 4. Let `P` select the *fine* unknowns and `r(e) = M_h⁻¹ Kᵀ e`,
`q(h) = M_e⁻¹ K h` be the H and E rates. One macro step replaces the H half step
of the fine unknowns by the recursion (`τ = dt / p`, `z` and `η` live on the fine
set, the coarse unknowns are frozen at `e^n` inside `r`)

```
z_0 = e^n,                 η_0 = −τ c_0 r(z_0)
z_(k+1) = z_k + τ q(η_k),  η_(k+1) = β_(k+1) η_k − τ c_(k+1) r(z_(k+1)),   k = 0 … p−2
h^(n+1/2) = h^(n−1/2) + (2/p) Σ_k η_k = h^(n−1/2) − dt Σ_j w_j r(z_j)
```

followed by the ordinary E update `e^(n+1) = e^n + dt q(h^(n+1/2))` of *all*
unknowns and the ordinary H half step of the coarse unknowns. The coefficients
come from the Chebyshev polynomials `T_k` at `δ = 1 + ν/p²`:
`ω = 2 T_p'(δ)/T_p(δ)`, `c_0 = p²/(ω δ)`, `c_k = (2p²/ω) T_k(δ)/T_(k+1)(δ)`,
`β_k = T_(k−1)(δ)/T_(k+1)(δ)` (`lts_coefficients`); the weights `w_j` follow from
the recursion and sum to one. On the fine E unknowns the sequence is exactly the
second-order LTS `e^(n+1) = 2 z_p − e^(n−1)` of the papers (checked on a 1D
chain to 1e-12), so for central fluxes without losses the coupled scheme
conserves the modified energy of the LTS, which in three-level form reads
`E^(n+1/2) = (‖e^n‖²_M − ⟨e^(n+1), e^(n−1)⟩_M)/(2 dt²)`; this invariant is
conserved to round-off (2e-15) for p = 1 … 7 in 2D and 3D
(`test_modified_energy_invariant`). For `ν = 0` the scheme is the original
Diaz–Grote LTS (`c_0 = 1/2`, `c_k = β_k = 1`, `w_0 = 1/p`, `w_j = 2(p−j)/p²`); for
`p = 1` it is the plain leapfrog step, bitwise (`test_one_substep_matches_plain_step`).

**Fine set and overlap.** `P` holds the DG unknowns, the Yee interface planes
(rows in 2D) and `lts_overlap` further Yee planes on the Yee side of each
interface (`lts_slabs`). The interface planes must be in `P`: with the DG
unknowns alone the scheme is unstable for every `p` (also on the 1D chain),
because the half-cell SBP closure of the interface plane carries the band's
stiffness. The overlap planes are the usual safety layer of the LTS literature;
1 is the default, 0 the minimum.

**Dissipative and absorbing terms.** The recursion uses the central-flux
operators with the external inputs frozen at time `n` (the Yee trace `E*` of the
frozen coarse planes, the ghost planes from the DG `η`). Conductivity, the UPML
of the band and of the Yee side columns, the DG CPML in 2D and the interface
penalty act once per macro step on the *averaged* rates: the band takes its
ordinary update with the averaged curl terms `Σ_j w_j T(z_j)`; the Yee grid takes
its ordinary (fused) update and the fine planes are corrected afterwards by
`−(dt/μ)(r_avg − r_0)`, which is exact also through the linear UPML/CPML
recursions (`B_fd += −b1 δr`, `H += −new·b1 δr` on UPML nodes). With `p = 1` all
of this reduces to the original step. Long runs with these terms (40 ns, Ricker
source, CPML, σ = 3 mS/m, terrain and Delaunay meshes, p = 2 … 6) decay exactly
like the plain scheme (`diag_diss2d`, final/peak 2.5e-4).

**Side UPML (3D): the macro step is capped by the absorber.** With the UPML of the
band integrated at the macro step (the band's PML elements are part of the fine
set and sub-cycle losslessly), the coupled scheme is unstable once the largest
UPML rate times the macro step exceeds about 0.8 (`s_max dt` 0.63 stable, 0.83
unstable; a strong conductivity at the macro step, `σ dt/ε` up to 3, is stable,
so this is specific to the UPML's auxiliary structure; sub-stepping the UPML
recursion `p` times per macro step and damping the recursion variables in the
PML elements do not help). `lts="auto"` therefore limits the macro step to
`lts_upml_limit / s_max` (default 0.6): with the default `dg_pml_strength = 0.4`
this is 0.65 of the Yee limit (p = 3 on the test model, p = 4 on B3), with
`dg_pml_strength = 0.2` the full Yee step is available at the price of a weaker
side absorber (`exp(−1.6 · 0.2 · N)`). The 2D band uses a CPML (stretched
derivatives), which needs no cap (40 ns runs at the full Yee step).

**Penalty fluxes with LTS: split damping (3D).** The explicit `α` penalty
(Section 5) is unstable inside the sub-cycled recursion in every arrangement
first tried: with the penalty fluxes applied to `(z, η)` inside the recursion, and
with central fluxes inside and the penalty evaluated once per macro step at
`(E^n, H^(n−1/2))` (for `α = 0.1` and even `α = 0.02` at `p ≥ 3`; 2D, `p = 2 … 7`:
growth to 1e100 within 10 ns). A scalar model explains why. For the leapfrog pair
`h' = a h − b τ ω e`, `e' = e + τ ω h'` the scheme is stable iff `b Q < 2 (1 + a)`,
where `Q` is the effective `τ²ω²` (for the LTS: the stabilised recursion
polynomial, `Q < 4`). An explicit penalty at the macro step has `a = 1 − τ d`,
`b = 1` and needs `Q < 4 − 2 τ d`, which fails for the strongly damped grid-scale
modes, exactly those the recursion pushes towards `Q ≈ 4`. Crank–Nicolson
damping has `b = (1 + a)/2` and keeps `Q < 4`; the same structure results from a
Strang splitting of the update:

    F^new = A (F^old + D/2) + D/2,   D = F_cons − F^old,   A = (I − B/k)^k

with `F_cons` the conservative update (LTS recursion with *central* fluxes, UPML,
conductivity, interface penalty as before) and `B` the dissipative jump terms of
the penalty flux (`(dt/μ) curl P_H(H)` for `H`, `−(dt/ε) curl P_E(E)` with the PEC
penalty for `E`; `functional3d.lts_damp`, `lts_penalty_rate`). `|a| < 1` needs
`ρ(B)/k < 2`; the solver estimates `ρ(B)` by power iteration at set-up and takes
`k = ⌈ρ / 1.4⌉` (`lts_pen_k`; `k = 1` for `α ≤ 0.3` on the models here, 4 for
`α = 1`). The impedance weighting of the averages in the upwind flux (`Δb0`) is
dropped, so the split scheme is "central averages + jump penalty", consistent like
the original. One more ingredient was needed: damping inside the band's side-UPML
elements destabilises the scheme for `α ≥ 0.3` (the UPML update is itself a
damped recursion, so `D` depends on `F^old` there), so the penalty is switched off
on every face of a UPML element and of its neighbours (`lts_penalty_faces`); the
UPML absorbs the grid-scale modes there. With that, `HybridMaxwell3D(flux=α,
lts=...)` accepts `α > 0` (Silver–Müller side faces are still rejected); the 2D
solver still requires central fluxes.

Why this matters: **central fluxes are not accurate on rugged terrain.** On the
model of `examples/hybrid/fwi3d_lts.py` (Section 12.6: surface 0.17 m ± 6 cm with
random roughness, adaptive column mesh, Yee 2/3 cm, 400 MHz, 12 ns, centre shot,
data of the converged 4 mm staircase solver; 4 mm vs 2.86 mm: 0.09 %):

| band | flux | time stepping | `dt` [s] | `p`, `k` | error vs staircase data |
|---|---|---|---|---|---|
| P1, 2 cm columns | central | global | 8.33e-12 | – | 3.02 % (36–43 % after 7 ns) |
| P2, 2 cm | central | global | 4.83e-12 | – | 2.45 % |
| P1, terrain-following mesh | central | global | 3.98e-12 | – | 2.73 % |
| P1, 2 cm | α = 0.1 | global | 7.67e-12 | – | 0.33 % |
| P2, 2 cm | α = 0.1 | global | 4.44e-12 | – | 0.35 % |
| P1, 1 cm | α = 1 | global | 1.16e-12 | – | 0.33 % |
| P1, 2 cm | α = 0.1 | LTS, split penalty | 1.16e-11 | 2, 1 | **0.32 %** (0.02 % from global) |
| P2, 2 cm | α = 0.1 | LTS, split penalty | 1.16e-11 | 3, 1 | **0.36 %** (0.09 % from global) |
| P1, 2 cm | α = 0.3 / 1.0 | LTS, split penalty | 1.16e-11 | 2, 1 / 2, 4 | 0.33 % / 0.33 % |

The central-flux variants disagree with each other by 3.5–3.7 % (grid-scale
spurious modes that depend on the mesh); every penalty variant agrees with the
others to 0.1–0.2 %. On the smooth B3/A3 surfaces central fluxes were as
accurate as `α = 0.1` (12.4), which is why this went unnoticed. Stability of
the split scheme: random data in a closed PEC box decay for `p = 2, 3, 5` with
`α = 0.1` and `α = 1` (1800 macro steps); on the rugged model random initial
fields over 5000 macro steps (58 ns) drift exactly like the global scheme
(max |E_dg| 3.8 → 4.5 against 3.5 → 4.2 for global stepping over the same time;
the Yee fields keep their static part).

**Stabilisation.** The unstabilised LTS (`ν = 0`) is energy conserving but its
modified energy is indefinite at isolated time steps; on the dense spectrum of a
3D tetrahedral band some mode always resonates: with random initial data the 2D
amplitudes settle at 20–100 times the level of the plain scheme and in 3D they
grow exponentially (×10 over 2000 fine-step equivalents at `p = 3`). With
`ν = 0.1` or `0.3` (damped Chebyshev recursion) the amplitudes stay at the level
of the plain scheme for `p = 2 … 6` at `0.9` of the Yee limit, with overlap 0 and
1, and sub-steps at 0.7 and 0.9 of the band's limit (4000 fine-step equivalents,
closed PEC box, central fluxes). Default `ν = 0.3`; the price is that the band's
sub-step must stay slightly below its limit (the stabilised polynomial leaves
`(0, 4)` at about `3.5` instead of `4`), which the default `cfl_safety` already
provides.

**Cost.** Per macro step the band evaluates `p` E-curls in the recursion,
`p − 1` H-curls of `η` and one H-curl for the E update: exactly the `2p` operator
evaluations of `p` global steps. The leapfrog LTS does not reduce the work of
the fine part; what it removes are the `p − 1` extra steps of the coarse part
(the Yee curls and their side UPML, the DG update and UPML passes, the mortar
and ghost-plane products, the elementwise updates), at the price of `p − 1`
Yee curls on the few slab planes and the recursion variables of the band. The
gain is therefore bounded by the share of the global step that is not the DG
operator; it is largest when the Yee grid is large and the band thin
(`fdtd_refinement = 1`, deep domains), and nil when the Yee grid is refined to
the band's step (`fdtd_refinement = 3`, where `p = 1`). What the LTS buys in
every case is the freedom to refine the band (`dg_h`, P2) without touching the
step of the volume. Measured costs: 12.4 and 12.5.

**Cost-aware selection (3D).** `lts="cost"` evaluates the cost model of
`DeepGPR.subgrid.schedule.CostModel` (Yee and DG unknowns, `2p` DG operator
evaluations per macro step plus the split-penalty damping, a per-step overhead)
for `p = 1` (global stepping at the band's limit with the explicit penalty) up to the
`p` of `"auto"` and takes the cheapest per simulated second (`lts_cost_table` in the
summary). On the Yee 2/3 cm FWI model of 12.6 the model predicts a tie at P1 (132
against 132 ms per ns) and a 1.1× gain for LTS at P2, in line with the measurements.

**Time-step selection.** `lts="auto"`: `dt = lts_cfl × 1/(c_max √(Σ 1/h²))` (the
Yee limit of the refined grid with the fastest medium) and the smallest `p` with
`dt / p ≤ cfl_safety × 2/√λ_max` (the band's limit from the Lanczos estimate of
Section 5); `p = 1` falls back to the plain scheme. An integer `lts = p` with an
explicit `dt` uses `dt` as the macro step.

### 12.3 Adjoint

The LTS step is a composition of the same differentiable operators as the plain
step (fused DG and Yee kernels with exact transposes, the mortar, slab views of
the Yee arrays), so the reverse sweep of Section 8.2 differentiates it with
autograd: the gradient is the exact discrete adjoint of the sub-cycled scheme,
interface coupling and sub-cycling included. The hand-written step transposes
(`vjp2d`, `vjp3d`) are bypassed (`supported()` returns `False` for `lts > 1`),
which costs about 1.3–1.5× in the gradient (Section 11.4). Verified by central
finite differences in 2D and 3D with `lts = 3` (`test_gradient_finite_differences`:
relative errors below 1e-7 / 1e-6 in float64, CPML, conductivity and penalties
on). CUDA Graph replay captures the LTS step like the plain one, and the
propagator replays it both in the forward pass and in the segment recomputation
of the reverse sweep (`adjoint.py`); only the autograd VJP of each step runs
eagerly.

### 12.4 Verification (CPU) and results

```bash
python -m unittest tests.test_hybrid_lts tests.test_subgrid_schedule   # 40 tests, ~6 min CPU
```

* one sub-step reproduces the plain step bitwise (2D, 3D);
* the three-level invariant is conserved to 2e-15 for p = 1, 2, 3 (overlap 0 and
  1), 2D and 3D, closed PEC box, central fluxes;
* random initial data at the automatic schedule stay bounded over 2000 (2D) /
  1200 (3D) macro steps;
* consistency with the global leapfrog on a smooth problem with absorbers,
  conductivity and (3D) the interface penalty: relative receiver difference
  below 5 % in the tests, 0.4 % on the 3D test model at p = 3 (the difference is
  the time-discretisation error of the two schemes, both second order);
* central finite differences of the misfit against the autograd adjoint of the
  LTS step: below 1e-7 (2D, p = 3) and 1e-6 (3D, p = 3) in float64;
* the subgrid-hybrid step without subgrids equals the hybrid with LTS to 1e-12;
* split penalty (`test_split_penalty_*`): random data in a closed box decay with
  `α = 0.1`, `p = 3`; agreement with the global penalty scheme; central finite
  differences of the adjoint below 1e-6 (eps inside and below the band, sigma);
  on CUDA the fused step and its transpose equal the PyTorch step;
* the hand-written transposes are bypassed; on CUDA the fused LTS step equals the PyTorch step and CUDA Graph
  replay equals the eager step (GPU tests).

Accuracy and cost against the converged references
(`examples/hybrid/lts_benchmark3d.py`, phase-4 cases B3 and A3, Yee part on the
2 cm model grid, central fluxes in both arms, float32, one shot, 16 ns, RTX 4090,
graph replay, the fused kernels of 12.5; `results/lts_benchmark3d*.json`):

| case | scheme | p | dt [s] | steps (16 ns) | error vs `ref_7` | LTS vs global receivers | forward | gradient (2.5 ns window) |
|---|---|---|---|---|---|---|---|---|
| B3 sine | global | 1 | 8.46e-12 | 1892 | 5.65 % | – | 3.08 s (1.63 ms/step) | 7.16 ms/step |
| B3 sine | LTS | 3 | 2.50e-11 | 641 | 5.54 % | 0.14 % | 2.59 s (4.04 ms/step) | 22.5 ms/step |
| A3 flat | global | 1 | 1.08e-11 | 1479 | 6.17 % | – | 2.40 s (1.62 ms/step) | 7.13 ms/step |
| A3 flat | LTS | 3 | 2.50e-11 | 641 | 6.09 % | 0.12 % | 2.60 s (4.06 ms/step) | 22.8 ms/step |

* Central fluxes cost nothing on the column mesh: the global central-flux
  errors equal the phase-4 values with `α = 0.1` (5.64 % / 6.00 %, 11.5).
* The LTS reproduces the global scheme to 0.1 % (receivers) and is marginally
  more accurate (the Yee part runs near its own limit, where Yee is less
  dispersive). The macro step is capped at `p = 3` by the side UPML
  (`lts_upml_limit`, see above); without the cap `p = 4`.
* Gradients. The relative gradient differences printed by the benchmark (50–85 %)
  come from the source node of a 2.5 ns window during which the source is still
  on: the self-field of the Yee source node depends on the time step and
  dominates the norm of `∂J/∂ε` for `J = ½‖d‖²`. Over a 6 ns window
  (`lts_gradient_mask_gpu.py`) the LTS and global gradients of B3 agree to
  0.17 % (`ε`) and 0.13 % (`σ`) overall and to 4–7 % below the band, where the
  gradient is small; on the test model every path (float32/float64, fused/
  PyTorch, autograd/hand-written) agrees to 0.65 % (`lts_gradient_check_gpu.py`),
  i.e. the two discrete schemes' own time-discretisation difference.
* **Cost.** With the fused recursion kernels and the operator epilogue (12.5)
  the LTS forward run is 1.19× faster than global stepping on B3 (4.04 ms per
  macro step against 3 × 1.63 ms) and 0.92× on A3, whose flat surface allows a
  larger global step (1.08e-11 s: the automatic schedule still takes `p = 3`
  sub-steps of the UPML-capped macro step, at 77 % of the band's limit). The
  gradient is 0.93× (B3) and 0.72× (A3): the autograd transpose of the
  sub-cycled step against the hand-written transpose of the plain one. These
  models are band-dominated (11.1 M DG against 2.4 M Yee unknowns; the six DG
  operator evaluations per macro step take 2.5 of its 3.9 ms of GPU time, and
  three global steps need the same six), so the LTS cannot gain more than the
  non-operator share of the global step; the first implementation (PyTorch
  recursion, 7.73 ms per macro step eagerly, 6.8 ms with graph replay) lost it
  to launch-bound elementwise work, which 12.5 removes. The LTS therefore is a modest forward gain on
  band-dominated 3D models, a larger one on Yee-dominated layouts (12.5), and in
  every case the freedom to refine the band without shrinking the step of the
  volume.
* 2D (`lts_benchmark2d.py`, rugged model of 8.7, 14 shots, 24 ns, float32,
  `results/rugged2d/lts_benchmark2d*.json`). Central fluxes on the Delaunay mesh
  with the antennas inside the band excite spurious modes (94 % error for both
  arms, `*_delaunay_central.json`), so the benchmark uses the terrain-following
  mesh, whose thin layers under the 61° slopes shrink the band's step to 2.2 ps:

  | Yee part | scheme | p | dt [s] | steps | error vs `ref9` | LTS vs global | 14-shot forward | 1-shot gradient |
  |---|---|---|---|---|---|---|---|---|
  | 2.5 cm (`r = 1`) | global | 1 | 2.21e-12 | 10866 | 10.23 % | – | 8.08 s (0.74 ms/step) | 0.63 ms/step |
  | 2.5 cm (`r = 1`) | LTS | 25 | 5.31e-11 | 452 | 9.99 % | 0.67 % | 8.53 s (18.9 ms/step) | 27.3 ms/step |
  | 0.83 cm (`r = 3`) | global | 1 | 2.21e-12 | 10866 | 4.16 % | – | 9.88 s (0.91 ms/step) | 0.65 ms/step |
  | 0.83 cm (`r = 3`) | LTS | 9 | 1.77e-11 | 1357 | 4.15 % | 0.07 % | 9.51 s (7.0 ms/step) | 10.4 ms/step |

  The LTS is stable and as accurate as the global scheme even with 25
  sub-steps (the stabilised recursion), and costs the same time (0.95× / 1.04×,
  fused 2D slab kernels and graph replay) because the band dominates: 61k
  triangles against a 64 × 128 (or 192 × 384) Yee grid, and the band evaluates
  the same `2p` operators per macro step as the global scheme (the GPU time of
  one `r = 1` macro step is 1.6 ms, 0.6 of them the 50 DG operator evaluations;
  the global step takes 0.07 ms). The gradient of the autograd transpose is 2×
  slower than the hand-written one (0.56× / 0.50×).

### 12.5 Fused kernels of the recursion (`csrc/lts3d.cu`)

The first implementation of the recursion was PyTorch glue around the fused DG
operator: per sub-step about ten full-size elementwise temporaries on the band
and some 70 small kernels per slab (Yee differences, masks, mortar products,
UPML sub-steps), which made the LTS slower than global stepping (7.73 ms per
macro step on B3, 1.5× the three global steps) although it issues no more DG
operator evaluations. `csrc/lts3d.cu` (wrappers in `cuda3d.py` / `cuda2d.py`,
autograd Functions with hand-written transposes) now fuses everything but the
DG operator and the mortar products:

| kernel | replaces | transpose |
|---|---|---|
| `z`, `eta` (3D; `z2`, `eta2` 2D) | the DG recursion updates `z += (dt/p ε) curl`, `η = β η − c_k (dt/p μ) curl`, `cavg += w curl` from the six curl terms (3D: superseded by the epilogue below, kept for the 2D step and as the reference) | elementwise; the `cb0` cotangent summed per element (one thread per element, no atomics) |
| epilogue of `terms_fwd` (`dgtd3d.cu`, modes 1 / 2; `DGCudaOps.terms_eta`, `terms_z`) | the same two updates applied to the curl inside the operator kernel: the six terms `T` are never written or read back (`−88 MB` per evaluation on B3) | `terms_bwd1` forms the six-term cotangent in registers from the cotangents of `η_new`/`cavg_new` (or `z_new`); the `cb0` cotangent from the curl kept by the forward only in the gradient pass |
| `slab_q`, `slab_r` (3D; `slab_q2`, `slab_r2` 2D) | the Yee slab recursion: E rates `curl_bwd(η)` on the fine planes (masked, × `dt/p ε`), H rates `curl_fwd(z)`, the `η` / average updates and, in the last sub-step, the corrections `−(dt/μ) fine_h (r_avg − r_0)` of the fine H planes (2D: with the CPML auxiliaries) | gathers of the difference cotangents with the matching boundary rules; `cb0` cotangent per node |
| `fold` | the `p` UPML sub-steps of the side-UPML nodes of the fine Yee planes (gathered entries) | the recursion run forward in registers and backward; factor cotangents per entry |
| `ltsupd` (modes H and E) | the macro update of the band, `H − (dt/μ) cavg` / `ca E + cb curl`, with the `p`-fold UPML sub-steps of the PML elements in the same pass (replaces `update_h`/`update_e` + `index_select`/`index_copy` of the PML elements + the fold) | one thread per element: `ca`/`cb` cotangents summed over nodes and shots, factor cotangents per entry |

The mortar products of all couplings and both components are one
`SparseOp` each (`functional3d.trace_op`, `ghost_op`; 2D `functional.trace_op`,
`ghost_op`), which also serves the plain step (its launch count fell from 128 to
94 per step). The propagator replays the captured LTS step in its forward pass
and in the segment recomputation of the reverse sweep. Correctness: the fused
LTS step equals the PyTorch step to 1e-11 (float64, random state), the
finite-difference gradient tests pass on CUDA through the fused transposes, the
transpose of the fused step equals autograd through the PyTorch step
(`test_fused_step_transpose_matches_pytorch_step`, 2D and 3D), graph replay
equals the eager step.

Kernel-level profile of one B3 macro step (`p = 3`, float32, one shot,
`torch.profiler`; `examples/hybrid/lts_profile3d.py` times the pieces with
CUDA events):

| | GPU time | of which |
|---|---|---|
| global step (× 3 = 5.2 ms) | 1.74 ms | DG operator 2 × 0.41, updates 0.46, Yee 0.12, mortar / index 0.3 |
| LTS macro step, PyTorch recursion | 7.0 ms (eager) / 6.8 ms (graph) | elementwise temporaries, ~600 launches |
| LTS macro step, fused DG recursion only | 5.0 ms | slab glue 1.8, UPML folds 1.0, mortar 0.5 |
| LTS macro step, all fused | 4.4 ms | DG operator 6 × 0.40 = 2.4, `ltsupd` 0.37, `eta` 0.36, `z` 0.28, mortar / index / copies 0.6, slabs + folds 0.1 |
| LTS macro step, + operator epilogue | **3.9 ms** | DG operator 6 × 0.42 = 2.5, `ltsupd` 0.35, mortar / index / copies 0.7, slabs 0.07 |

The epilogue removes the `η`/`z` passes entirely (the operator kernel is not
slower for it: its cost is the gathers and the volume products, not the write
of `T`), 0.5 ms per macro step. The remaining excess over the six operator
evaluations (1.4 ms) is the macro update with the UPML folds (0.35) and the
mortar / index glue (≈ 0.7 ms, ≈ 330 launches); a further 0.2–0.3 ms would come
from folding the E update (`ltsupd` mode 1) into the epilogue of the E half
step's operator call and the H update into the last `η` call, which is not
done.

Yee-dominated layouts (`examples/hybrid/lts_benchmark3d_deep.py`: 2 cm grid,
1.6 m deep, sine surface 0.26 ± 0.03 m, 6-layer band 0.20–0.32 m, central
fluxes, float32, 16 ns, two repetitions):

| lateral size | Yee nodes | DG nodes (K) | side UPML | p | global | LTS | receivers | forward speed-up |
|---|---|---|---|---|---|---|---|---|
| 1.2 × 1.2 m | 594 k | 922 k (230 k) | 0.4 | 3 | 1.15–1.17 ms/step | 2.71–2.72 ms/step | 0.01 % | 1.24–1.26× |
| 1.2 × 1.2 m | 594 k | 922 k (230 k) | 0.2 | 5 | 1.21 ms/step | 4.13–4.27 ms/step | 0.13 % | 1.15–1.19× |
| 2.0 × 2.0 m | 1.34 M | 2.07 M (518 k) | 0.2 | 5 | 2.14–2.24 ms/step | 8.3–8.6 ms/step | 0.02 % | 1.04–1.06× |

Even with 80 Yee planes under a 6-layer band the band still holds more unknowns
than the Yee grid (24 P1 nodes per column cell against 6 Yee components) and
costs about 30× more per cell and step, so the LTS gain stays at 1.0–1.25×
on this GPU, where the fused Yee update of 1.3 M nodes takes 0.2 ms (the `p = 5`
and 2.0 m rows were measured before the operator epilogue, which adds about
0.05× to them). The gain
grows with the Yee share (deeper domains, coarser bands, P1 instead of P2 in the
band) and with `p` only when the Yee part is the cost. The automatic schedule
maximises the macro step (the Yee limit or the UPML cap) and rounds `p` up,
which can leave the band's sub-step well below its limit (A3: 77 %); when the
band dominates, a schedule that fits `p` whole band steps into the macro step
(`lts = p` with `dt = p × dt_band`) is cheaper.

### 12.6 3D joint ε/σ FWI on rugged terrain with LTS (`examples/hybrid/fwi3d_lts.py`)

**Set-up.** 2 cm model grid, 0.56 m deep, 0.64 m × 0.64 m, 8-cell absorbers;
surface 0.17 m ± 6 cm (9 × 9 elevation samples: a sine pattern plus random
roughness, PCHIP); ground εr 5, σ 3 mS/m; an εr 7.5 sphere and, 24 cm away, a
40 mS/m sphere (radius 9 cm, centres 0.38 m deep); a known reflector εr 8 below
0.50 m (in the initial model). 3 × 3 `Ez` sources and 7 × 7 receivers at 4 cm
depth in the air (7–19 cm above the ground), 400 MHz Ricker, 12 ns, offsets below
12 cm muted. Observed data: native DeepGPR on 4 mm with cell-averaged surface
materials (converged: 4 mm and 2.86 mm differ by 0.09 %). Inversion: hybrid with
Yee 2/3 cm (`fdtd_refinement = 3`), P1 band 0.08–0.28 m, flux 0.1 (with LTS: the
split penalty of 12.2), float32, one shot at a time; L-BFGS on `ε = ε0 + m_e`,
`σ = σ0 + ω0 ε0 m_s` (one unit of either changes the loss tangent at 400 MHz by the
same amount; the joint-stage gradients are then balanced, 0.17 against 0.11);
stage 1 εr only, 250 MHz low-pass, 8 iterations; stage 2 joint, full band, 24.

**Three design lessons from the diagnostics** (each found by a forward check
against the 4 mm data before the inversion):

* *Fluxes.* Central fluxes give 2.5–3 % data error on this surface (12.2), the
  0.1 penalty 0.33 %: the LTS needs the split penalty here.
* *Antenna–band distance.* With a single Yee cell between the antennas and the
  band's interface plane (band top 0.06 m on the 2 cm grid of the AMR study) the
  data error was 35 % (spurious late-time fields), with two cells 3.0 %. Sources and
  receivers must stay at least two Yee cells from the interface plane (or inside
  the band); the band top is therefore 0.08 m.
* *Data weighting.* 98 % of the data energy arrives before 6 ns (direct wave and
  surface reflection), while the anomalies and the reflector echo come later; the
  anomaly signal is 1.65 % of the data. A first run without gain reduced the
  misfit by 94 % in 15 iterations but left the models almost unchanged (εr error
  0.947, σ error 1.009: σ slightly worse; `results/fwi3d_lts/old_nogain/`). The
  misfit therefore uses time-gained data `(t − 1 ns)^2 d` (observed and simulated):
  anomaly signal 15.9 %, modelling error of the inversion solver 0.88 % (global) /
  0.89 % (LTS), LTS against global 4e-4.

**Results** (RTX 4090, float32, 32 L-BFGS iterations each, 72 gradient
evaluations; `results/fwi3d_lts/report_P1.md`, `final_P1.png`, `evolution_P1.gif`):

| arm | p | dt [s] | misfit (final/initial) | εr error | σ error | εr in the εr sphere (true 7.5) | σ in the σ sphere (true 40 mS/m) | σ in the εr sphere | εr in the σ sphere | wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid + LTS (split penalty) | 2 | 1.155e-11 | 0.0024 | 0.883 | 0.973 | 5.74 | 5.5 mS/m | 2.9 mS/m | 5.03 | 167 min |
| hybrid, global step | 1 | 7.711e-12 | 0.0026 | 0.887 | 0.973 | 5.71 | 5.4 mS/m | 3.0 mS/m | 5.03 | 124 min |

* **The two discretisations invert alike.** Their final updates differ by 4.9 %
  (εr) and 11.9 % (σ) of the global arm's update, every metric agrees to the
  second digit, and the misfit curves track each other per iteration: the exact
  adjoint of the sub-cycled scheme (hybrid interface, sub-cycling, split penalty)
  gives the same inversion as the plain scheme.
* **What the data resolve.** Both anomalies are located; the permittivity sphere
  comes out smeared and at about 30 % of its contrast (mean 5.7), the
  conductivity sphere at about 7 % (mean 5.5 mS/m) with diffuse σ updates near
  the reflector, and a low-εr band at the top of the σ sphere is the main
  crosstalk (the sphere averages show none: σ 2.9–3.0 mS/m in the εr sphere, εr
  5.03 in the σ sphere). σ moves only after εr has explained most of the data
  (from iteration ~14), the usual order in joint surface-GPR FWI; the misfit is at
  the modelling floor of the solver (0.88 % data error) at the end, so more
  iterations would not change much. These are the limits of reflection data from
  above (3 × 3 sources, 7 × 7 receivers, one band), not of the solver.
* **Cost.** LTS took 1.35× the wall time of global stepping (139 against 103 s per
  gradient evaluation, both with the hand-written transposes, the LTS arm on an
  exclusive GPU): at P1 with Yee 2/3 cm the macro step (p = 2) saves 1.5× Yee
  steps but evaluates the DG operator twice as often per unit time (recursion,
  E update and split damping), as predicted by the cost model (`lts="cost"` would
  choose global stepping here).
* A first run without the time gain (`old_nogain/`) reduced the misfit by 94 % in
  15 iterations while εr error stayed at 0.947 and σ error at 1.009: the gain is
  what makes the anomalies matter to the misfit.
* The deep variant of this model with scheduled refinement (coarse level, Huygens
  boxes, global refinement): `docs/SUBGRID.md` Section 12.

### 12.7 Hand-written transpose of the LTS step (`vjp3d_lts.py`)

The reverse sweep of the LTS step on CUDA no longer differentiates the sub-cycled
step with autograd. `vjp3d_lts.step_vjp` recomputes the recursion intermediates
(`z^m`, `η^m`, the slab fields with their ghost planes, `E*^m`, the averages, the
split-damping stages, `T_e`) from `u^n` and `u^(n+1)` and applies the transposes
in reverse order: the epilogue terms kernels (the six-term cotangent formed in
registers from the cotangents of `η`/`cavg` or `z`), the slab kernels, the folds,
the LTS macro updates, the Yee H/E transposes, and explicit transposes of the
trace, the ghost planes, the interface penalty, `X_h` and the sources
(`adjoint._fused_vjp`; `DEEPGPR_HYBRID_LTS_AUTOGRAD=1` restores autograd). It equals
autograd through the PyTorch step to 3e-10 (float64; states, every coefficient
cotangent and the source samples; central, split penalty with `k = 1` and `k = 4`,
`tests/test_hybrid_lts.py::test_hand_written_lts_transpose_matches_autograd`), and
the finite-difference gradient tests pass through it on CUDA.

Cost on an exclusive GPU (model of 12.6, Yee 2/3 cm, one shot, float32, flux 0.1;
`gradient ≈ (2 forward + reverse) per step / dt`):

| band | scheme | dt [s] | forward step | reverse step | gradient per simulated ns |
|---|---|---|---|---|---|
| P1 | global (vjp3d) | 7.71e-12 | 1.29 ms | 3.15 ms | 743 ms |
| P1 | LTS p = 2, hand-written | 1.155e-11 | 2.45 ms | 6.81 ms | 1013 ms (autograd 11.11 ms → 1385 ms) |
| P2 | global (vjp3d) | 4.47e-12 | 2.02 ms | 5.56 ms | 2147 ms |
| P2 | LTS p = 3, hand-written | 1.155e-11 | 5.28 ms | 16.16 ms | 2313 ms (autograd 21.43 ms → 2769 ms) |

The hand-written transpose is 1.63× (P1) and 1.33× (P2) faster than autograd.
The LTS gradient is nevertheless 1.36× (P1) and 1.08× (P2) the cost of global
stepping on this model, and its forward run 0.91× / 1.19× as fast (graph replay,
same model with central fluxes: 163 vs 180 ms per ns at P1, 393 vs 331 ms at P2).
With the split penalty a macro step evaluates the DG operator `2p` times for the
recursion and the E update plus `2k` times for the damping (6 against 3 for the 1.5
global steps it replaces at P1), so the LTS saves Yee work but adds DG work here,
and the reverse step recomputes the recursion. Remedies, not done: a face-only
penalty kernel, a schedule whose macro step is a whole multiple of the band's step
(`lts="cost"` already picks p = 1 when the LTS does not pay), and keeping the
recursion intermediates of the segment recomputation instead of recomputing them.


### 12.8 Field data: 2D sand-pit profile with the hybrid and its exact adjoint

The measured 500 MHz common-offset profile over a flat sand pit (private data;
the native field FWI of that profile: processing v2, wavelet by Wiener
deconvolution of the direct wave, even traces for training (87), odd traces held
out (86), L1 loss, Adam, stages 450 MHz / 700 MHz / full band with 100 / 100 / 150
iterations, TV 3e-4, model updated from 6 cm below the surface) inverted with
`HybridTM2D`: flat surface, P1 DG band from 1 cm above to 5 cm below the sand
surface, flux 0.1, Yee 1 cm (`fdtd_refinement = 1`), float32, ten shots per
propagation, the wavelet re-estimated with the hybrid's own direct wave (script
`hyb2d.py` next to the data: it swaps the simulator of the native script and
changes nothing else). Forward check against native DeepGPR at 0.5 cm: hybrid
5.1 %, native 1 cm 6.2 %.

| run | solver | held-out PE, final (best) | training PE | εr in the sand | time | s/iteration |
|---|---|---|---|---|---|---|
| native 1 cm | DeepGPR FDTD | 0.249 (0.244) | 0.126 | 2.87–5.36 | 47 min | 7.8 |
| hybrid | DG band on the surface + Yee 1 cm | 0.219 (0.218) | 0.130 | 4.46–4.83 | 377 min | 63 |
| native 0.5 cm (cost-matched, stopped at iteration 140) | DeepGPR FDTD, TV 1.5e-4 | 0.72 at 140 (0.383 at 50) | 0.418 at 140 | 1.58–7.12 | 136 min for 140 | 56 |

* **The hybrid with its exact discrete adjoint inverts measured data.** Same
  targets and layering as the native inversion; the held-out prediction error is
  12 % lower, the gap opens in the 700 MHz stage and persists, and the model has
  none of the near-surface outliers of the native run (εr 2.87–5.36 in the native
  model, 4.46–4.83 in the hybrid one), which made the native held-out error rise
  from 0.244 to 0.249 in the last stage.
* **A finer native grid at the same cost does not do it with the same settings.**
  Native DeepGPR at 0.5 cm costs about as much per iteration as the hybrid (56
  against 63 s). With the same optimiser settings (TV weight halved, which keeps
  the regularisation of this pixel-sum TV per unit length) it leads at iteration 10
  (held-out PE 0.423 against 0.444 native 1 cm and 0.451 hybrid) and is level
  with both at iteration 50 (0.383 against 0.381 and 0.380), then grows
  thin alternating layers (about 4 cm period, 14–40 cm deep, εr 1.6–7.1) that fit
  the training traces and wreck the prediction (0.72 at iteration 140, where it
  was stopped). Whether a retuned optimiser (smaller Adam rate, stronger TV)
  would let it match the hybrid was not tested.
* **Cost: 8× the native run.** The P1 band's stability limit makes the step 2.8×
  shorter than the native one (7.74e-12 s, 4136 steps for 32 ns), and the band and
  the coupling cost more per step; a 10-shot gradient takes 5.8 s (forward 1.4 s,
  reverse 4.4 s). The surface is flat, so the hybrid has no terrain to resolve
  here; local time stepping does not pay in 2D (12.4).
* **Memory.** One 10-shot gradient peaks at 8.3 GiB (65 checkpoints of 62 MB plus
  the segment); the native script keeps all propagations of one batch in the graph
  until its backward, so the run uses batches of ten shots (the gradient is
  accumulated over all batches before the Adam step, so this changes nothing
  else).

### 12.9 Where does the hybrid pay?  Regime tests against a careful staircase FDTD

The question for a showcase model: in which situations does the hybrid beat a
conventional FDTD at equal cost?  Three sources of evidence (2026-11-09/10):

* **3D, air-launched antennas** (the terrain comparison in `/home/ll/dbpr_fwi/fwi3d_cmp.py`:
  slopes median 39°, max 68°, εr 7 over 10, antennas 5–26 cm above the ground, reference
  native 4 mm): binary staircase 2/3 cm 1.6 %, hybrid with Yee 2/3 cm 2.2 %, hybrid with
  Yee 2 cm 18 %; gradient per shot 7 s (staircase 2/3 cm) against 18 s (hybrid 2/3 cm).
* **Cost per unit volume** (RTX 4090, float32, per simulated ns and m³): Yee 2/3 cm
  0.11 s, native staircase 4 mm 0.94 s, P1 band with 2 cm elements 0.35 s.  A band
  covering 15 % of the volume plus a 2/3 cm Yee grid costs as much as a staircase at
  about 6 mm (measured on the deep model of SUBGRID.md 12: hybrid 297 ms/ns, native
  4 mm 1540 ms/ns per shot).  The hybrid can only win where a staircase needs cells
  below about 6 mm (λ/20 at 1 GHz in εr 5) for the accuracy the hybrid reaches.
* **2D tests** (`examples/hybrid/showcase2d_staircase.py`, `showcase2d_hybrid.py`, CPU,
  TM): a 1.2 m profile with slopes up to 63° (90th percentile 43°), dry ground (εr 5) or
  a 2.5 cm wet layer (εr 18, 20 mS/m) along the surface, 8 common-offset pairs (12 cm),
  400 MHz.  Relative data error (plain) against a 1.25 mm reference:

  | case | staircase, snapped antennas, 6.67 mm | staircase, off-grid antennas, 6.67 / 4 / 2 mm | hybrid, antennas in the band, Yee 6.67 mm | cost of 8 shots: staircase 6.67 / 4 / 2 mm, hybrid |
  |---|---|---|---|---|
  | dry, antennas 5–17 cm above ground | 0.10 % | (nodes) 0.10 / 0.03 / 0.01 % | – | 1.9 / 8.8 / 67 s, – |
  | wet layer, antennas 5–17 cm above | 0.14 % | (nodes) 0.14 / 0.05 / 0.01 % | – | same |
  | dry, antennas 2 cm above ground | 2.8 % | 0.29 / 0.10 / 0.02 % | 0.23 % | 1.3 / 8.5 / 74 s, 83 s |
  | wet layer, antennas 2 cm above | 5.5 % | 0.79 / 0.22 / 0.05 % | 2.4 % (layer only sampled per element) | 1.5 / 6.2 / 69 s, 85 s |

  Staircase = cell-averaged materials (area fractions); "snapped" = antennas moved to the
  nearest node (gprMax practice), "off-grid" = bilinear source / receiver weights at the
  exact positions (HybridTM2D in pure Yee mode; equal to native DeepGPR at nodes).  The
  snapped native reference itself differs from the off-grid one by 0.6 % (dry) and 1.2 %
  (wet).

**Reading.** With the two standard refinements of a staircase FDTD, cell-averaged
materials and off-grid antennas, the staircase error on steep terrain, thin wet layers
and ground-coupled antennas stays below 1 % at 6.67 mm and drops to 0.2 % at 4 mm, at a
fraction of the hybrid's cost; the large staircase errors seen before came from node
snapping of antennas near the surface and from binary materials.  In none of these
cases does the hybrid win at equal cost.  Two limits of the evidence: the 2D tests are
TM, where E is tangential to every interface, the case in which the arithmetic mean of
cell averaging is the right effective permittivity; in 3D the normal E component at a
high-contrast interface needs the harmonic mean, so averaging converges more slowly there.
12.10 measures this in 3D: with ground-coupled antennas over rough wet ground the
averaged staircase converges only at first order, and the hybrid wins.  And the hybrid's band could not follow the
wet layer (element materials are model pixels at the centroids).

### 12.10 3D regime test: ground-coupled antennas over rough, wet ground (`examples/hybrid/coupled3d.py`)

The 2D TM tests of 12.9 cannot show the staircase's weak point in 3D: the field of a
ground-coupled antenna crosses the surface in the normal direction, where the right
average permittivity is harmonic, not arithmetic. This test measures it on the local RTX
4090 (2026-11-10).

**Set-up.** 2 cm model grid, 0.58 m × 0.58 m, 8-cell absorbers; rough surface 0.14–0.27 m
deep (slopes up to 65°, 90th percentile 47°, median 30°); ground εr 15, σ 10 mS/m; 9 shots
× 8 receivers (offsets 8 and 14 cm), Ez, 400 MHz Ricker, 14 ns. Every antenna sits on an Ez
node shared by all odd refinements of 2 cm, so no solver moves it; antennas 1–3 cm above
the local surface (inside the band for the hybrid). Staircase: native DeepGPR with
cell-averaged (or binary) materials at 2 cm / k. Hybrid: P1/P2 band 0.06–0.34 m with 1 or
2 cm elements, flux 0.1, global stepping. Variants: point dipoles or finite antennas (4 cm
cos² line currents, both solvers); without or with a buried sphere (εr 5, radius 6 cm), in
which case each run also simulates the model without it and the *target response* (data
with minus data without the sphere) is scored; and the full method
(`HybridSubgridMaxwell3D`: band + coarse Yee + a Huygens box of ratio 3 around the
sphere, 0.66 m deep model).

**Point dipoles (wet, no target).** The averaged staircase converges at first order
(observed order 1.1–1.3; successive differences 3.7 / 1.4 / 0.69 / 0.43 % for 6.67 → 4 →
2.86 → 2.22 → 1.82 mm), so even 1.82 mm is about 1.7 % from the limit; binary materials
are worse (4.1 / 2.7 / 2.3 % at 4 / 2.86 / 2.22 mm). The hybrid is worse still: 12.5 %
(P1, 2 cm), 7.7 % (P2, 2 cm), 3.4 % (P1, 1 cm), 2.3 % (P2, 1 cm). A homogeneous control
(the air given the ground's material; same mesh and antennas) separates the causes: the
staircase is then exact apart from dispersion (4 mm: 0.36 %), but the hybrid keeps 10.9 /
5.3 / 3.3 / 1.1 % (P1 2 cm / P2 2 cm / P1 1 cm / P2 1 cm). The error is the projection
of a point dipole onto 1–2 cm elements (largest at the 8 cm offsets, a 30–35 ps delay for
P1 2 cm) plus a 2–3 % amplitude deficit of P1. Over 99.8 % of the energy of
ground-coupled data is the 1–6 ns direct coupling, so without a target the late window
carries no information.

**Finite antennas.** With the 4 cm line antennas the hybrid's homogeneous-medium error
drops to 4.4 % (P1, 2 cm elements), 1.2 % (P2, 2 cm), 0.44 % (P1, 1 cm, 386 ms per shot
and ns) and 0.28 % (P2, 1 cm, 1573 ms); the staircase reaches 0.37 % at 4 mm (342 ms).
Away from the surface the band is about as efficient as a staircase.

**Finite antennas with a buried sphere.** The staircase's target response converges at
first order too (successive differences 5.5 / 2.0 / 1.1 %, observed order 1.1); the target
response is 0.75 % of the background data. The two families were scored against each
other and against a first-order Richardson extrapolation of the staircase (from 2.22 and
1.82 mm; from 2.86 and 2.22 mm the numbers differ by less than 0.1 % in the background):

* *Background field* (the coupling and surface waves the antennas see): the hybrid family
  has converged (P1 against P2 with 1 cm elements 0.32 %, Yee 6.67 against 4 mm 0.05 %),
  the staircase approaches it as it is refined (2.86 mm 2.30 %, 2.22 mm 1.88 % from the
  hybrid), and the extrapolation lands 0.87 % from it. Hybrid P1 with 1 cm elements:
  0.96 % from the extrapolation at 1565 ms per shot and ns; staircase 2.22 mm: 1.60 % at
  4636 ms, 1.82 mm: 1.31 % at about 9800 ms. **The hybrid is about twice as accurate at a
  third of the cost.**
* *Target response*: set by how the Yee grid represents the sharp sphere and carries the
  scattered wave; Yee 6.67 against 4 mm under the same band changes it by 3.5 %. At equal
  Yee spacing the hybrid is about 1 % better (4 mm: 5.7 % against the staircase's 6.8 %),
  which is the surface's share.

**The full method** (0.66 m deep model, sphere at 0.48 m). Reference: first-order
Richardson extrapolation of the staircase from 2.22 and 1.82 mm (the extrapolations from
2.86/2.22 and 2.22/1.82 mm differ by 1.2 % in the target response and 0.5 % in the
background, the uncertainty of the reference):

| run | target response | background | ms per shot and simulated ns | peak GB |
|---|---|---|---|---|
| staircase 4 mm | 6.38 % | 2.93 % | 458 | 1.1 |
| staircase 2.86 mm | 4.39 % | 2.06 % | 1826 | 3.0 |
| staircase 2.22 mm | 3.36 % | 1.60 % | 5004 | 6.3 |
| staircase 1.82 mm | 2.75 % | 1.31 % | 9780 | 11.5 |
| hybrid, band P1 1 cm + Yee 6.67 mm | 8.62 % | 0.96 % | 1606 | 4.4 |
| hybrid, band P1 1 cm + Yee 4 mm | 5.19 % | 0.96 % | 2386 | 8.9 |
| full method, band P1 1 cm + Yee 6.67 mm + 2.22 mm box | 5.62 % | 0.93 % | 1918 | 6.1 |
| the same with a P2 band | 5.40 % | 0.93 % | 7717 | 9.1 |
| **full method, band P1 1 cm + Yee 4 mm + 1.33 mm box** | **3.01 %** | **0.89 %** | **2950** | **11.2** |

The staircase converges towards the full method as it is refined (target response
5.2 → 3.7 → 3.0 → 2.7 %, background 3.2 → 2.3 → 1.9 → 1.7 % from 4 to 1.82 mm), the
background agrees with the converged hybrid family to 0.3–0.4 %. The full method with a
4 mm main grid is as accurate as a 2.2–1.8 mm careful staircase on the target response
and more accurate than both on the background, at 0.59 and 0.30 of their wall time; its
memory equals the 1.82 mm staircase and is 1.8 times the 2.22 mm one (band, 4 mm main
grid, the fine model). With a 6.67 mm main grid the box helps less than a 4 mm grid
without a box: the scattered wave crosses 7 cm of coarse Yee between the band and the box
(11 cells per wavelength at 1 GHz in εr 15), and a P2 band changes nothing (0.28 / 0.62 %
from P1), so there the main grid, not the band, limits the target response.

**Air-launched antennas** (11–22 cm above the ground, in the Yee grid 2 cm above the
band; same sphere, 0.58 m model; against the staircase extrapolation from 2.86 and
2.22 mm). The staircase is accurate early: background 1.85 / 0.79 / 0.51 / 0.40 % and
target response 8.0 / 4.1 / 2.8 / 2.2 % at 6.67 / 4 / 2.86 / 2.22 mm (observed order 1.4).
With Yee 6.67 mm the hybrid's background error stays at 1.9 % whatever the band (P1/P2,
2/1 cm elements); with Yee 4 mm and a P2 band it is 0.82 % at 6622 ms per shot and ns,
sixteen times the cost of the 4 mm staircase at the same accuracy. The cause is a
coupling artefact: the antennas' near field crosses the band's top interface three Yee
cells away, and the mortar coupling does not conserve charge, so a static, dipole-shaped
field stays behind after the pulse (late-time mean 0.14 against 0.002 for the staircase,
about 10⁻³ of the peak; 0.027 with five cells of 4 mm). Antennas inside the band (the
coupled cases) show no such floor: there the hybrid's late signal matches the staircase's
within 10 %.

**Regime map from these tests.**

* The hybrid wins where the field that matters is the near-surface field of
  ground-coupled antennas over rough, high-contrast ground: about half the error of a
  2.2–1.8 mm careful staircase at a third of its cost, because the averaged staircase
  converges only at first order there (the normal field crosses the surface).
* With a refined box around a compact target and a 4 mm main grid (the full method) it
  also matches a 2.2–1.8 mm staircase on the target response, at 0.3–0.6 of the wall
  time and 1–1.8 times the memory.
* It loses with air-launched antennas (staircase error small, coupling artefact), with
  point dipoles inside the band (projection of the delta onto 1–2 cm elements; use finite
  antennas), and in 2D TM (12.9).
* Costs that decide the margin: 60 % of the band lies in the side absorbers (the band
  spans the extended width), the band's step is set by 1 cm air elements (3.4 ps), and
  local time stepping cannot help (the band dominates, 12.6–12.7).

### 12.11 Fully fused 3D step and whole-sweep CUDA Graphs (`csrc/glue3d.cu`, `fused3d.py`, `sweep3d.py`)

After 11.4 and 12.5 the Yee, DG and LTS operators were fused kernels, but the
operations between them were not. These are the mortar trace `E*` and the
interface input `X_h`, the ghost planes, the interface penalty rows, source
injection, receiver sampling, the slab traces and ghost planes of the LTS
recursion, the scatters of the fine-plane UPML folds and the split-penalty
damping arithmetic. They ran as small PyTorch kernels. The profile of the
step-(a) model (12.6) showed, per step:

| | ATen kernel calls | share of GPU time |
|---|---|---|
| global step, forward | 98 | 23 % |
| global step, gradient | 475 | 32 % |
| LTS, gradient | 983 | 33 % (+10 % memcpy) |

The hand-written transposes also cloned, zeroed and added full Yee and DG arrays.
The reverse step ran eagerly. Forward graph replay was limited to states up to
32 MB because each replay copied the state twice.

**What was changed.**
* `csrc/glue3d.cu`, `glue3d.py`: each glue operation and its transpose is one or
  two launches. The mortar matrices are stored in CSR form both by output and by
  input, so the kernels gather. Atomics are used only where entries accumulate
  into one coefficient (face penalty `γ`, source `cb`) or one node (sources and
  receivers).
* `csrc/yee3d.cu`: the Yee transpose can accumulate in place, both the
  coefficient cotangents (into persistent per-shot accumulators) and the
  cotangent of the differentiated field. The kernel wrappers in `cuda3d.py` take
  output buffers.
* `csrc/lts3d.cu`: the fold kernels scatter their results directly into the
  destination fields. Their transpose gathers the output cotangents and zeroes
  them in place.
* The split-penalty damping runs as epilogues of the operator kernel (mode 2
  with `ch` for H and `dt/ε` for E).
* `fused3d.py`: `step_into` / `vjp_into` (global step, pure Yee) and
  `lts_step_into` / `lts_vjp_into` write the new state, or the cotangent of the
  old state, into given buffers. The transpose consumes its input cotangent in
  place instead of cloning it. `LTSPre` holds the coefficient-derived arrays
  (`dt/ε` on the slabs and in the band, gathered fine-plane UPML factors).
* `sweep3d.py`: CUDA Graphs of whole checkpointed sweeps on fixed buffers.
  - The segment length `k` is even.
  - `k` forward graphs `F_j: U_j → U_(j+1)` sample the receivers. They serve
    both the forward pass and the recomputation of each segment.
  - `k` reverse graphs `R_j: (U_j, U_(j+1), L_((j+1) mod 2)) → L_(j mod 2)`
    accumulate the coefficient cotangents.
  - Forward-only runs use two buffers and two graphs.
  - The only copies are one checkpoint per segment.
  - Buffers belong to the solver and are shared by its propagators. Graphs
    belong to each propagator and are kept across calls; coefficients are
    refreshed by copy.
  - `fit_k` shortens `k` when the buffers would not fit. An out-of-memory
    error during capture falls back to the eager fused steps.
* Checkpoints stay on the device whenever they fit, instead of being offloaded
  above a fixed 4 GB. `DEEPGPR_HYBRID_CKPT_GPU_MB` still forces the old rule.
* `HybridPropagator.release()` frees the graphs and buffers.
* `DEEPGPR_HYBRID_DISABLE_GRAPHS=1` runs the same fused steps eagerly.
  `DEEPGPR_HYBRID_AUTOGRAD_VJP=1` differentiates the PyTorch-glue step with
  autograd. The 2D solver (12.12) and the subgridded step (`docs/SUBGRID.md`
  13) were converted the same way afterwards.

**Numerical validation** (float64, small rugged-surface model).
* Fused step against the PyTorch-glue step: at most 4e-16. Configurations:
  P1 and P2, `r = 3` on the terrain mesh, ABC sides without penalty, pure Yee,
  and LTS (central, split penalty with `k = 1` and `k = 5`, `r = 3`).
* Fused transpose against reverse-mode differentiation of the reference step:
  at most 3e-15 for every state, coefficient and source cotangent.
* Propagator data and gradients (graph sweep against eager fused steps against
  the autograd sweep): at most 1e-11. This also holds for odd `checkpoint_every`
  and for calls repeated with other models.

On the step-(a) model at full length (float32, 12 ns) the outputs agree with
the code before this change to round-off:

| | difference |
|---|---|
| data | 1.2e-5 |
| ε gradient, L2 | 1.3e-5 |
| σ gradient | 2e-6 |
| pure Yee | bitwise identical |

**Speed.** RTX 4090 (server, idle card), float32, one shot, minimum of three runs,
ms per simulated ns. Model A is the step-(a) FWI model (12.6) at its full 12 ns.
Model B is `wet_coupled_line_amr` of 12.10 (P1 band with 1 cm elements, Yee 4 mm,
14 ns), forward only. "Pure FDTD" is the same solver without the band, on the same
Yee grid, with the same kernels and adjoint machinery.

| model, path | forward before | forward after | gradient before | gradient after |
|---|---|---|---|---|
| A, hybrid, global step | 204.5 | **127.5** (1.60×) | 861.6 | **555.1** (1.55×) |
| A, hybrid, LTS (p = 2) | 215.7 | **142.6** (1.51×) | 1273.1 | **682.3** (1.87×) |
| A, pure FDTD 2/3 cm | 41.5 | **37.8** (1.10×) | 254.2 | **172.6** (1.47×) |
| B, hybrid, global step | 1908.5 | **1739.4** (1.10×) | – | – |
| B, pure FDTD 4 mm | 306.5 | **283.2** (1.08×) | – | – |

Notes on the table:
* "Gradient before" of A, global step, is measured with the checkpoints on the
  device (`DEEPGPR_HYBRID_CKPT_GPU_MB=16000`). With the old default its 39
  checkpoints (4.9 GB) went to host memory and the gradient took 1045.5 ms/ns,
  1.88× the new default.
* The pure FDTD gradient also improved: it used autograd before and now uses
  the hand-written transpose and graphs.
* Model B's gradient over 14 ns does not fit on a 24 GB card in either version
  (state 0.69 GB, about 2√nt ≈ 128 states with √nt checkpointing).

Hybrid against pure FDTD on the same grid, after the change:

| | forward | gradient | before the change |
|---|---|---|---|
| A, global step | 3.4× slower | 3.2× slower | 4.9× / 3.4× |
| A, LTS | 3.8× | 4.0× | 5.2× / 5.0× |
| B (DG-dominated) | 6.1× | – | 6.2× |

**Where the time goes now** (A, eager profile, GPU time).
* Global step: forward 98.6 % in the Yee/DG kernels (11 launches per step);
  gradient 96 % (42 hybrid kernels and 29 ATen calls per step, the latter
  accumulator additions).
* LTS: forward 95 %, gradient 89 %. The remaining 10 % are DG-sized elementwise
  operations of the recursion and damping bookkeeping.
* The gap to pure FDTD is no longer glue:
  - the DG band (P1 operator and updates: about one third of the forward);
  - the band-limited `dt` (7.7 ps against 10.6 ps, 1.38× more steps);
  - Yee kernels about 1.5× slower per node in hybrid mode (side UPML through
    compact index sets, masks), including the 23 % of Yee rows inside the band,
    which are inactive but still swept.

  Skipping those rows and 32-bit index arithmetic in the Yee kernels are the
  next kernel-level steps (not done).

**Follow-up: accumulating transposes.** The remaining DG-sized elementwise work of
the LTS gradient (summing per-call cotangents into accumulators) was folded into
the kernels: the DG operator transpose adds its face-coefficient cotangents into
running sums in the kernel layouts (converted once at the end of the sweep), the
`dt/ε = 2 cb / (1 + ca)` chain rule is applied once to running sums instead of at
every step, the slab and fold transposes accumulate in place, and the damping
transpose uses one combine kernel. Model A, same card and protocol:

| | forward | gradient |
|---|---|---|
| global step | 127.5 → 127.4 | 555.1 → **547.9** (1.3 %) |
| LTS (p = 2) | 142.6 → 141.1 | 682.3 → **651.1** (4.6 %) |
| pure FDTD 2/3 cm | 37.8 → 37.8 | 172.6 → 172.9 |

### 12.12 The 2D solver on fused kernels and whole-sweep CUDA Graphs (`fused2d.py`)

The 2D (TM) hybrid had fused Yee and DG kernels (11.4 history, `cuda2d.py`) but
the same PyTorch glue as the 3D step had before 12.11, a hand-written transpose
(`vjp2d.py`) only for the global step, and autograd for the LTS step.

**What was changed.**
* `csrc/glue3d.cu` ops 14–17: the 2D mortar trace (Ez rows → DG `E*`) and the
  ghost rows (DG → Yee Hy ghosts) with their transposes; the point kernels take
  the number of DG field components (1 for the 2D Ez). `glue3d.py`:
  `CouplingGlue2D`.
* `csrc/yee2d.cu`, `csrc/dg2d.cu`, the 2D slab kernels of `csrc/lts3d.cu`: the
  transposes accumulate in place (cotangents of the differentiated fields,
  coefficient cotangents, `cb0`, slab corrections); the wrappers in `cuda2d.py`
  take output buffers.
* `fused2d.py`: `step_into` / `vjp_into` (global step, pure Yee) and
  `lts_step_into` / `lts_vjp_into`. The latter is a new hand-written transpose of
  the 2D LTS step (recursion recomputed with its intermediates stored, then the
  reverse recursion); before, the 2D LTS gradient used autograd.
* `sweep3d.py` serves both dimensions (`module(st)`); the 2D propagator, the 2D
  `solver.run` and `functional.step`/`sample` (without gradients) use the fused
  steps.

**Numerical validation.**
* float64, small rugged models, 1 and 2 shots: fused step against the
  PyTorch-glue step at most 2e-16; fused transpose against reverse-mode
  differentiation of the reference step at most 4e-16 (P1, P2, pure Yee, LTS
  with p = 2, 3 and p = 4 at P2).
* Propagator data and gradients (graph sweep, eager fused steps, autograd
  sweep; 600 steps, ε, σ and wavelet): at most 5e-15.
* The rugged configuration below (terrain-following mesh, central fluxes, Yee
  2.5/3 cm) in float64 over 2.5 ns: global step (1132 steps) and LTS (p = 9)
  gradients within 7e-15 of the autograd sweep.
* float32 at full length against the code before the change: pure Yee bitwise
  identical; hybrid data 1e-6–7e-6; ε gradient 1e-5 (sand-pit profile) to
  2e-4–5e-4 (rugged model, 10^4 steps of a DG-dominated band; round-off, see
  the float64 check).

**Speed.** RTX 4090 (server, idle card), float32, minimum of three runs after a
warm-up, ms per simulated ns of one propagation
(`examples/hybrid/glue_benchmark2d.py`).
* F: the sand-pit field profile of 12.8 (run `H_l1_v2`: 1 cm grid 172 × 973,
  band 1 cm above to 5 cm below the sand, flux 0.1, r = 1), one batch of 10
  shots, 32 ns. F, FDTD: the same grid without the band.
* R3 / R1: the rugged model of `lts_benchmark2d.py` (terrain-following P1 band
  0.15–0.95 m, h = 1.25 cm, central fluxes; Yee 2.5/3 cm or 2.5 cm), 14 shots,
  24 ns. R3, FDTD: pure Yee on the 2.5/3 cm grid.

| model, path | forward before | forward after | gradient before | gradient after |
|---|---|---|---|---|
| F, hybrid, global step | 84.2 | **31.5** (2.67×) | 282.0 | **127.1** (2.22×) |
| F, pure FDTD 1 cm | 13.9 | **8.2** (1.70×) | 42.8 | **31.8** (1.35×) |
| R3, hybrid, global step | 312.4 | **280.7** (1.11×) | 1702.8 | **1387.2** (1.23×) |
| R3, hybrid, LTS (p = 9) | 312.3 | **282.5** (1.11×) | 1775.9 | **1580.5** (1.12×) |
| R3, pure FDTD 2.5/3 cm | 16.3 | **6.1** (2.68×) | 45.7 | **25.2** (1.82×) |
| R1, hybrid, global step | 295.8 | **227.4** (1.30×) | 1490.2 | **1204.7** (1.24×) |
| R1, hybrid, LTS (p = 25) | 269.5 | **254.0** (1.06×) | 1535.3 | **1439.1** (1.07×) |

For the field FWI of 12.8 one 10-shot gradient went from 9.0 s to 4.1 s and a
10-shot forward from 2.7 s to 1.0 s. Peak memory is unchanged except where the
checkpoints now stay on the device (R3 global gradient 9.6 → 17.8 GB, R1 global
5.0 → 9.0 GB; the old rule offloaded them above 4 GB).

Hybrid against pure FDTD on the same Yee grid, after the change: F 3.8×
(forward) and 4.0× (gradient), before 6.1× and 6.6×. R3 46× and 55× (global),
because its 0.8 m thick band of 0.55 M DG unknowns steps at 2.2 ps against the
Yee grid's 16 ps; the operator kernels of that band are the time, so the gains
on R are small, and the LTS does not reduce the band's work (12.2). On this
model the staircase FDTD at 0.5 cm is more accurate than the hybrid at a fraction
of its cost (8.7, 12.9).

### 12.13 Single-pass Yee transposes (`csrc/yee3d.cu`, `csrc/yee2d.cu`)

After 12.11, 12.12 and the subgridded step (`docs/SUBGRID.md` 13) the gradients
of the pure-Yee and subgridded models were dominated by the Yee kernels. In the
sand-pit benchmark with one whole-region box (SUBGRID 13) the Yee transposes took
49 % of the gradient's GPU time and the Yee updates (forward pass and
recomputation) 42 %.

**The old 3D transpose** ran in two passes. Pass 1, one thread per shot and node,
formed the cotangents of the old fields, the CPML/UPML auxiliaries and the
coefficients and wrote the cotangent of each of the six differences into a
temporary array of `6 S N` values; pass 2 gathered them onto the differentiated
field. That is 48 B of extra traffic per shot and node (24 B written and read
back) on top of about 55 B (H) and 85 B (E) that the transpose has to move. The
`ca`/`cb` cotangents were read-modified-written per shot (16 B per shot and
node), and with the term tables in `__constant__` memory the small per-term
arrays sat in local memory (64–112 B stack frames).

**What was changed.**
* 3D (`yee_bwd`): one kernel, one thread per node looping over the shots. The
  thread computes everything at its own node as before; the transposed
  differences are gathered by re-evaluating the difference cotangents of the
  three upstream (H) or downstream (E) neighbours instead of storing them. Masks,
  coefficients, slab positions and the neighbours' coefficients are read once for
  all shots; per shot the plain loads are issued before any branch; the term
  tables are compile-time functions (no stack frame); the side UPML is a
  compile-time switch (fewer registers on grids without it); 128-thread blocks,
  32-bit node arithmetic.
* The `ca`/`cb` cotangents and those of the material UPML factors are summed over
  the shots in registers (`accf` bit 2). The fused sweeps (`fused3d.py`,
  `subgrid/fused.py`) and the autograd function keep accumulators of `N` values
  instead of `S × N`; per-shot output remains available (`vjp3d.py`).
* 2D: the transposes were already single-pass. The E transpose now splits the
  shots of a node over up to 8 threads ("lanes") of a block of 32 nodes; each lane
  reads the coefficients of the node and its neighbours once and sums the
  `ca`/`cb` cotangents of its shots in registers, and the lanes are summed in
  shared memory in a fixed order (`fused2d.py` accumulators of `N` values). One
  thread per node with all shots in a loop was slower on small grids: the rugged
  model R1 has 8.5 × 10^3 Yee nodes, i.e. 33 blocks for 128 SMs (gradient 0.6 %
  slower). The H transpose stays one thread per shot and node: it has no
  coefficient cotangents to sum and its coefficients stay in L2 anyway (the
  node-major version was 17 % slower on the sand-pit profile).
* The transposes read `g`, `gpsi_out` and `gaux_out` at neighbouring nodes, so
  these must not share memory with the outputs. No caller does this (the subgrid
  cotangent chains alternate between two buffers for the admissible odd ratios);
  `YeeCudaOps` copies such an input if it happens.

**Numerical validation** (float64).
* Kernels against reverse-mode differentiation of the PyTorch Yee updates
  (`functional3d._yee_curl` / `_yee_upml`, the 2D formulas of
  `functional._step_reference`; no CUDA): 1, 2, 3 and 9 shots (3D) and 1, 2, 3,
  9, 10 and 17 shots (2D, more shots than lanes), with and without side UPML, CPML
  on one or three axes, inactive nodes; every output mode (per shot, summed over
  the shots, accumulated, outputs aliasing the inputs). At most 2.9e-15 (3D) and
  1.0e-15 (2D).
* The step and propagator checks of 12.11, 12.12 and SUBGRID 13 give the same
  bounds as before: fused step transposes ≤ 2.0e-15 (3D) and ≤ 4.1e-16 (2D),
  subgridded step ≤ 1.8e-15, propagator gradients ≤ 5.2e-15, DWR indicator
  3.7e-14.
* With the PyTorch Yee updates in the reference step instead of the CUDA autograd
  functions, the 2D check stays at 4e-16. The 3D hybrid configurations give 3e-14
  (step) and 5e-12 (`fd_cb`), the same numbers as the previous code: round-off of
  the two UPML implementations on the random test states, whose UPML values reach
  6e9.
* float32, full-length benchmark runs against the previous code: data bitwise
  identical (the update kernels are unchanged); ε and σ gradients within 1e-9 to
  5e-6 (relative L2). On pure FDTD, whose gradients are bitwise reproducible from
  run to run, the 2e-7 to 2e-6 come from the changed summation order of the
  transposes; on the 3D models with boxes or a DG band, two runs of one code
  differ by the same amounts (12.14).

**Kernel times** (server RTX 4090, idle card, ms per call, random data;
`H`/`E` update = the forward kernels, unchanged):

| grid | H transpose | E transpose | H / E update |
|---|---|---|---|
| SP box 82 × 151 × 91, 9 shots | 1.392 → **0.717** (1.94×) | 1.939 → **1.081** (1.79×) | 0.496 / 0.527 |
| SP main grid 52 × 75 × 55, 9 shots | 0.293 → **0.175** (1.67×) | 0.394 → **0.261** (1.51×) | 0.139 / 0.145 |
| A Yee grid 133 × 145 × 145, 1 shot, side UPML | 0.531 → **0.320** (1.66×) | 0.851 → **0.682** (1.25×) | 0.286 / 0.287 |
| 2D F 213 × 1014, 10 shots (ms per step, profile; hybrid / pure FDTD run) | 0.098 (unchanged) | 0.188 → **0.163 / 0.155** (1.15× / 1.22×) | 0.085 / 0.078 |

On the SP box the transposes now move about 780 GB/s, against about 1 TB/s of the
update kernels. The hybrid E transpose gains least: with one shot nothing is
shared between shots, and its side-UPML variant needs 122 registers (one third of
the threads of the other variants).

**Speed** (as in 12.11, 12.12 and SUBGRID 13: server RTX 4090, idle card, float32,
minimum of three runs, ms per simulated ns; before and after measured on the same
card in one session). Forward times are unchanged.

| model | gradient before | gradient after |
|---|---|---|
| SP, whole-region 0.75 cm box (9 shots) | 520.6 | **416.7** (1.25×) |
| SP, oracle boxes (9 shots) | 359.6 | **296.2** (1.21×) |
| SP, 2.25 cm main grid alone (9 shots) | 31.1 | **25.2** (1.23×) |
| SP, uniform 0.75 cm FDTD (1 shot) | 171.0 | **132.6** (1.29×) |
| T, hybrid + box (1 shot) | 563.1 | **511.9** (1.10×) |
| T, hybrid alone (1 shot) | 319.9 | **316.5** (1.01×) |
| A, hybrid, global step | 552.4 | **507.1** (1.09×) |
| A, hybrid, LTS (p = 2) | 658.5 | **627.0** (1.05×) |
| A, pure FDTD 2/3 cm | 174.2 | **136.7** (1.27×) |
| F (2D), hybrid | 128.0 | **124.2** (1.03×) |
| F (2D), pure FDTD 1 cm | 31.8 | **29.0** (1.10×) |
| R3 (2D), hybrid, global step | 1379.1 | **1371.7** (1.01×) |
| R3 (2D), hybrid, LTS (p = 9) | 1578.8 | 1579.8 (1.00×) |
| R3 (2D), pure FDTD 2.5/3 cm | 25.3 | **22.9** (1.10×) |
| R1 (2D), hybrid, global step | 1194.8 | 1197.7 (1.00×) |
| R1 (2D), hybrid, LTS (p = 25) | 1437.8 | 1437.2 (1.00×) |

The gain follows the share of the Yee transposes: largest for pure Yee and the
subgridded models (1.21–1.29×), smaller for the 3D hybrid (the band and the
side-UPML variant), none on the DG-bound 2D rugged model. Since pure FDTD gained
more than the hybrid, the hybrid's relative cost grew: model A, gradient, hybrid
against pure FDTD on the same grid 3.71× (global step) and 4.59× (LTS), before
3.17× and 3.78×; 2D sand-pit profile 4.28× (before 4.03×). The subgrid ratios are
in `docs/SUBGRID.md` 13.

**Where the time goes now** (SP whole-region box, gradient, eager profile on the
server, 16.5 ms per step against 21.1 before): Yee updates of the forward pass
and the recomputations 54 %, Yee transposes 35 % (were 49 %), correction,
precursor and point kernels 11 %. On model A the side-UPML E transpose is the
largest single kernel (18 %). Remaining kernel-level levers, not done: the hybrid
Yee kernels read full-grid int32 UPML position maps (12 B per node), and the
inactive Yee rows inside the band are still swept (12.11). The term tables of the
update kernels: 12.14.

### 12.14 Compile-time term tables in the Yee update kernel (`csrc/yee3d.cu`)

The 3D Yee update kernel `yee_fwd` (forward pass and recomputations of the hybrid
main grid, of the pure-Yee solver and of the subgrid boxes) read the term tables
(axis, differentiated field and sign of each curl term) from `__constant__`
memory. Indexed with these run-time values, the small per-term arrays (node
coordinates, field pointers, CPML profiles) were placed in local memory: a 64 B
stack frame per thread. The kernel now uses the compile-time functions of the
transpose (12.13) in unrolled term loops, so every index is a constant: no stack
frame, 40–44 registers instead of 36–42, no spills. The arithmetic is unchanged.
The 2D kernels have no term tables (they are written per component).

**Numerical validation.**
* float64, one fused step of 15 configurations (hybrid global step with DG
  orders 1 and 2, pure Yee, LTS p = 3 and p = 2 with refinement 3, each with 1 and
  3 shots; subgridded r = 3 with one and two boxes, r = 5, hybrid with a box at
  time ratios 3 and 1): the outputs are bitwise identical to those of the
  previous kernel (374 arrays, 2.9 × 10^7 values). Against the PyTorch Yee
  updates they differ by 1.7e-16 (pure Yee) to 3.8e-14 (hybrid, the UPML
  round-off of 12.13), the same numbers as with the previous kernel.
* float32, full-length benchmark runs (table below): data bitwise identical,
  except on the hybrid model B, whose forward pass is not bitwise reproducible
  (two runs of the previous code differ by 6.8e-7, previous against new code by
  5.8e-7 and 6.5e-7, relative L2). Gradients bitwise identical on pure FDTD (SP
  coarse, SP fine, A FDTD). On the models with boxes or a DG band they differ by
  4e-10 to 4.6e-6 (relative L2), and two runs of the previous code differ by the
  same amounts (SP static 1.8e-6, SP oracle 2.1e-6, T static 4.6e-6, T coarse
  1.8e-9, A global 4.4e-10, A LTS 1.1e-9). The subgrid transposes (`hsg3d.cu`) and the source
  injection, coupling and point-source transpose kernels (`glue3d.cu`) accumulate
  with atomics where several entries share a node; these results are reproducible
  only to float32 round-off.

**Kernel times** (server RTX 4090, idle card, ms per call, random data):

| grid | H update | E update |
|---|---|---|
| SP box 82 × 151 × 91, 9 shots | 0.497 → **0.479** (1.04×) | 0.515 → **0.504** (1.02×) |
| SP main grid 52 × 75 × 55, 9 shots | 0.131 → **0.109** (1.20×) | 0.136 → **0.112** (1.22×) |
| A Yee grid 133 × 145 × 145, 1 shot, side UPML | 0.286 → **0.276** (1.04×) | 0.287 → **0.279** (1.03×) |

The kernels of the two large grids run close to the memory bandwidth of the card
(12.13), where fewer instructions help little; the small main grid (1.9 × 10^6
shot-nodes) gains about 20 %.

**Speed** (protocol of 12.13; before and after alternate per model on the same
card; ms per simulated ns):

| model | forward before → after | gradient before → after |
|---|---|---|
| SP, whole-region 0.75 cm box (9 shots) | 94.2 → **90.4** (1.04×) | 411.6 → **402.8** (1.02×) |
| SP, oracle boxes (9 shots) | 67.9 → **59.3** (1.14×) | 296.1 → **275.3** (1.08×) |
| SP, 2.25 cm main grid alone (9 shots) | 7.0 → **5.6** (1.26×) | 25.3 → **22.5** (1.12×) |
| SP, uniform 0.75 cm FDTD (1 shot) | 36.8 → **35.0** (1.05×) | 132.4 → **128.9** (1.03×) |
| T, hybrid + box (1 shot) | 124.2 → **112.7** (1.10×) | 512.3 → **482.8** (1.06×) |
| T, hybrid alone (1 shot) | 76.3 → **74.5** (1.02×) | 316.8 → **313.0** (1.01×) |
| A, hybrid, global step | 127.9 → **124.6** (1.03×) | 506.8 → **500.0** (1.01×) |
| A, hybrid, LTS (p = 2) | 142.3 → **140.5** (1.01×) | 627.2 → **623.6** (1.01×) |
| A, pure FDTD 2/3 cm | 38.0 → **36.3** (1.05×) | 136.3 → **132.7** (1.03×) |
| B, hybrid, global step (forward only, 14 ns) | 1748.3 → **1735.2** (1.01×) | — |
| B, pure FDTD 4 mm (forward only) | 286.0 → **273.6** (1.05×) | — |

The gains are largest where small grids carry the time (the 2.25 cm main grid
with 9 shots, the three oracle boxes, the one-shot terrain box of 0.8 × 10^6
nodes) and 1–5 % elsewhere; the DG-bound hybrid models barely move. Pure FDTD
again gained a little more than the hybrid: model A, hybrid against pure FDTD on
the same grid, forward 3.43× (3.37× before) and gradient 3.77× (3.72×), LTS
gradient 4.70× (4.60×); model B forward 6.34× (6.11×). The subgrid ratios are in
`docs/SUBGRID.md` 13.

**Where the time goes now** (eager profiles on the server, graphs off). SP
whole-region box: forward 3.94 → 3.79 ms per step, of which Yee updates 84 % (86 %
before); gradient 16.47 → 16.17 ms per step, Yee updates 52 % (54 %), Yee
transposes 35 %, correction, precursor and point kernels 13 %. Model A, global
step: forward 1.00 → 0.98 and gradient 3.97 → 3.93 ms per step; in the gradient
Yee updates 29 %, Yee transposes 26 %, DG and coupling kernels 45 % (the side-UPML
E transpose and the DG `terms_fwd` kernel 18 % each).

### 12.15 Compile-time term tables in the DG and LTS slab kernels (`csrc/dgtd3d.cu`, `csrc/lts3d.cu`)

The DG operator kernels `terms_fwd` and `terms_bwd1` and the four LTS slab kernels
(`slab_q_fwd/bwd`, `slab_r_fwd/bwd`) read the term tables from `__constant__` memory.
Indexed with these run-time values, the face normal, face jump, lifted cotangent and
element Jacobian `R` of the DG kernels (64 B) and the slab coordinates, extents, strides,
steps and field pointers of the slab kernels (96–120 B) were placed in local memory.
They now use the compile-time functions of 12.13/12.14 in unrolled term loops, with one
measured exception: the volume part of `terms_fwd` keeps the run-time table (below).

| kernel (float) | stack frame before → after | registers before → after |
|---|---|---|
| `terms_fwd` P1 / P2 | 64 / 64 → 40 / 40 B | 93 → 96 / 197 → 192 |
| `terms_bwd1` P1 / P2 | 64 / 840 → 0 / 680 B | 128 → 128 / 255 → 255 (P2 spills 776 → 680 B) |
| `slab_q_fwd` / `slab_q_bwd` | 112 / 120 → 0 / 0 B | 30 → 32 / 46 → 101 |
| `slab_r_fwd` / `slab_r_bwd` | 112 / 96 → 0 / 0 B | 28 → 26 / 28 → 29 |

Unchanged: `terms_bwd2` (96 / 240 B from the run-time face-node map `fmask`, not a term
table) and `fold_bwd` / `ltsupd_bwd` (520 B: the UPML fold trajectory kept per thread).

**`terms_fwd`: the compiler decides.** With compile-time tables throughout, `terms_fwd`
became slower although its code shrank (P1: 2272 → 1368 instructions, no local loads),
and the hybrid global-step models were 1–3 % slower end to end (server: model A forward
0.98×, model B 0.97×). The server builds the library with nvcc 11.8, the local
workstation with nvcc 12.4, and the two compilers unroll and schedule the kernel
differently. Variants, micro-benchmark on the band of model A (138 240 elements;
`terms_fwd` as called in the step, ms per call, mean of the E- and H-side calls; the
local card also drives the display and scatters by about ±5 %):

| `terms_fwd` variant | nvcc 11.8 (server) P1 / P2 | nvcc 12.4 (local) P1 / P2 |
|---|---|---|
| previous: run-time tables | 0.106 / 0.304 | 0.095 / 0.331 |
| compile-time tables | 0.116 / 0.320 | 0.116 / 0.338 |
| compile-time, face-node loop unrolled | 0.117 / 0.337 | 0.111 / 0.283 |
| compile-time, face-node loop rolled | 0.116 / 0.326 | 0.086 / 0.345 |
| compile-time, rolled for P1, unrolled for P2 | 0.116 / 0.335 | 0.084 / 0.285 |
| face part run-time, volume part compile-time | 0.119 / 0.343 | 0.114 / 0.366 |
| **face part compile-time, volume part run-time (adopted)** | **0.094 / 0.288** | **0.095 / 0.322** |

Every variant with the volume part on compile-time tables is slower than the previous
kernel with nvcc 11.8; the loop variants that win with nvcc 12.4 lose with nvcc 11.8. The
adopted variant is the only one not slower with either compiler. The cause was not
traced further (the counters of Nsight Compute are not accessible on these machines).

**Numerical validation** (float64). For the first time the reference step uses the
PyTorch Yee, DG and LTS slab operators (all hybrid CUDA operators detached), so the DG
and slab transposes are checked independently of their kernels.
* One fused step and its fused transpose for 16 configurations (global step P1 and P2;
  LTS p = 3 with P1 and P2, p = 2 with flux 1.0 and with refinement 3; flux variants;
  1 and 3 shots; subgridded hybrid with time ratios 3 and 1): step outputs bitwise
  identical to the previous kernels (1002 arrays, 9.3 × 10^7 values). The transposes
  differ only in the 39 arrays that pass through the atomics of `glue3d.cu` (interface
  penalty, point sources, subgrid), by at most 2.7e-16; two runs of one code differ by
  3e-17 to 3.6e-16.
* Against the PyTorch reference, before and after alike: steps ≤ 3.8e-14, field
  cotangents ≤ 1e-14, coefficient cotangents up to 2.6e-11 (`dg_ca`, `dg_cb`, `fd_cb`).
  The latter come from the random test states: their UPML auxiliaries drive the fields
  to 2e9–7e10 against coefficient cotangents of order 1e3. With the UPML auxiliaries of
  the random state set to zero, all coefficient cotangents agree to ≤ 5.5e-16 (global
  step and LTS p = 2). This also explains the 5e-12 of `fd_cb` in 12.13.
* float32, full-length benchmark runs: data bitwise identical (model B is not bitwise
  reproducible, 12.14); gradients within the run-to-run differences.

**Kernel times** (server, nvcc 11.8, eager profile, ms per simulated step):

| kernel | A global step P1 (forward / gradient) | A LTS P1 (forward / gradient) | A global step P2 (gradient) |
|---|---|---|---|
| `terms_fwd` | 0.279 → 0.266 / 0.700 → 0.665 | 0.758 → 0.716 / 2.286 → 2.157 | 1.462 → 1.410 |
| `terms_bwd1` | — / 0.259 → 0.240 (1.08×) | — / 0.711 → 0.642 (1.11×) | 0.884 → 0.711 (1.24×) |
| `slab_q_fwd` / `slab_r_fwd` | — | 0.098 → 0.062 / 0.115 → 0.079 (gradient) | — |
| `slab_q_bwd` / `slab_r_bwd` | — | 0.050 → 0.034 / 0.047 → 0.033 | — |

With nvcc 12.4 (local profile of A LTS): slab kernels 1.42–1.70×, `terms_bwd1` 1.07×,
`terms_fwd` 1.02–1.03×; forward 1.04× and gradient 1.03× per step.

**Speed** (server, nvcc 11.8, protocol of 12.13, before and after alternating per model on
one idle card; ms per simulated ns):

| model | forward before → after | gradient before → after |
|---|---|---|
| A, hybrid, global step | 124.6 → **122.5** (1.02×) | 500.1 → **493.4** (1.01×) |
| A, hybrid, LTS (p = 2) | 140.4 → **133.6** (1.05×) | 623.6 → **595.7** (1.05×) |
| T, hybrid + box | 112.7 → **111.1** (1.01×) | 481.8 → **477.3** (1.01×) |
| T, hybrid alone | 74.5 → **73.1** (1.02×) | 312.9 → **308.6** (1.01×) |
| B, hybrid (forward only, 14 ns) | 1733.4 → **1713.3** (1.01×) | — |

The LTS step gains most (slab kernels and the six `terms_fwd`/`terms_bwd1` calls per
macro step); P2 gains more than P1 in the transpose (gradient per step 1.03× on model A
with P2). Hybrid against pure FDTD on model A (pure FDTD unchanged, 12.14): forward
3.37× (3.43× before), gradient 3.72× (3.77×), LTS gradient 4.49× (4.70×); model B forward
6.26× (6.34×). No term table in `__constant__` memory is left in the hybrid kernels other
than the one of the `terms_fwd` volume part.
