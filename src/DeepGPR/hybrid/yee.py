"""PyTorch Yee-FDTD sub-solver for the 2D Ez-TM mode, with DGTD interface rows.

This module reproduces DeepGPR's native second-order 2D update
(``update_h_cpu``/``update_e_cpu``/``cpml_*_cpu`` in ``lib/deepgpr_cpu.c``)
operation by operation, on the same extended grid, with the same material
coefficients and the same CPML coefficients (built by
:func:`DeepGPR.solver.pml.build_pml_coeffs`). With no DGTD band it is the
plain DeepGPR scheme; ``tests/test_hybrid_dgtd.py`` checks it against
:func:`DeepGPR.compute`.

Grid (extended frame, field arrays ``(nshot, NX, NY)`` with ``NX = nx + 1``):
``Ez[i, j]`` at ``(i dx, j dy)``, ``Hx[i, j]`` at ``(i dx, (j + 1/2) dy)``,
``Hy[i, j]`` at ``((i + 1/2) dx, j dy)``. ``Ez`` on the outer rows/columns is
PEC.

A DGTD band between the Ez rows ``i_top`` and ``i_bottom`` removes the rows
strictly between them from the Yee update. Rows ``i_top``/``i_bottom`` are
*interface rows*: their Ez nodes carry half the usual weight in the discrete
energy, and their x-derivative of Hy becomes the one-sided half-cell
difference ``(Hy[i_b] - H_b) / (dx / 2)`` with the tangential trace ``H_b``
supplied by the DGTD side. This is a summation-by-parts closure; it is
implemented by writing the ghost value ``Hy_ghost = 2 H_b - Hy_inner`` into the
Hy row inside the band, so the regular update formula applies unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from ..config.constants import PEC_CONDUCTIVITY_THRESHOLD, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY


class YeeTM2D:
    """Second-order Yee update of ``(Ez, Hx, Hy)`` with CPML and optional band.

    Args:
        eps_r_pad, sigma_pad, mu_r_pad: Materials on the field grid
            ``(NX, NY)`` (extended model plus the one-cell Yee halo).
        dx, dy: Grid spacing [m].
        dt: Time step [s].
        pml: Six CPML thicknesses ``(x0, xm, y0, ym, z0, zm)``.
        pml_coefficients: The 12 coefficient tensors returned by
            :func:`DeepGPR.solver.pml.build_pml_coeffs` (``[6:]``).
        nshot: Number of simultaneous shots.
        band: ``None`` or ``(i_top, i_bottom)``; ``i_top`` may be ``None``
            when the band reaches the top PEC row and ``i_bottom`` ``None``
            when it reaches the bottom one.
    """

    def __init__(
        self,
        eps_r_pad: torch.Tensor,
        sigma_pad: torch.Tensor,
        mu_r_pad: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
        pml: Sequence[int],
        pml_coefficients: Sequence[torch.Tensor],
        nshot: int,
        band: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ):
        self.dx, self.dy, self.dt = float(dx), float(dy), float(dt)
        self.dtype, self.device = eps_r_pad.dtype, eps_r_pad.device
        self.NX, self.NY = eps_r_pad.shape
        self.nshot = int(nshot)
        self.pml = [int(p) for p in pml]
        self.band = band
        self._build_coefficients(eps_r_pad, sigma_pad, mu_r_pad)
        self._build_pml(pml_coefficients)
        self._build_row_ranges()
        shape = (self.nshot, self.NX, self.NY)
        self.Ez = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.Hx = torch.zeros_like(self.Ez)
        self.Hy = torch.zeros_like(self.Ez)

    # ------------------------------------------------------------------ setup
    def _build_coefficients(self, eps, sig, mu):
        NX, NY = self.NX, self.NY
        dt = self.dt
        HA = VACUUM_PERMEABILITY * mu / dt
        e_term = VACUUM_PERMITTIVITY * eps / dt
        s_term = 0.5 * sig
        EA = e_term + s_term
        EB = e_term - s_term
        pec = sig > PEC_CONDUCTIVITY_THRESHOLD
        valid = torch.zeros((NX, NY), dtype=torch.bool, device=self.device)
        valid[: NX - 1, : NY - 1] = True
        zero = torch.zeros_like(eps)
        self.ch_curl = torch.where(valid, (1.0 / self.dx) / HA, zero)
        self.ch_rhs = torch.where(valid, 1.0 / HA, zero)
        e_ok = valid & ~pec
        self.ce_hist = torch.where(e_ok, EB / EA, zero)
        self.ce_curl = torch.where(e_ok, (1.0 / self.dx) / EA, zero)
        self.ce_rhs = torch.where(e_ok, 1.0 / EA, zero)
        self.eps = VACUUM_PERMITTIVITY * eps
        self.mu = VACUUM_PERMEABILITY * mu
        self.ratio = self.dx / self.dy

    def _build_pml(self, coefficients):
        (x01, x02, xm1, xm2, y01, y02, ym1, ym2) = [
            c.to(self.device, self.dtype) for c in coefficients[:8]
        ]
        p0, p1, p2, p3 = self.pml[:4]
        NX, NY, S = self.NX, self.NY, self.nshot
        z = lambda *s: torch.zeros(s, dtype=self.dtype, device=self.device)  # noqa: E731

        def unpack(c, flip):
            # c: (4, 1, p) -> RA-1, RB, RE, RF, each (p,) ordered by grid index.
            values = [c[0, 0] - 1.0, c[1, 0], c[2, 0], c[3, 0]]
            return [v.flip(0) if flip else v for v in values]

        self.cpml = []
        # E-side: x0 rows 1..p (i1 = p - i), xm rows NX-1-p .. NX-2 (i1 = i - (NX-1-p)).
        if p0 > 0:
            self.cpml.append(dict(kind="e_xface", rows=(1, p0 + 1), coef=unpack(x01, True),
                                  phi=z(S, p0, NY)))
            self.cpml.append(dict(kind="h_xface", rows=(0, p0), coef=unpack(x02, True),
                                  phi=z(S, p0, NY)))
        if p1 > 0:
            self.cpml.append(dict(kind="e_xface", rows=(NX - 1 - p1, NX - 1), coef=unpack(xm1, False),
                                  phi=z(S, p1, NY)))
            self.cpml.append(dict(kind="h_xface", rows=(NX - 1 - p1, NX - 1), coef=unpack(xm2, False),
                                  phi=z(S, p1, NY)))
        if p2 > 0:
            self.cpml.append(dict(kind="e_yface", cols=(1, p2 + 1), coef=unpack(y01, True),
                                  phi=z(S, NX, p2)))
            self.cpml.append(dict(kind="h_yface", cols=(0, p2), coef=unpack(y02, True),
                                  phi=z(S, NX, p2)))
        if p3 > 0:
            self.cpml.append(dict(kind="e_yface", cols=(NY - 1 - p3, NY - 1), coef=unpack(ym1, False),
                                  phi=z(S, NX, p3)))
            self.cpml.append(dict(kind="h_yface", cols=(NY - 1 - p3, NY - 1), coef=unpack(ym2, False),
                                  phi=z(S, NX, p3)))

    def _build_row_ranges(self):
        """Active row ranges ``[a, b)`` for Ez/Hx and for Hy."""
        NX = self.NX
        if self.band is None:
            self.e_ranges = [(1, NX - 1)]
            self.hy_ranges = [(0, NX - 1)]
            self.ghost_top = self.ghost_bottom = None
            return
        top, bottom = self.band
        self.e_ranges, self.hy_ranges = [], []
        if top is not None:
            if top < 2:
                raise ValueError("The FDTD region above the band needs at least two Ez rows.")
            self.e_ranges.append((1, top + 1))
            self.hy_ranges.append((0, top))
        if bottom is not None:
            if bottom > NX - 3:
                raise ValueError("The FDTD region below the band needs at least two Ez rows.")
            self.e_ranges.append((bottom, NX - 1))
            self.hy_ranges.append((bottom, NX - 1))
        self.ghost_top = top
        self.ghost_bottom = None if bottom is None else bottom - 1
        p0, p1 = self.pml[0], self.pml[1]
        if top is not None and top <= p0 + 1:
            raise ValueError("The DGTD band must start below the top (x0) CPML.")
        if bottom is not None and bottom >= NX - 2 - p1:
            raise ValueError("The DGTD band must end above the bottom (xm) CPML.")
        # A band that reaches an outer row absorbs that face's CPML itself.
        self.cpml = [
            entry for entry in self.cpml
            if not ("rows" in entry and (
                (top is None and entry["rows"][0] < NX // 2)
                or (bottom is None and entry["rows"][0] >= NX // 2)))
        ]

    # ------------------------------------------------------------- time step
    def update_h(self) -> None:
        """``H^(n+1/2)`` from ``Ez^n`` (DeepGPR ``update_h`` + ``cpml_h``)."""
        Ez, Hx, Hy = self.Ez, self.Hx, self.Hy
        NY = self.NY
        for a, b in self.e_ranges:
            uh_y = self.ch_curl[a:b, : NY - 1] * self.dx / self.dy
            Hx[:, a:b, : NY - 1] = Hx[:, a:b, : NY - 1] - uh_y * (
                Ez[:, a:b, 1:NY] - Ez[:, a:b, : NY - 1])
        for a, b in self.hy_ranges:
            Hy[:, a:b, 1 : NY - 1] = Hy[:, a:b, 1 : NY - 1] + self.ch_curl[a:b, 1 : NY - 1] * (
                Ez[:, a + 1 : b + 1, 1 : NY - 1] - Ez[:, a:b, 1 : NY - 1])
        for entry in self.cpml:
            ra, rb, re, rf = entry["coef"]
            phi = entry["phi"]
            if entry["kind"] == "h_xface":
                a, b = entry["rows"]
                d = (Ez[:, a + 1 : b + 1, 1 : NY - 1] - Ez[:, a:b, 1 : NY - 1]) / self.dx
                ph = phi[:, :, 1 : NY - 1]
                Hy[:, a:b, 1 : NY - 1] += self.ch_rhs[a:b, 1 : NY - 1] * (
                    ra[:, None] * d + rb[:, None] * ph)
                phi[:, :, 1 : NY - 1] = re[:, None] * ph - rf[:, None] * d
            elif entry["kind"] == "h_yface":
                a, b = entry["cols"]
                for r0, r1 in self.e_ranges:
                    d = (Ez[:, r0:r1, a + 1 : b + 1] - Ez[:, r0:r1, a:b]) / self.dy
                    ph = phi[:, r0:r1, :]
                    Hx[:, r0:r1, a:b] -= self.ch_rhs[r0:r1, a:b] * (ra * d + rb * ph)
                    phi[:, r0:r1, :] = re * ph - rf * d

    def set_ghost(self, top_trace: Optional[torch.Tensor], bottom_trace: Optional[torch.Tensor]):
        """Write the SBP ghost values from the DGTD tangential traces ``H_b``.

        ``top_trace``/``bottom_trace`` have shape ``(nshot, NY)`` (the trace at
        the Ez nodes of the interface row; the outer PEC columns are unused).
        """
        if self.ghost_top is not None and top_trace is not None:
            g = self.ghost_top
            self.Hy[:, g, :] = 2.0 * top_trace - self.Hy[:, g - 1, :]
        if self.ghost_bottom is not None and bottom_trace is not None:
            g = self.ghost_bottom
            self.Hy[:, g, :] = 2.0 * bottom_trace - self.Hy[:, g + 1, :]

    def update_e(self) -> None:
        """``Ez^(n+1)`` from ``H^(n+1/2)`` (DeepGPR ``update_e`` + ``cpml_e``)."""
        Ez, Hx, Hy = self.Ez, self.Hx, self.Hy
        NY = self.NY
        for a, b in self.e_ranges:
            ue1 = self.ce_curl[a:b, 1 : NY - 1]
            Ez[:, a:b, 1 : NY - 1] = (
                self.ce_hist[a:b, 1 : NY - 1] * Ez[:, a:b, 1 : NY - 1]
                + ue1 * (Hy[:, a:b, 1 : NY - 1] - Hy[:, a - 1 : b - 1, 1 : NY - 1])
                - ue1 * self.dx / self.dy * (Hx[:, a:b, 1 : NY - 1] - Hx[:, a:b, : NY - 2])
            )
        for entry in self.cpml:
            ra, rb, re, rf = entry["coef"]
            phi = entry["phi"]
            if entry["kind"] == "e_xface":
                a, b = entry["rows"]
                d = (Hy[:, a:b, 1 : NY - 1] - Hy[:, a - 1 : b - 1, 1 : NY - 1]) / self.dx
                ph = phi[:, :, 1 : NY - 1]
                Ez[:, a:b, 1 : NY - 1] += self.ce_rhs[a:b, 1 : NY - 1] * (
                    ra[:, None] * d + rb[:, None] * ph)
                phi[:, :, 1 : NY - 1] = re[:, None] * ph - rf[:, None] * d
            elif entry["kind"] == "e_yface":
                a, b = entry["cols"]
                for r0, r1 in self.e_ranges:
                    d = (Hx[:, r0:r1, a:b] - Hx[:, r0:r1, a - 1 : b - 1]) / self.dy
                    ph = phi[:, r0:r1, :]
                    Ez[:, r0:r1, a:b] -= self.ce_rhs[r0:r1, a:b] * (ra * d + rb * ph)
                    phi[:, r0:r1, :] = re * ph - rf * d

    # ------------------------------------------------------- energy, operator
    def node_weights(self) -> torch.Tensor:
        """Row weights of the Ez/Hx nodes (``1/2`` on interface rows)."""
        w = torch.zeros(self.NX, dtype=self.dtype, device=self.device)
        for a, b in self.e_ranges:
            w[a:b] = 1.0
        if self.band is not None:
            for row in self.band:
                if row is not None:
                    w[row] = 0.5
        return w

    def hy_weights(self) -> torch.Tensor:
        w = torch.zeros(self.NX, dtype=self.dtype, device=self.device)
        for a, b in self.hy_ranges:
            w[a:b] = 1.0
        return w

    def energy(self, Ez_prev: torch.Tensor, Hx_prev: torch.Tensor,
               Hy_prev: torch.Tensor) -> torch.Tensor:
        """Leapfrog-invariant energy ``1/2 (eps E^n E^n + mu H^(n-1/2) H^(n+1/2))``
        per shot, called after a step with the ``E^n`` and ``H^(n-1/2)`` saved
        before it (the current H is ``H^(n+1/2)``)."""
        cell = self.dx * self.dy
        we = self.node_weights()[:, None]
        wy = self.hy_weights()[:, None]
        NY = self.NY
        e_part = (we * self.eps * Ez_prev**2)[:, :, 1 : NY - 1].sum(dim=(1, 2))
        hx_part = (we * self.mu * Hx_prev * self.Hx)[:, :, : NY - 1].sum(dim=(1, 2))
        hy_part = (wy * self.mu * Hy_prev * self.Hy)[:, :, 1 : NY - 1].sum(dim=(1, 2))
        return 0.5 * cell * (e_part + hx_part + hy_part)

    def interface_row_values(self, row: int) -> torch.Tensor:
        return self.Ez[:, row, :]

    def zero_(self) -> None:
        for t in (self.Ez, self.Hx, self.Hy):
            t.zero_()
        for entry in self.cpml:
            entry["phi"].zero_()


__all__ = ["YeeTM2D"]
