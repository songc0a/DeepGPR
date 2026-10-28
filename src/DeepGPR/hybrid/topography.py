"""Ground-surface descriptions for the hybrid solver.

DeepGPR's 2D models are indexed ``eps_r[i, j]`` with axis 0 (``x``) the depth
direction (positive downwards) and axis 1 (``y``) the horizontal direction.
A surface is therefore described by its depth ``x = d(y)`` [m] as a function of
the horizontal position ``y`` [m], both in the *physical* model frame (node
``(i, j)`` of the user's model sits at ``(i dx, j dy)``). Points with
``x < d(y)`` are air, points with ``x >= d(y)`` are ground.

Elevation data (height ``z`` positive upwards) is converted with
``d(y) = datum - z(y)``; see :meth:`Surface.from_elevation`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


class Surface:
    """Base class: ``depth(y)`` returns the surface depth [m] at positions ``y`` [m]."""

    def depth(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y) -> np.ndarray:
        return self.depth(np.asarray(y, dtype=np.float64))

    def point_at_height(self, y: float, height: float):
        """Physical ``(x, y)`` of a point ``height`` [m] above the surface at ``y``."""
        return float(self.depth(np.array([y]))[0]) - float(height), float(y)

    def is_air(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask of points above the surface."""
        return np.asarray(x) < self.depth(np.asarray(y, dtype=np.float64))

    @staticmethod
    def flat(depth: float) -> "Surface":
        """Horizontal surface at ``depth`` [m]."""
        return CallableSurface(lambda y: np.full_like(y, float(depth), dtype=np.float64))

    @staticmethod
    def sine(depth: float, amplitude: float, wavelength: float, phase: float = 0.0) -> "Surface":
        """``d(y) = depth + amplitude sin(2 pi y / wavelength + phase)``."""
        k = 2 * np.pi / float(wavelength)
        return CallableSurface(lambda y: float(depth) + float(amplitude) * np.sin(k * y + phase))

    @staticmethod
    def slope(depth: float, gradient: float, y0: float = 0.0,
              y_start: Optional[float] = None, y_end: Optional[float] = None) -> "Surface":
        """Plane slope ``d(y) = depth + gradient (y - y0)``, optionally flat outside
        ``[y_start, y_end]`` (continuous ramp between two flat parts)."""
        def fn(y):
            yc = y
            if y_start is not None or y_end is not None:
                yc = np.clip(y, -np.inf if y_start is None else y_start,
                             np.inf if y_end is None else y_end)
            return float(depth) + float(gradient) * (yc - float(y0))
        return CallableSurface(fn)

    @staticmethod
    def from_samples(y: Sequence[float], depth: Sequence[float], method: str = "pchip") -> "Surface":
        """Surface interpolated from samples ``depth(y)`` (``"linear"`` or ``"pchip"``)."""
        return SampledSurface(y, depth, method=method)

    @staticmethod
    def from_elevation(y: Sequence[float], elevation: Sequence[float], datum: float,
                       method: str = "pchip") -> "Surface":
        """Surface from elevation samples ``z(y)`` (up positive): ``d = datum - z``."""
        z = np.asarray(elevation, dtype=np.float64)
        return SampledSurface(y, float(datum) - z, method=method)


class CallableSurface(Surface):
    """Surface defined by a vectorised callable ``depth(y)``."""

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self._function = function

    def depth(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(self._function(np.asarray(y, dtype=np.float64)), dtype=np.float64)
        return np.broadcast_to(values, np.shape(y)).copy()


def _pchip_slopes(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite slopes."""
    h = np.diff(y)
    delta = np.diff(d) / h
    slopes = np.zeros_like(d)
    if d.size == 2:
        slopes[:] = delta[0]
        return slopes
    for k in range(1, d.size - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # One-sided three-point end slopes, limited to keep monotonicity.
    for end, (hk, hk1, dk, dk1) in (
        (0, (h[0], h[1], delta[0], delta[1])),
        (-1, (h[-1], h[-2], delta[-1], delta[-2])),
    ):
        slope = ((2 * hk + hk1) * dk - hk * dk1) / (hk + hk1)
        if np.sign(slope) != np.sign(dk):
            slope = 0.0
        elif np.sign(dk) != np.sign(dk1) and abs(slope) > abs(3 * dk):
            slope = 3 * dk
        slopes[end] = slope
    return slopes


class SampledSurface(Surface):
    """Surface interpolated from discrete samples (constant extrapolation)."""

    def __init__(self, y: Sequence[float], depth: Sequence[float], method: str = "pchip"):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        if y.size != d.size or y.size < 2:
            raise ValueError("Surface samples need at least two matching y and depth values.")
        order = np.argsort(y)
        y, d = y[order], d[order]
        if np.any(np.diff(y) <= 0):
            raise ValueError("Surface sample positions must be distinct.")
        if method not in ("linear", "pchip"):
            raise ValueError("method must be 'linear' or 'pchip'.")
        self.y, self.d, self.method = y, d, method
        self._slopes = _pchip_slopes(y, d) if method == "pchip" else None

    def depth(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.method == "linear":
            return np.interp(y, self.y, self.d)
        yc = np.clip(y, self.y[0], self.y[-1])
        k = np.clip(np.searchsorted(self.y, yc) - 1, 0, self.y.size - 2)
        h = self.y[k + 1] - self.y[k]
        t = (yc - self.y[k]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return (h00 * self.d[k] + h10 * h * self._slopes[k]
                + h01 * self.d[k + 1] + h11 * h * self._slopes[k + 1])


class Surface3D:
    """Ground surface of a 3D model: depth ``x = d(y, z)`` [m] over the horizontal
    axes ``y`` (axis 1) and ``z`` (axis 2), physical model frame."""

    def __init__(self, function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self._function = function

    def depth(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        y, z = np.broadcast_arrays(np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64))
        values = np.asarray(self._function(y, z), dtype=np.float64)
        return np.broadcast_to(values, y.shape).copy()

    def __call__(self, y, z) -> np.ndarray:
        return self.depth(y, z)

    def is_air(self, x, y, z) -> np.ndarray:
        return np.asarray(x) < self.depth(y, z)

    @staticmethod
    def flat(depth: float) -> "Surface3D":
        return Surface3D(lambda y, z: np.full_like(y, float(depth)))

    @staticmethod
    def sine(depth: float, amplitude: float, wavelength_y: float, wavelength_z: float,
             phase_y: float = 0.0, phase_z: float = 0.0) -> "Surface3D":
        """``d = depth + amplitude sin(2 pi y / L_y + phase_y) cos(2 pi z / L_z + phase_z)``."""
        ky, kz = 2 * np.pi / float(wavelength_y), 2 * np.pi / float(wavelength_z)
        return Surface3D(lambda y, z: float(depth) + float(amplitude)
                         * np.sin(ky * y + phase_y) * np.cos(kz * z + phase_z))

    @staticmethod
    def from_grid(y: Sequence[float], z: Sequence[float], depth) -> "Surface3D":
        """Separable monotone-cubic (PCHIP along z, then along y) interpolation of
        gridded samples ``depth[iy, iz]`` (constant extrapolation)."""
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)
        if d.shape != (y.size, z.size):
            raise ValueError("depth must have shape (len(y), len(z)).")
        rows = [SampledSurface(z, d[i]) for i in range(y.size)]

        def fn(yq, zq):
            shape = yq.shape
            yq, zq = yq.reshape(-1), zq.reshape(-1)
            cols = np.stack([r.depth(zq) for r in rows])           # (ny, nq)
            out = np.empty(yq.size)
            for q in range(yq.size):
                out[q] = SampledSurface(y, cols[:, q]).depth(np.array([yq[q]]))[0]
            return out.reshape(shape)
        return Surface3D(fn)


__all__ = ["CallableSurface", "SampledSurface", "Surface", "Surface3D"]
