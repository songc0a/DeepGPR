"""Matplotlib helpers for models, radargrams and inversion results.

All functions accept PyTorch tensors or NumPy arrays, never modify their
inputs, import matplotlib lazily, and return the ``Axes`` (or ``Figure``) so
they can be embedded in larger figures. Following the examples, the first
array axis of a model is drawn vertically.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def _to_numpy(data: Any) -> np.ndarray:
    """Convert a tensor or array-like to a NumPy array without copying if possible."""
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return np.asarray(data.numpy())
    return np.asarray(data)


def _model_slice(model: Any, axis: int = 2, index: Optional[int] = None) -> np.ndarray:
    """Return a 2D view of a 2D, ``(nx, ny, 1)`` or 3D model."""
    array = _to_numpy(model)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        position = array.shape[axis] // 2 if index is None else index
        array = np.take(array, position, axis=axis)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D model, got shape {array.shape}.")
    return array


def _axes(ax: Any = None, figsize: Tuple[float, float] = (6.0, 4.0)) -> Any:
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax
    _, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def plot_model(
    model: Any,
    *,
    ax: Any = None,
    title: Optional[str] = None,
    spacing: Optional[Sequence[float]] = None,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None,
    slice_axis: int = 2,
    slice_index: Optional[int] = None,
) -> Any:
    """Draw a permittivity/conductivity model (or a slice of a 3D model).

    Args:
        model: ``(nx, ny)``, ``(nx, ny, 1)`` or ``(nx, ny, nz)`` data.
        ax: Existing axes; a new figure is created when ``None``.
        title: Axes title.
        spacing: ``(d_first_axis, d_second_axis)`` [m] to label axes in metres.
        cmap: Colormap name.
        vmin, vmax: Colour limits.
        colorbar_label: Label of the colour bar.
        slice_axis: Axis sliced for 3D models.
        slice_index: Slice position (default: centre).

    Returns:
        The matplotlib ``Axes``.
    """
    array = _model_slice(model, slice_axis, slice_index)
    ax = _axes(ax)
    extent = None
    if spacing is not None:
        d0, d1 = float(spacing[0]), float(spacing[1])
        extent = (0.0, array.shape[1] * d1, array.shape[0] * d0, 0.0)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    image = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    colorbar = ax.figure.colorbar(image, ax=ax)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    return ax


def plot_bscan(
    receiver_data: Any,
    *,
    shot: int = 0,
    dt: Optional[float] = None,
    ax: Any = None,
    title: Optional[str] = None,
    clip_percentile: float = 99.0,
    cmap: str = "gray",
) -> Any:
    """Draw a radargram (time x receiver) of one shot.

    Args:
        receiver_data: ``(nstep, nt, nrx)`` output of :func:`DeepGPR.compute`
            or a single ``(nt, nrx)`` gather.
        shot: Shot index for 3D input.
        dt: Time step [s]; when given, the time axis is labelled in ns.
        ax: Existing axes.
        title: Axes title.
        clip_percentile: Symmetric colour clip as a percentile of ``|data|``.
        cmap: Colormap name.

    Returns:
        The matplotlib ``Axes``.
    """
    array = _to_numpy(receiver_data)
    if array.ndim == 3:
        array = array[shot]
    if array.ndim != 2:
        raise ValueError(f"Expected (nstep, nt, nrx) or (nt, nrx) data, got {array.shape}.")
    limit = float(np.percentile(np.abs(array), clip_percentile)) or 1.0
    ax = _axes(ax)
    extent = None
    if dt is not None:
        extent = (-0.5, array.shape[1] - 0.5, array.shape[0] * dt * 1e9, 0.0)
        ax.set_ylabel("Time (ns)")
    else:
        ax.set_ylabel("Time step")
    ax.set_xlabel("Receiver")
    ax.imshow(array, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto", extent=extent)
    if title:
        ax.set_title(title)
    return ax


def plot_trace(
    trace: Any,
    *,
    dt: Optional[float] = None,
    ax: Any = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Plot a single 1D trace (e.g. one receiver) against time.

    Args:
        trace: 1D data.
        dt: Time step [s]; when given the x axis is in ns.
        ax: Existing axes.
        label: Legend label.
        title: Axes title.
    """
    array = _to_numpy(trace).reshape(-1)
    ax = _axes(ax)
    if dt is not None:
        ax.plot(np.arange(array.size) * dt * 1e9, array, label=label)
        ax.set_xlabel("Time (ns)")
    else:
        ax.plot(array, label=label)
        ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    if label:
        ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_model_comparison(
    true_model: Any,
    initial_model: Any,
    inverted_model: Any,
    *,
    spacing: Optional[Sequence[float]] = None,
    cmap: str = "jet",
    colorbar_label: Optional[str] = None,
    titles: Sequence[str] = ("True model", "Initial model", "Inverted model"),
    figsize: Tuple[float, float] = (12.0, 4.0),
) -> Any:
    """Side-by-side true / initial / inverted models with shared colour limits.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    arrays = [_model_slice(model) for model in (true_model, initial_model, inverted_model)]
    vmin = min(float(np.nanmin(array)) for array in arrays)
    vmax = max(float(np.nanmax(array)) for array in arrays)
    figure, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    for axis, array, title in zip(axes, arrays, titles):
        plot_model(
            array,
            ax=axis,
            title=title,
            spacing=spacing,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
        )
    return figure


__all__ = ["plot_bscan", "plot_model", "plot_model_comparison", "plot_trace"]
