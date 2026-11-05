"""Stateless multilinear-algebra helpers shared by the Tucker modules.

Only pure functions live here. The model parameterization
(:mod:`DeepGPR.tucker.model`) and the wavefield sketch
(:mod:`DeepGPR.tucker.sketch`) never share tensors, random matrices or
optimizer state through this module.

Conventions: a tensor with ``batch`` leading batch axes has its Tucker modes
on the remaining axes; mode ``n`` is axis ``batch + n``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Sequence

import torch


@contextmanager
def matmul_precision(allow_tf32: bool) -> Iterator[None]:
    """Temporarily set ``torch.backends.cuda.matmul.allow_tf32``.

    PyTorch < 1.12 enables TF32 matmuls by default on Ampere and newer GPUs
    (about 1e-3 relative precision), which would dominate the error of a
    float32 Tucker sketch. The previous setting is always restored.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def mode_dot(x: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    """Mode product ``y = x ×_axis matrix`` with an unbatched matrix.

    Args:
        x: Tensor whose ``axis`` has size ``I``.
        matrix: ``(J, I)`` matrix.
        axis: Contracted axis of ``x`` (non-negative).

    Returns:
        ``x`` with ``axis`` replaced by an axis of size ``J``.
    """
    y = torch.tensordot(x, matrix, dims=([axis], [1]))
    return torch.movedim(y, -1, axis)


def batched_mode_dot(x: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    """Mode product with one matrix per batch entry.

    Args:
        x: ``(B, ...)`` tensor whose ``axis`` (>= 1) has size ``I``.
        matrix: ``(B, J, I)``.
        axis: Contracted axis of ``x``.

    Returns:
        ``(B, ...)`` with ``axis`` replaced by an axis of size ``J``.
    """
    moved = torch.movedim(x, axis, -1)
    shape = moved.shape
    flat = moved.reshape(shape[0], -1, shape[-1])
    y = torch.bmm(flat, matrix.transpose(1, 2))
    y = y.reshape(*shape[:-1], matrix.shape[1])
    return torch.movedim(y, -1, axis)


def unfold(x: torch.Tensor, mode: int, batch: int = 0) -> torch.Tensor:
    """Mode-``mode`` unfolding ``X_(n)`` with the remaining modes in order.

    Args:
        x: Tensor with ``batch`` leading batch axes.
        mode: Tucker mode (counted after the batch axes).
        batch: Number of leading batch axes.

    Returns:
        ``(*batch_shape, I_mode, prod(other modes))``.
    """
    axis = batch + mode
    moved = torch.movedim(x, axis, batch)
    return moved.reshape(*x.shape[:batch], x.shape[axis], -1)


def khatri_rao_contract(
    x: torch.Tensor, matrices: Dict[int, torch.Tensor], keep: int, batch: int = 1
) -> torch.Tensor:
    """Contract every mode except ``keep`` with the columns of Khatri-Rao factors.

    Computes ``Y[b, i, j] = sum_{other} X[b, ...] prod_m A_m[i_m, j]``, i.e.
    ``Y = X_(keep) (A_{m_1} ⊙ ... ⊙ A_{m_K})`` for the Khatri-Rao product of
    the matrices of all other modes, without ever forming that product.

    Args:
        x: Tensor with ``batch`` leading axes and ``M`` Tucker modes.
        matrices: ``{mode: (I_mode, k)}`` for every mode except ``keep``.
        keep: The mode that remains.
        batch: Number of leading batch axes.

    Returns:
        ``(*batch_shape, I_keep, k)``.
    """
    modes = x.dim() - batch
    others = [mode for mode in range(modes) if mode != keep]
    if sorted(matrices) != others:
        raise ValueError("khatri_rao_contract needs one matrix for every other mode.")
    if not others:
        raise ValueError("khatri_rao_contract needs at least two modes.")
    # The first contraction is a dense GEMM; contracting the largest mode
    # first gives the smallest intermediate.
    order = sorted(others, key=lambda mode: -x.shape[batch + mode])
    remaining = list(range(modes))
    first = order[0]
    y = torch.tensordot(x, matrices[first], dims=([batch + first], [0]))
    remaining.remove(first)
    letters = "abcdefghijklmnopqrstuvwxy"
    for mode in order[1:]:
        position = remaining.index(mode)
        rank_letter = "z"
        batch_letters = letters[:batch]
        mode_letters = letters[batch : batch + len(remaining)]
        source = batch_letters + mode_letters + rank_letter
        target = (
            batch_letters
            + mode_letters[:position]
            + mode_letters[position + 1 :]
            + rank_letter
        )
        y = torch.einsum(
            f"{source},{mode_letters[position]}{rank_letter}->{target}", y, matrices[mode]
        )
        remaining.remove(mode)
    return y


def gram_leading_vectors(unfolded: torch.Tensor, rank: int) -> torch.Tensor:
    """Leading left singular vectors of a (batched) unfolding via its Gram matrix.

    Args:
        unfolded: ``(..., I, P)``.
        rank: Number of vectors (``<= I``).

    Returns:
        ``(..., I, rank)`` orthonormal columns, ordered by decreasing singular value.
    """
    gram = unfolded @ unfolded.transpose(-1, -2)
    # eigh returns ascending eigenvalues.
    _, vectors = torch.linalg.eigh(gram)
    return vectors[..., -rank:].flip(-1)


def leading_left_singular_vectors(unfolded: torch.Tensor, rank: int) -> torch.Tensor:
    """Leading left singular vectors of an unbatched unfolding via the SVD.

    More accurate than :func:`gram_leading_vectors` (no squared condition
    number); used for one-off model initialisation.
    """
    left, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
    return left[..., :rank]


def tensor_norm(x: torch.Tensor, batch: int = 0) -> torch.Tensor:
    """Frobenius norm over the Tucker modes (per batch entry)."""
    if batch == 0:
        return torch.linalg.norm(x.reshape(-1))
    return torch.linalg.norm(x.reshape(*x.shape[:batch], -1), dim=-1)


def relative_error(
    estimate: torch.Tensor, reference: torch.Tensor, floor: Optional[float] = None
) -> float:
    """``||estimate - reference|| / ||reference||`` with an explicit zero rule.

    Returns 0.0 when both are exactly zero and ``inf`` when only the reference
    is zero, so a zero reference never produces a NaN.

    Args:
        estimate: Approximation.
        reference: Exact tensor (same shape).
        floor: Optional denominator floor; ``None`` keeps the zero rule above.
    """
    difference = float(torch.linalg.norm((estimate - reference).reshape(-1).double()))
    scale = float(torch.linalg.norm(reference.reshape(-1).double()))
    if floor is not None:
        return difference / max(scale, floor)
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """``<a, b> / (||a|| ||b||)``; 1.0 for two zero tensors, 0.0 if only one is zero."""
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    norm_a = float(torch.linalg.norm(a))
    norm_b = float(torch.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(torch.dot(a, b)) / (norm_a * norm_b)


def normalize_ranks(ranks: object, ndim: int, name: str = "ranks") -> Sequence[int]:
    """Expand an int or validate a sequence of ``ndim`` positive integer ranks."""
    if isinstance(ranks, bool):
        raise TypeError(f"{name} must be a positive integer or a sequence of them.")
    if isinstance(ranks, int):
        values = (ranks,) * ndim
    else:
        try:
            values = tuple(ranks)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be a positive integer or a sequence of them.") from exc
        if len(values) != ndim:
            raise ValueError(f"{name} must contain {ndim} values, got {len(values)}.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must contain positive integers, got {values!r}.")
    return values


__all__ = [
    "batched_mode_dot",
    "cosine_similarity",
    "gram_leading_vectors",
    "khatri_rao_contract",
    "leading_left_singular_vectors",
    "matmul_precision",
    "mode_dot",
    "normalize_ranks",
    "relative_error",
    "tensor_norm",
    "unfold",
]
