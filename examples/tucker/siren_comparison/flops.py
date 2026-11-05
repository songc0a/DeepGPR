"""Analytic operation counts of SIREN, Tucker-SIREN and the Tucker reconstruction.

Conventions (stated, not measured):

* A multiply-add counts as 2 FLOPs. Linear layer ``n_in -> n_out`` on ``P``
  points: ``2 P n_in n_out`` (+ ``P n_out`` bias adds).
* ``sin(omega * z)``: ``P n`` multiplies (omega) and ``P n`` sine evaluations.
  Sines run on the special-function units, so they are reported separately
  (``transcendental``) and not added to the FLOP total.
* Backward of a linear layer: ``2x`` its forward GEMM FLOPs (input and weight
  gradients; the first layer's input gradient is not needed but PyTorch
  computes it only if requested - we count the standard ``2x``). Sine
  backward: one cosine (transcendental) and two multiplies per element.
* Tucker reconstruction ``G x_1 U_1 ... x_N U_N``: modes applied in the order
  of ``DeepGPR.tucker.tucker_reconstruct`` (increasing ``I_n / r_n``); a mode
  product on a tensor of ``K`` elements costs ``2 K I_n`` FLOPs. Backward:
  ``2x`` forward (gradient to the input tensor and to the factor).
* Property mapping (sigmoid, affine, ``10**x``) is elementwise on the full
  grid for every method; counted as ``mapping_elementwise`` (not FLOPs).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence


def siren_counts(points: int, in_features: int, out_features: int, hidden: int, layers: int) -> Dict[str, float]:
    gemm = 2 * points * (in_features * hidden + (layers - 1) * hidden * hidden + hidden * out_features)
    bias = points * (layers * hidden + out_features)
    omega = points * layers * hidden
    sines = points * layers * hidden
    forward = gemm + bias + omega
    backward = 2 * gemm + bias + 2 * omega * 1  # weight/input grads + sine-derivative multiplies
    return dict(
        forward_flops=forward,
        backward_flops=backward,
        gemm_forward_flops=gemm,
        transcendental_forward=sines,
        transcendental_backward=sines,
        activation_elements=points * (2 * layers * hidden + out_features),
    )


def tucker_reconstruct_counts(shape: Sequence[int], ranks: Sequence[int]) -> Dict[str, float]:
    order = sorted(range(len(shape)), key=lambda m: shape[m] / max(ranks[m], 1))
    sizes = list(ranks)
    forward = 0.0
    intermediates = 0.0
    for mode in order:
        current = math.prod(sizes)
        forward += 2 * current * shape[mode]
        sizes[mode] = shape[mode]
        intermediates += math.prod(sizes)
    return dict(forward_flops=forward, backward_flops=2 * forward, intermediate_elements=intermediates)


def method_counts(spec: Dict, shape: Sequence[int]) -> Dict[str, float]:
    """Total analytic counts for one generation of (eps, sigma) on ``shape``."""
    cells = math.prod(shape)
    out: Dict[str, float] = dict(cells=cells, mapping_elementwise=2 * cells)
    if spec["method"] == "siren":
        s = siren_counts(cells, len(shape), 2, spec["hidden"], spec["layers"])
        out.update(
            network_points=cells,
            network_forward_flops=s["forward_flops"],
            network_backward_flops=s["backward_flops"],
            network_transcendental=s["transcendental_forward"],
            reconstruction_forward_flops=0.0,
            reconstruction_backward_flops=0.0,
        )
    else:
        ranks = [spec["rank"]] * len(shape)
        net_f = net_b = trans = 0.0
        points = 0
        for prop in range(2):  # eps and sigma: independent networks
            for n, r in zip(shape, ranks):
                s = siren_counts(n, 1, r, spec["hidden"], spec["layers"])
                net_f += s["forward_flops"]
                net_b += s["backward_flops"]
                trans += s["transcendental_forward"]
                points += n
        rec = tucker_reconstruct_counts(shape, ranks)
        out.update(
            network_points=points,
            network_forward_flops=net_f,
            network_backward_flops=net_b,
            network_transcendental=trans,
            reconstruction_forward_flops=2 * rec["forward_flops"],
            reconstruction_backward_flops=2 * rec["backward_flops"],
            reconstruction_intermediate_elements=2 * rec["intermediate_elements"],
        )
    out["total_forward_flops"] = out["network_forward_flops"] + out["reconstruction_forward_flops"]
    out["total_backward_flops"] = out["network_backward_flops"] + out["reconstruction_backward_flops"]
    return out


__all__ = ["method_counts", "siren_counts", "tucker_reconstruct_counts"]
