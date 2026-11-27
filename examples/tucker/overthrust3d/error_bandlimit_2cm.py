"""Relative error of the block-Tucker gradient after a radial low-pass at the imaging-condition limit (2 cm, shot 5).

    python error_bandlimit_2cm.py gradcheck_block_2cm error_character_2cm
"""
import json
import sys
from pathlib import Path

import numpy as np

DX = 0.02
CASES = [("full_it30_shot5", "full_it30_shot5", "block0.03"), ("full_it30_shot5", "full_it30_shot5_src0.1", "block0.03_src0.1"),
         ("full_it30_shot5", "full_it30_shot5_src0.1", "block0.02_src0.1"), ("full_it30_shot5", "full_it30_shot5_src0.1", "block0.015_src0.1"),
         ("full_init_shot5", "full_init_shot5_src0.1", "block0.03_src0.1"), ("full_init_shot5", "full_init_shot5_src0.1", "block0.015_src0.1")]
CUTS = (10.0, 12.0, 14.0, 16.0)  # cycles / m; sharp radial cut-off (periodic FFT)

src, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for model in ("full_it30_shot5", "full_init_shot5"):
    g = np.load(src / f"{model}_shot05_exact.npy")
    nx, ny, nz = g.shape
    kr = np.sqrt(np.fft.fftfreq(nx, DX)[:, None, None] ** 2 + np.fft.fftfreq(ny, DX)[None, :, None] ** 2
                 + np.fft.rfftfreq(nz, DX)[None, None, :] ** 2).astype(np.float32)
    G = np.fft.rfftn(g)
    gn = float(np.linalg.norm(g.astype(np.float64)))
    filt = {c: np.fft.irfftn(G * (kr <= c), s=g.shape) for c in CUTS}
    res[model] = {"exact_change": {c: float(np.linalg.norm(filt[c] - g) / gn) for c in CUTS}}
    for m, tag, method in CASES:
        if m != model:
            continue
        b = np.load(src / f"{tag}_shot05_{method}.npy")
        e = (b - g).astype(np.float32)
        E = np.fft.rfftn(e)
        row = {"unfiltered": float(np.linalg.norm(e.astype(np.float64)) / gn)}
        for c in CUTS:
            ef = np.fft.irfftn(E * (kr <= c), s=g.shape)
            row[c] = float(np.linalg.norm(ef.astype(np.float64)) / np.linalg.norm(filt[c].astype(np.float64)))
        res[model][method] = row
        print(model, method, {k: round(v * 100, 2) for k, v in row.items()}, flush=True)
    print(model, "exact changed by the cut-off (%)", {k: round(v * 100, 3) for k, v in res[model]["exact_change"].items()}, flush=True)
(out / "error_bandlimit_2cm.json").write_text(json.dumps(res, indent=1))
