"""Block-Tucker gradient error (2 cm, shot 5): DCT spectra (even extension, no taper, no wrap-around) and the error
left after a DCT low-pass, split into the top 0.2 m and the ground below.

    python error_character_dct_2cm.py gradcheck_block_2cm error_character_2cm_dct
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

DX = 0.02
TOP = 10  # rows (0.2 m)
CASES = {"full_it30_shot5": [("full_it30_shot5", "block0.03"), ("full_it30_shot5_src0.1", "block0.03_src0.1"),
                             ("full_it30_shot5_src0.1", "block0.02_src0.1"), ("full_it30_shot5_src0.1", "block0.015_src0.1")],
         "full_init_shot5": [("full_init_shot5_src0.1", "block0.03_src0.1"), ("full_init_shot5_src0.1", "block0.015_src0.1")]}
CUTS = (8.0, 12.0, 16.0)
EDGES = np.linspace(0, 25 * np.sqrt(3), 120)


def norm(a):
    return float(np.sqrt(np.sum(a.astype(np.float64) ** 2)))


src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=False)
res = {"k_cycles_per_m": (0.5 * (EDGES[1:] + EDGES[:-1])).tolist(), "top_rows": TOP}
for model, cases in CASES.items():
    g = np.load(src / f"{model}_shot05_exact.npy")
    shape = g.shape
    k = [np.arange(n) / (2 * n * DX) for n in shape]
    kr = np.sqrt(k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2).astype(np.float32)
    G = dctn(g, type=2, norm="ortho", workers=16)
    spec_g, _ = np.histogram(kr.ravel(), bins=EDGES, weights=(G.astype(np.float64) ** 2).ravel())
    r = {"spectrum_exact": spec_g.tolist(),
         "exact_power_above": {c: float(spec_g[res_k > c].sum() / spec_g.sum()) for c, res_k in
                               ((c, np.array(res["k_cycles_per_m"])) for c in (6.0, 12.0))},
         "exact_top_share": float(norm(g[:, :TOP]) ** 2 / norm(g) ** 2)}
    gf = {c: idctn(G * (kr <= c), type=2, norm="ortho", workers=16) for c in CUTS}
    r["exact_change"] = {c: {"all": norm(gf[c] - g) / norm(g), "top": norm(gf[c][:, :TOP] - g[:, :TOP]) / norm(g[:, :TOP]),
                             "below": norm(gf[c][:, TOP:] - g[:, TOP:]) / norm(g[:, TOP:])} for c in CUTS}
    print(model, "exact: top-0.2 m energy share", round(r["exact_top_share"], 3), "power above 6/12 /m",
          {c: round(v, 5) for c, v in r["exact_power_above"].items()},
          "changed by cut-off (all/top/below, %)",
          {c: [round(v * 100, 2) for v in d.values()] for c, d in r["exact_change"].items()}, flush=True)
    for tag, method in cases:
        e = np.load(src / f"{tag}_shot05_{method}.npy") - g
        Ee = dctn(e, type=2, norm="ortho", workers=16)
        spec_e, _ = np.histogram(kr.ravel(), bins=EDGES, weights=(Ee.astype(np.float64) ** 2).ravel())
        row = {"spectrum": spec_e.tolist(),
               "unfiltered": {"all": norm(e) / norm(g), "top": norm(e[:, :TOP]) / norm(g[:, :TOP]),
                              "below": norm(e[:, TOP:]) / norm(g[:, TOP:])},
               "error_power_above": {c: float(spec_e[np.array(res["k_cycles_per_m"]) > c].sum() / spec_e.sum()) for c in (6.0, 12.0)}}
        for c in CUTS:
            ef = idctn(Ee * (kr <= c), type=2, norm="ortho", workers=16)
            row[c] = {"all": norm(ef) / norm(gf[c]), "top": norm(ef[:, :TOP]) / norm(gf[c][:, :TOP]),
                      "below": norm(ef[:, TOP:]) / norm(gf[c][:, TOP:])}
        r[method] = row
        print(model, method, "error power above 6/12 /m", {c: round(v, 3) for c, v in row["error_power_above"].items()},
              "rel error % (all/top/below):", {key: [round(v * 100, 2) for v in row[key].values()]
                                                for key in ("unfiltered",) + CUTS}, flush=True)
    res[model] = r
(out / "error_character_dct_2cm.json").write_text(json.dumps(res, indent=1))
