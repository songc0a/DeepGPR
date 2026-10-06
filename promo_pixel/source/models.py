"""Subsurface models used both as DeepGPR inputs and as pixel-art layers."""
import numpy as np

# material id -> (eps_r, sigma [S/m])
MAT = {
    0: (1.0, 0.0),      # air
    1: (5.0, 1e-3),     # topsoil
    2: (9.0, 3e-3),     # clay
    3: (14.0, 5e-3),    # wet base
    4: (4.0, 5e-4),     # boulder
    5: (1.0, 1e3),      # metal (PEC in DeepGPR: sigma > 100 S/m)
    6: (1.0, 0.0),      # cavity (air)
}

W, AIR, DEPTH = 384, 10, 152          # scene model: 384 x (10 + 152) cells, 1 cm
SHOT_X = (80, 192, 304)


def scene_ids():
    """Material ids, shape (W, AIR + DEPTH). Index [x, y], y down."""
    ids = np.zeros((W, AIR + DEPTH), np.int32)
    x = np.arange(W)
    h1 = np.round(42 + 5 * np.sin(x / 31.0) + 3 * np.sin(x / 11.0 + 1.0)).astype(int)
    h2 = np.round(101 + 7 * np.sin(x / 47.0 + 2.0) + 3 * np.sin(x / 17.0)).astype(int)
    d = np.arange(DEPTH)[None, :]
    g = np.where(d < h1[:, None], 1, np.where(d < h2[:, None], 2, 3))
    xx, dd = np.meshgrid(x, np.arange(DEPTH), indexing="ij")

    def blob(cx, cd, rx, rd, mat, wob=0.0, seed=0):
        rng = np.random.default_rng(seed)
        ang = np.arctan2((dd - cd) / rd, (xx - cx) / rx)
        r = np.hypot((xx - cx) / rx, (dd - cd) / rd)
        edge = 1 + wob * (np.sin(3 * ang + rng.uniform(0, 6)) + 0.6 * np.sin(5 * ang + rng.uniform(0, 6)))
        g[r <= edge] = mat

    blob(134, 62, 11, 8, 4, 0.10, 1)       # boulders
    blob(246, 26, 9, 7, 4, 0.12, 2)
    blob(40, 84, 10, 7, 4, 0.10, 3)
    blob(352, 96, 12, 8, 4, 0.10, 4)
    blob(192, 72, 24, 10, 6, 0.08, 5)      # cavity
    blob(80, 30, 6.4, 6.4, 5)              # metal pipe
    g[296:313, 54:65] = 5                  # buried chest
    ids[:, AIR:] = g
    return ids


def to_fields(ids):
    eps = np.zeros(ids.shape, np.float32)
    sig = np.zeros(ids.shape, np.float32)
    for k, (e, s) in MAT.items():
        eps[ids == k] = e
        sig[ids == k] = s
    return eps, sig


# ---------------------------------------------------------------- FWI model
FW, FH, FAIR = 128, 72, 8


def fwi_true():
    """True eps_r and sigma [S/m] for the dual-parameter FWI demo, shape (128, 72)."""
    x = np.arange(FW)
    d = np.arange(FH)[None, :]
    h1 = np.round(20 + 4 * np.sin(x / 15.0) + 2 * np.sin(x / 6.0 + 1)).astype(int)
    h2 = np.round(48 + 5 * np.sin(x / 21.0 + 2.2)).astype(int)
    eps = np.where(d < h1[:, None], 3.0, np.where(d < h2[:, None], 4.5, 6.0)).astype(np.float32)
    sig = np.where(d < h1[:, None], 2.0, np.where(d < h2[:, None], 5.0, 8.0)).astype(np.float32)
    xx, dd = np.meshgrid(x, np.arange(FH), indexing="ij")
    m = np.hypot((xx - 36) / 9.0, (dd - 34) / 8.0) <= 1          # dry cavity-like body
    eps[m] = 2.0
    sig[m] = 0.5
    m = (np.abs(xx - 90) <= 11) & (np.abs(dd - 33) <= 6)          # wet, conductive block
    eps[m] = 8.0
    sig[m] = 14.0
    m = np.hypot((xx - 64) / 7.0, (dd - 58) / 5.0) <= 1           # small lens: permittivity only
    eps[m] = 3.0
    return eps, sig * 1e-3
