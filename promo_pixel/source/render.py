"""DeepGPR pixel-art promo: every frame is composed here from the indexed palette."""
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter

import models
from world import *  # noqa: F401,F403
from world import _rng, _xx, _yy

LANG = "zh"
TXT = {
    "s1a": {"zh": "地表之下，藏着什么？", "en": "What lies beneath the surface?"},
    "s1b": {"zh": "探地雷达：发射电磁波，倾听回波。", "en": "GPR: send a radar pulse, listen for echoes."},
    "s2a": {"zh": "DeepGPR：在 PyTorch 里求解麦克斯韦方程", "en": "DeepGPR: Maxwell's equations inside PyTorch"},
    "s2b": {"zh": "FDTD 正演 · 2D / 3D · 2/4/8 阶 · 自动 CPML", "en": "FDTD · 2D/3D · order 2/4/8 · automatic CPML"},
    "s3a": {"zh": "可微分的探地雷达正演与全波形反演", "en": "Differentiable GPR modelling & inversion"},
    "s4t": {"zh": "梯度，一行代码", "en": "Gradients in one line"},
    "s4a": {"zh": "精确离散伴随：一行 backward() 求出梯度",
            "en": "Exact discrete adjoint: one backward() call"},
    "s5t": {"zh": "双参数全波形反演", "en": "Dual-parameter FWI"},
    "s5a": {"zh": "介电常数 εr 与电导率 σ 同时反演",
            "en": "εr and σ recovered together by DeepGPR"},
    "s6t": {"zh": "大模型，也装得下", "en": "Built for large models"},
    "s6a": {"zh": "CUDA + C/OpenMP 双后端，同一套 PyTorch 接口", "en": "CUDA + C/OpenMP backends · one PyTorch API"},
    "s6m": {"zh": "波场历史内存", "en": "History memory"},
    "s6d": {"zh": "默认", "en": "default"},
    "chips": {"zh": ["检查点分段", "时间降采样", "主机异步卸载", "多卡 DDP"],
              "en": ["Checkpointing", "Time sampling", "Host offload", "Multi-GPU DDP"]},
    "true": {"zh": "真实", "en": "TRUE"}, "inv": {"zh": "反演", "en": "FWI"},
    "s7a": {"zh": "MIT 开源 · Linux / Windows / macOS", "en": "MIT licensed · Linux / Windows / macOS"},
    "s7b": {"zh": "看见地表之下。", "en": "See beyond the surface."},
}


def tx(k):
    return TXT[k][LANG]


# ------------------------------------------------------------------ data
SC = np.load("out_scene.npz")
MOV = SC["movies"].astype(np.float32)               # (3, 110, 384, 162)
NMOV = MOV.shape[1]
_g = np.array([[np.percentile(np.abs(MOV[k, m]), 99.6) for m in range(NMOV)] for k in range(3)])
# monotone envelope so the gain never flickers
MGAIN = 1.0 / np.maximum(np.maximum.accumulate(_g[:, ::-1], axis=1)[:, ::-1], 1e-9) * 0.9
BS = np.load("out_bscan.npz")
import os

_fw = np.load(os.environ.get("FWI_NPZ", "out_fwi.npz"))
FW = {k: _fw[k] for k in _fw.files}
if FW["Et"].shape[0] == 64:          # half-resolution pilot: upsample for layout tests only
    for k in ("E", "S", "snap"):
        FW[k] = np.repeat(np.repeat(FW[k], 2, 1), 2, 2)
    for k in ("Et", "St", "g_eps", "g_sig"):
        FW[k] = np.repeat(np.repeat(FW[k], 2, 0), 2, 1)
    FW["sxs"] = FW["sxs"] * 2
CU = np.load("out_cube.npz")

T_SHOT = (2.5, 7.0, 11.0)
MOV_RATE = 15.0
MOV_END = (52, 45, 45)


def _bscan_img():
    """Background-removed, gained envelope of the common-offset section -> (rows, traces) in [0, 1]."""
    from scipy.signal import hilbert
    d = BS["data"].astype(np.float64)              # (95, nt)
    d = d - np.median(d, 0, keepdims=True)
    d = d[:, :760] * (np.arange(760)[None, :] / 760.0 + 0.03) ** 1.5
    env = np.abs(hilbert(d, axis=1))
    rows = 34
    env = env[:, 50:50 + rows * 20].reshape(d.shape[0], rows, 20).mean(2)
    env /= np.percentile(env, 99.0)
    return np.clip(env.T, 0, 1) ** 0.8


BSCAN = _bscan_img()
BS_X = BS["xs"]

_reveal = {"mask": np.zeros((models.DEPTH, W), bool), "prog": [-1, -1, -1]}
_xr = np.arange(W)[None, :]
_PEC = IDS.T == 5
_hx = hash2(*np.meshgrid(np.arange(W), np.arange(models.DEPTH)), 77)


def shot_field(k, m):
    return MOV[k, m].T[models.AIR:]                 # (depth, x)


def update_reveal(k, m):
    """Ground the radar wave has passed becomes visible (within reach of shot k)."""
    st = _reveal
    while st["prog"][k] < m:
        st["prog"][k] += 1
        a = shot_field(k, st["prog"][k]) * MGAIN[k, st["prog"][k]]
        hit = binary_dilation(np.abs(a) > 0.10, iterations=2)
        hit |= binary_dilation(hit, iterations=8) & _PEC      # no field inside metal: reveal it with its rim
        reach = np.abs(_xr - models.SHOT_X[k]) < 62 + 16 * _hx
        if k == 0:
            reach |= _xr < models.SHOT_X[0]
        if k == 2:
            reach |= _xr > models.SHOT_X[2]
        st["mask"] = binary_fill_holes(st["mask"] | (hit & reach))


def cart_x(t):
    if t < 2.5:
        return -30 + (80 + 30) * ease(t / 2.5)
    if t < 6.0:
        return 80
    if t < 7.0:
        return 80 + 112 * ease(t - 6.0)
    if t < 10.0:
        return 192
    if t < 11.0:
        return 192 + 112 * ease(t - 10.0)
    return 304


def cart_moving(t):
    return t < 2.5 or 6.0 <= t < 7.0 or 10.0 <= t < 11.0


STARS2 = [(int(_rng.integers(0, W)), int(_rng.integers(0, 120)), int(_rng.integers(0, 64)), int(_rng.integers(0, 3)))
          for _ in range(110)]


def night_bg(cv, f, grid=False):
    """Plain night backdrop for the UI shots (same sky colours as the world)."""
    v = np.clip(_yy / 300.0 + 0.05, 0, 1)
    cv[:] = dither_ramp(v, ["ink", "ink", "night", "night", "navy"])
    for (sx, sy, ph, kind) in STARS2:
        sy2 = (sy * 2) % H
        tw = ((f // 6) + ph) % 16
        if tw:
            cv[sy2, sx] = C[("fog", "steel", "slate")[kind]]


def world(cv, f, off=0, ground=None):
    """World with the camera tilted up by `off` rows."""
    off = int(off)
    if off > 0:
        cv[:off] = C["ink"]
        for (sx, sy, ph, kind) in STARS2:
            if sy < off - 1 and ((f // 6) + ph) % 16:
                cv[sy, sx] = C[("white", "fog", "steel")[kind]]
    tmp = np.zeros((GY, W), np.uint8)
    sky(tmp, f)
    cv[off:off + GY] = tmp[:max(0, min(GY, H - off))]
    g = GROUND if ground is None else ground
    n = H - (off + GY)
    if n > 0:
        cv[off + GY:] = g[:n]


def caption(cv, lines, t, y=174, h=38, cps=26.0, cols=("white", "yellow")):
    """RPG text box with a typewriter reveal. lines: [(text, t_start), ...]"""
    panel(cv, 6, y, 372, h, fill="ink", edge="fog")
    for i, (s, t0) in enumerate(lines):
        n = int(max(0.0, t - t0) * cps)
        if n <= 0:
            continue
        s2 = s[:n]
        text(cv, s2, 16, y + 4 + 16 * i, c=cols[i % 2], outline=None)
        if n < len(s) and int(t * 8) % 2 == 0:
            rect(cv, 16 + text_w(s2) + 1, y + 6 + 16 * i, 6, 12, cols[i % 2])
    if int(t * 3) % 2 == 0:
        for k in range(3):
            rect(cv, 366 + k, y + h - 8 + k, 5 - 2 * k, 1, "fog")


def monitor(cv, x, y, t, cx):
    """CRT-style radargram that fills in trace by trace as the cart moves."""
    wimg = BSCAN.shape[1]
    box = np.full((54, wimg + 12), T, np.uint8)
    panel(box, 0, 0, wimg + 10, 52, fill="ink", edge="steel")
    small(box, "B-SCAN", 5, 4, c="cyan", outline=None)
    small(box, "30 NS", wimg + 5, 4, c="steel", outline=None, align="r")
    reg = box[14:48, 5:5 + wimg]
    reg[:] = C["night"]
    n = int(np.searchsorted(BS_X, cx, side="right"))
    if n > 0:
        reg[:, :n] = dither_ramp(BSCAN[:, :n], ["night", "teal0", "teal1", "cyan", "ice"])
        if n < wimg:
            reg[:, n - 1] = C["white"]
    if int(t * 4) % 2 == 0:
        rect(box, 34, 5, 3, 3, "red")
    blit(cv, box, x, y)


# ------------------------------------------------------------------ logo
_LOGO = {
    "D": ["######..", "#######.", "##...###", "##....##", "##....##", "##....##", "##....##", "##....##", "##...###",
          "#######.", "######..", "", "", ""],
    "e": ["", "", "", "", ".#####.", "##...##", "##...##", "#######", "##.....", "##...##", ".#####.", "", "", ""],
    "p": ["", "", "", "", "######.", "##...##", "##...##", "##...##", "##...##", "##...##", "######.", "##.....",
          "##.....", "##....."],
    "G": [".######.", "########", "##....##", "##......", "##......", "##..####", "##..####", "##....##", "##....##",
          "########", ".######.", "", "", ""],
    "P": ["#######.", "########", "##....##", "##....##", "########", "#######.", "##......", "##......", "##......",
          "##......", "##......", "", "", ""],
    "R": ["#######.", "########", "##....##", "##....##", "########", "#######.", "##.###..", "##..###.", "##...###",
          "##....##", "##....##", "", "", ""],
}


def _glyph(ch):
    rows = _LOGO[ch]
    w = max(len(r) for r in rows)
    return np.array([[c == "#" for c in r.ljust(w, ".")] for r in rows], bool)


def logo_letters(scale):
    """Bevelled letter sprites for 'DeepGPR' and their x offsets."""
    out, x = [], 0
    for i, ch in enumerate("DeepGPR"):
        m = upscale(_glyph(ch), scale)
        face, hi, lo = ("ice", "white", "cyan") if i < 4 else ("yellow", "white", "orange")
        spr = np.full((m.shape[0] + 3, m.shape[1] + 3), T, np.uint8)
        o = outline_of(m)
        sh = np.zeros_like(spr, bool)
        sh[2:2 + o.shape[0], 2:2 + o.shape[1]] = o[:spr.shape[0] - 2, :spr.shape[1] - 2]
        spr[sh] = C["ink"]
        body = np.zeros_like(spr, bool)
        body[1:1 + m.shape[0], 1:1 + m.shape[1]] = m
        ob = np.zeros_like(spr, bool)
        ob[:o.shape[0], :o.shape[1]] = o[:spr.shape[0], :spr.shape[1]]
        spr[ob] = C["ink"]
        spr[body] = C[face]
        up = np.roll(body, 1, 0)
        dn = np.roll(body, -1, 0)
        spr[body & ~up] = C[hi]
        spr[body & ~dn] = C[lo]
        out.append((spr, x))
        x += m.shape[1] + scale
    return out, x - scale


LOGO4, LOGO4_W = logo_letters(4)
LOGO2, LOGO2_W = logo_letters(2)


def draw_logo(cv, letters, total_w, cx, y, t=None, flash=None):
    x0 = cx - total_w // 2
    for i, (spr, lx) in enumerate(letters):
        dy = 0
        if t is not None:
            tl = t - i * 0.07
            if tl < 0:
                continue
            if tl < 0.22:
                dy = int(-70 * (1 - tl / 0.22) ** 2)
            elif tl < 0.34:
                dy = int(4 * np.sin((tl - 0.22) / 0.12 * np.pi))
            if 0.22 <= tl < 0.30:
                s2 = spr.copy()
                s2[(s2 != T) & (s2 != C["ink"])] = C["white"]
                blit(cv, s2, x0 + lx, y + dy)
                continue
        blit(cv, spr, x0 + lx, y + dy)


# ------------------------------------------------------------------ shots 1+2
def shot_field_scene(t, f):
    cv = canvas()
    sky(cv, f)
    for k in range(3):
        m = int((t - T_SHOT[k]) * MOV_RATE)
        if m >= 0:
            update_reveal(k, min(m, MOV_END[k] - 1))
    mask = _reveal["mask"]
    if t > 13.0:
        mask = mask | (_hx < (t - 13.0) / 0.7)
    g = np.where(mask, GROUND, unknown_ground(f))
    cv[GY:] = g
    reg = cv[GY:]
    for k in range(3):
        mf = (t - T_SHOT[k]) * MOV_RATE
        m = int(mf)
        if 0 <= m < MOV_END[k]:
            fade = min(1.0, (MOV_END[k] - mf) / 8.0)
            a = shot_field(k, m)
            lvl = wave_levels(a, MGAIN[k, m] * fade)
            paint_wave(reg, lvl)
    cx = cart_x(t)
    draw_cart(cv, cx, f, moving=cart_moving(t))
    # emission ticks under the antenna right after each shot
    for k in range(3):
        dtk = t - T_SHOT[k]
        if 0 <= dtk < 0.25:
            small(cv, "PING", int(cx) + 16, GY - 30 - int(dtk * 24), c="yellow")
    if t < 6.0:
        caption(cv, [(tx("s1a"), 0.4), (tx("s1b"), 3.0)], t)
    else:
        caption(cv, [(tx("s2a"), 6.2), (tx("s2b"), 8.4)], t)
        my = int(-60 + 66 * ease((t - 6.0) / 0.5))
        monitor(cv, 272, my, t, cx)
    return cv


# ------------------------------------------------------------------ shot 3: logo
def shot_logo(t, f):
    cv = canvas()
    off = int(106 * ease(t / 0.6))
    world(cv, f, off)
    draw_cart(cv, 304, f, moving=False, ground=off + GY)
    if t > 0.5:
        tl = t - 0.5
        # one radar ring expanding behind the logo
        r = tl * 150.0
        if r < 260:
            d = np.hypot(_xx - 192, (_yy - 58) * 1.6)
            ring = (np.abs(d - r) < 5) & (_yy < off + GY - 14) & (BAYER_FULL < 0.5 * (1 - r / 260))
            cv[ring] = C["slate"]
        draw_logo(cv, LOGO4, LOGO4_W, 192, 28, t=tl)
    if t > 1.35:
        s = tx("s3a")
        n = int((t - 1.35) * 30)
        text(cv, s[:n], 192 - text_w(s) // 2, 92, c="white")
    if t > 2.0:
        small(cv, "PYTORCH x CUDA x C/OPENMP", 192, 113, c="cyan", align="c")
    if t > 2.3:
        cmd = "$ pip install DeepGPR"
        wbox = text_w(cmd) + 22
        panel(cv, 192 - wbox // 2, 126, wbox, 24, fill="ink", edge="lime")
        n = int((t - 2.45) * 22)
        text(cv, cmd[:max(0, n)], 192 - wbox // 2 + 8, 130, c="lime", outline=None)
        if n < len(cmd) or int(t * 4) % 2 == 0:
            rect(cv, 192 - wbox // 2 + 9 + text_w(cmd[:max(0, n)]), 132, 6, 12, "lime")
    return cv


# ------------------------------------------------------------------ model panels
EPS_RAMP = ["sand", "soil4", "soil3", "soil2", "soil1", "soil0"]
SIG_RAMP = ["ice", "cyan", "teal1", "teal0", "navy", "night"]
FW_SX = FW["sxs"]


def _acq(img):
    """Mark the acquisition: sources along the top, receivers along the bottom."""
    img[-2, 1:127:3] = C["fog"]
    for x in FW_SX:
        img[0:2, x] = C["yellow"]
        img[0:2, x + 1] = C["ink"]
        img[0:2, x - 1] = C["ink"]
    return img


def model_img(m, lo, hi, ramp):
    """(128, 72) model -> indexed (72, 128) image with the acquisition overlaid."""
    return _acq(dither_ramp((m.T - lo) / (hi - lo), ramp))


def signed_img(g):
    g2 = g.T / (np.percentile(np.abs(g[:, 6:-6]), 98.0) + 1e-30)
    img = np.full(g2.shape, C["navy"], np.uint8)
    paint_wave(img, wave_levels(g2 * 0.9, 1.0))
    return _acq(img)


ELO, EHI, SLO, SHI = 1.5, 8.5, 0.0, 14.0


def put_panel(cv, img, x, y, label=None, lc="white", edge="fog"):
    h, w = img.shape
    rect(cv, x + 1, y + 1, w + 2, h + 2, "ink")
    frame(cv, x - 1, y - 1, w + 2, h + 2, edge)
    cv[y:y + h, x:x + w] = img
    if label:
        wl = small_w(label) + 6
        rect(cv, x, y, wl, 11, "ink")
        small(cv, label, x + 3, y + 2, c=lc, outline=None)


# ------------------------------------------------------------------ shot 4: autograd
CODE = [
    ("import DeepGPR", 0.5, "fog"),
    ("d = DeepGPR.compute(...)[-1]", 1.3, "white"),
    ("loss = (d-d_obs).abs().sum()", 3.0, "white"),
    ("loss.backward()", 4.0, "yellow"),
]
SNAP = FW["snap"].astype(np.float32)        # (nt/8, 128, 72)
_sg = 1.0 / np.maximum(np.maximum.accumulate(np.percentile(np.abs(SNAP), 99.5, axis=(1, 2))[::-1])[::-1], 1e-9)


# frame at which the wavefront reaches the receiver line (bottom of the model)
SNAP_K = int(np.abs(SNAP[:, :, -4]).max(1).argmax()) + 4


def shot_autograd(t, f):
    cv = canvas()
    night_bg(cv, f)
    text(cv, tx("s4t"), 8, 6, c="yellow")
    small(cv, "AUTOGRAD", 376, 11, c="steel", align="r")
    # terminal
    panel(cv, 6, 28, 242, 156, fill="ink", edge="fog")
    rect(cv, 7, 29, 240, 11, "night")
    for i, col in enumerate(("red", "yellow", "lime")):
        rect(cv, 12 + 7 * i, 32, 4, 5, col)
    small(cv, "FWI.PY", 126, 31, c="steel", outline=None, align="c")
    cps = 30.0
    for i, (line, t0, col) in enumerate(CODE):
        n = int((t - t0) * cps)
        if n <= 0:
            continue
        y = 46 + 18 * i
        small(cv, str(i + 1), 10, y + 5, c="slate", outline=None)
        text(cv, line[:n], 19, y, c=col, outline=None)
        if n < len(line):
            rect(cv, 19 + text_w(line[:n]) + 1, y + 2, 6, 12, col)
    t_bw = 4.0 + len(CODE[3][0]) / cps + 0.15
    outs = [("eps_r.grad", "dL/deR"), ("sigma.grad", "dL/ds"), ("source.grad", "dL/dSRC")]
    for i, (name, _) in enumerate(outs):
        ta = t_bw + 0.55 + 0.35 * i
        if t > ta:
            y = 46 + 18 * (4 + i) + 4
            text(cv, ">", 12, y, c="cyan", outline=None)
            text(cv, name, 22, y, c="cyan", outline=None)
            rect(cv, 22 + 96, y + 7, 60, 1, "teal0")
            small(cv, "OK", 22 + 164, y + 5, c="lime", outline=None)
    # right column: forward wavefield, then the two gradients
    x0, ya, yb = 252, 30, 112
    e0 = model_img(FW["E"][0], ELO, EHI, EPS_RAMP)
    if t < t_bw:
        img = e0.copy()
        tf = t - (1.3 + len(CODE[1][0]) / cps)
        if tf > 0:
            m = int(tf * 22)
            if m < SNAP_K:
                lvl = wave_levels(SNAP[m].T, _sg[m] * 0.9)
                paint_wave(img, lvl)
            put_panel(cv, img, x0, ya, "FORWARD  EZ", "yellow")
        else:
            put_panel(cv, img, x0, ya, "MODEL  eR", "fog")
        blank = np.full((72, 128), C["ink"], np.uint8)
        blank[BAYER_FULL[:72, :128] < 0.12] = C["night"]
        put_panel(cv, blank, x0, yb, None, edge="slate")
        small(cv, "?", x0 + 62, yb + 32, c="slate", outline=None)
    else:
        p = (t - t_bw) / 0.9
        ge = signed_img(FW["g_eps"])
        gs = signed_img(FW["g_sig"])
        blank = np.full((72, 128), C["ink"], np.uint8)
        # the adjoint sweeps back from the receivers: reveal top-down
        rows = np.arange(72)[:, None] * np.ones((1, 128))
        m1 = rows < p * 90 - 10 * BAYER_FULL[:72, :128] * 2
        m2 = rows < (p - 0.45) * 90 - 10 * BAYER_FULL[:72, :128] * 2
        put_panel(cv, np.where(m1, ge, e0), x0, ya, "dL/deR", "yellow")
        put_panel(cv, np.where(m2, gs, blank), x0, yb, "dL/ds", "cyan")
    caption(cv, [(tx("s4a"), 0.6)], t, y=190, h=22)
    return cv


# ------------------------------------------------------------------ shot 5: FWI
NIT = FW["E"].shape[0] - 1
ET_IMG = model_img(FW["Et"], ELO, EHI, EPS_RAMP)
ST_IMG = model_img(FW["St"], SLO, SHI, SIG_RAMP)
LOSS = FW["loss"]
N1 = int(FW["n1"])
_SPARK = [(int(_rng.integers(140, 268)), int(_rng.integers(28, 178)), int(_rng.integers(0, 20))) for _ in range(26)]


def shot_fwi(t, f):
    cv = canvas()
    night_bg(cv, f)
    text(cv, tx("s5t"), 8, 5, c="yellow")
    prog = min(1.0, max(0.0, (t - 0.8) / 6.4))
    it = int(round(prog * NIT))
    xa, xb, ya, yb = 7, 141, 28, 106
    put_panel(cv, ET_IMG, xa, ya, "eR", "sand")
    put_panel(cv, ST_IMG, xa, yb, "s  MS/M", "ice")
    put_panel(cv, model_img(FW["E"][it], ELO, EHI, EPS_RAMP), xb, ya, "eR", "sand", edge="yellow")
    put_panel(cv, model_img(FW["S"][it], SLO, SHI, SIG_RAMP), xb, yb, "s  MS/M", "ice", edge="yellow")
    for x, k in ((xa, "true"), (xb, "inv")):
        s = tx(k)
        if LANG == "zh":
            rect(cv, x + 128 - 36, ya, 36, 18, "ink")
            text(cv, s, x + 128 - 34, ya + 1, c="white" if k == "true" else "yellow", outline=None)
        else:
            rect(cv, x + 128 - small_w(s) - 6, ya, small_w(s) + 6, 11, "ink")
            small(cv, s, x + 128 - 3, ya + 2, c="white" if k == "true" else "yellow", outline=None, align="r")
    # loss curve
    lx, ly, lw, lh = 276, 28, 102, 72
    panel(cv, lx - 1, ly - 1, lw + 2, lh + 2, fill="ink", edge="fog", shadow=True)
    small(cv, "LOSS", lx + 4, ly + 3, c="red", outline=None)
    small(cv, "LOG", lx + lw - 4, ly + 3, c="slate", outline=None, align="r")
    ll = np.log10(LOSS)
    lo, hi = ll.min() - 0.15, ll.max() + 0.1
    for gy in range(4):
        yy = ly + 14 + gy * 18
        cv[yy, lx + 3:lx + lw - 3:3] = C["night"]
    px = lambda i: lx + 5 + int(i * (lw - 11) / max(1, NIT - 1))
    py = lambda v: ly + lh - 5 - int((v - lo) / (hi - lo) * (lh - 20))
    sb = N1
    cv[ly + 12:ly + lh - 3:2, px(sb)] = C["slate"]
    for i in range(min(it, NIT)):
        x1, y1 = px(i), py(ll[i])
        if i > 0:
            x0_, y0_ = px(i - 1), py(ll[i - 1])
            n = max(abs(x1 - x0_), abs(y1 - y0_), 1)
            for s in range(n + 1):
                cv[y0_ + (y1 - y0_) * s // n, x0_ + (x1 - x0_) * s // n] = C["red"]
        if i == it - 1:
            rect(cv, x1 - 1, y1 - 1, 3, 3, "yellow")
    # status box
    sx, sy = 276, 106
    panel(cv, sx - 1, sy - 1, 104, 74, fill="ink", edge="fog")
    small(cv, "ITER", sx + 5, sy + 5, c="steel", outline=None)
    small(cv, "%02d/%02d" % (it, NIT), sx + 97, sy + 5, c="white", outline=None, align="r")
    rect(cv, sx + 5, sy + 16, 92, 5, "night")
    rect(cv, sx + 5, sy + 16, int(92 * it / NIT), 5, "lime")
    stage2 = it > sb
    small(cv, "UPDATE", sx + 5, sy + 27, c="steel", outline=None)
    small(cv, "eR", sx + 62, sy + 27, c="sand", outline=None)
    small(cv, "+ s", sx + 97, sy + 27, c="ice" if stage2 else "slate", outline=None, align="r")
    small(cv, "L1 MISFIT + TV", sx + 5, sy + 39, c="steel", outline=None)
    small(cv, "%d SHOTS  128 RX" % len(FW_SX), sx + 5, sy + 51, c="steel", outline=None)
    if it >= NIT:
        small(cv, "DONE", sx + 5, sy + 63, c="lime", outline=None)
        for (qx, qy, ph) in _SPARK:
            k = (f + ph) % 20
            if k < 4:
                cv[qy, qx] = C["white"]
                if k in (1, 2):
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        cv[qy + dy, qx + dx] = C["yellow"]
    else:
        small(cv, "ADAM + ADJOINT", sx + 5, sy + 63, c="slate", outline=None)
    caption(cv, [(tx("s5a"), 0.5)], t, y=188, h=24)
    return cv


# ------------------------------------------------------------------ shot 6: scale
NC = 56
_ci = (np.arange(NC) * CU["top"].shape[1] / NC).astype(int)
CTOP = CU["top"][:, _ci][:, :, _ci]
CXZ = CU["xz"][:, _ci][:, :, _ci]
CYZ = CU["yz"][:, _ci][:, :, _ci]
CEX = CU["eps_xz"][_ci][:, _ci]
CEY = CU["eps_yz"][_ci][:, _ci]
_cg = 1.0 / np.maximum(np.maximum.accumulate(np.percentile(np.abs(CU["xz"]), 99.5, axis=(1, 2))[::-1])[::-1], 1e-9)
CUBE_A, CUBE_K = 14, 72      # frames from 2 ns to 12 ns of the 3D run
GPU = sprite("""
.kkkkkkkkkkkkkkkkkkkkkk.
kggggggggggggggggggggggk
kgkkkkkkkkgggkkkkkkkkggk
kgkssssssskgkssssssskggk
kgksskkksskgksskkksskggk
kgkskfffkskgkskfffkskggk
kgkskfkfkskgkskfkfkskggk
kgkskfffkskgkskfffkskggk
kgksskkksskgksskkksskggk
kgkssssssskgkssssssskggk
kgkkkkkkkkgggkkkkkkkkggk
kggggggggggggggggggggggk
.kkkkkkkkkkkkkkkkkkkkkk.
...yy.yy.yy.yy.yy.yy....
""", {"k": "ink", "g": "green0", "s": "slate", "f": "fog", "y": "yellow"})
CPU = sprite("""
..y.y.y.y.y.y..
.kkkkkkkkkkkkk.
ykssssssssssskyy
.ksfffffffffsk.
yksfkkkkkkkfskyy
.ksfkcccckkfsk.
yksfkckkckkfskyy
.ksfkcccckkfsk.
yksfkkkkkkkfskyy
.ksfffffffffsk.
ykssssssssssskyy
.kkkkkkkkkkkkk.
..y.y.y.y.y.y..
""", {"k": "ink", "s": "slate", "f": "fog", "c": "cyan", "y": "yellow"})


def iso_cube(m):
    """Isometric cube: top = slice just below the surface, sides = centre cuts (3D DeepGPR run)."""
    N = NC
    spr = np.full((2 * N + 2, 2 * N), T, np.uint8)
    g = _cg[m] * 0.9
    air = int(round(CU["air"] * N / 72))

    def mat(eps):
        img = np.full(eps.shape, C["soil2"], np.uint8)
        img[eps > 9] = C["soil0"]
        img[eps < 2] = C["ink"]
        return img

    left = mat(CEX).T            # (z, x)
    right = mat(CEY).T           # (z, y)
    left[:air] = C["night"]
    right[:air] = C["night"]
    paint_wave(left, wave_levels(CXZ[m].T, g))
    paint_wave(right, wave_levels(CYZ[m].T, g))
    right[(np.add.outer(np.arange(N), np.arange(N)) % 2 == 0) & (right == C["soil2"])] = C["soil1"]
    top = np.full((N, N), C["green0"], np.uint8)
    hh = hash2(*np.meshgrid(np.arange(N), np.arange(N)), 4)
    top[hh < 0.12] = C["green1"]
    paint_wave(top, wave_levels(CTOP[m], g * 0.8))
    py, px = np.mgrid[0:N, 0:2 * N]
    i = py + (px - N) // 2
    j = py - (px - N + 1) // 2
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    spr[:N][ok] = top[np.clip(i, 0, N - 1), np.clip(j, 0, N - 1)][ok]
    for col in range(N):
        y0 = N // 2 + col // 2
        spr[y0 + 1:y0 + 1 + N, col] = left[:, col]
        col2 = N + col
        y1 = N - (col + 1) // 2
        spr[y1 + 1:y1 + 1 + N, col2] = right[:, N - 1 - col]
    # ink silhouette edges
    filled = spr != T
    edge = filled & ~(np.roll(filled, 1, 0) & np.roll(filled, -1, 0) & np.roll(filled, 1, 1) & np.roll(filled, -1, 1))
    spr[edge] = C["ink"]
    for col in range(N):                      # inner fold lines
        spr[N // 2 + col // 2 + 1, col] = C["ink"]
        spr[N - (col + 1) // 2 + 1, N + col] = C["ink"]
    spr[N + 1:2 * N + 1, N] = C["ink"]
    return spr


def shot_scale(t, f):
    cv = canvas()
    night_bg(cv, f)
    text(cv, tx("s6t"), 8, 5, c="yellow")
    m = CUBE_A + int(t * 12) % CUBE_K
    blit(cv, iso_cube(m), 12, 32)
    small(cv, "3D FDTD", 68, 152, c="cyan", align="c")
    small(cv, "REAL DEEPGPR RUN", 68, 164, c="steel", align="c")
    # memory bars
    x0, y0 = 138, 30
    panel(cv, x0, y0, 240, 74, fill="ink", edge="fog")
    if LANG == "zh":
        text(cv, tx("s6m"), x0 + 8, y0 + 4, c="white", outline=None)
        small(cv, "WAVEFIELD HISTORY", x0 + 232, y0 + 9, c="steel", outline=None, align="r")
    else:
        small(cv, "WAVEFIELD HISTORY MEMORY", x0 + 8, y0 + 8, c="white", outline=None)
    bars = [("FP32  E+R", 1.0, "100%", "red", 0.5), ("FP32  E-ONLY", 0.5, "50%", "yellow", 1.3),
            ("INT8  BLOCKS", 0.25, "~25%", "lime", 2.1)]
    for i, (name, frac, lab, col, t0) in enumerate(bars):
        y = y0 + 24 + 16 * i
        if t < t0:
            continue
        small(cv, name, x0 + 8, y + 2, c="fog", outline=None)
        bx, bw = x0 + 84, 112
        rect(cv, bx, y, bw, 10, "night")
        p = ease((t - t0) / 0.5)
        wfill = int(bw * (1.0 + (frac - 1.0) * p)) if i else int(bw * p)
        for seg in range(0, wfill, 4):
            rect(cv, bx + seg, y + 1, 3, 8, col)
        small(cv, lab, x0 + 232, y + 2, c=col, outline=None, align="r")
    # feature chips
    chips = tx("chips")
    for i, s in enumerate(chips):
        t0 = 3.0 + 0.35 * i
        if t < t0:
            continue
        cx_, cy_ = 138 + (i % 2) * 122, 110 + (i // 2) * 24
        dy = int(6 * (1 - ease((t - t0) / 0.25)))
        panel(cv, cx_, cy_ + dy, 118, 20, fill="navy", edge="cyan")
        rect(cv, cx_ + 6, cy_ + dy + 7, 6, 6, "lime")
        rect(cv, cx_ + 7, cy_ + dy + 8, 4, 4, "ink")
        rect(cv, cx_ + 8, cy_ + dy + 9, 2, 2, "lime")
        if LANG == "zh":
            text(cv, s, cx_ + 18, cy_ + dy + 2, c="white", outline=None)
        else:
            small(cv, s.upper(), cx_ + 18, cy_ + dy + 7, c="white", outline=None)
    # backends
    if t > 4.6:
        p = ease((t - 4.6) / 0.3)
        yb = 160 + int(8 * (1 - p))
        blit(cv, GPU, 150, yb)
        small(cv, "CUDA GPU", 180, yb + 3, c="lime", outline=None)
    if t > 5.0:
        p = ease((t - 5.0) / 0.3)
        yb = 160 + int(8 * (1 - p))
        blit(cv, CPU, 266, yb)
        small(cv, "C/OPENMP CPU", 288, yb + 3, c="cyan", outline=None)
    caption(cv, [(tx("s6a"), 0.5)], t, y=188, h=24)
    return cv


# ------------------------------------------------------------------ shot 7: outro
def shot_outro(t, f):
    cv = canvas()
    off = 106
    world(cv, f, off)
    cx = 30 + 60 * t
    draw_cart(cv, cx, f, moving=True, ground=off + GY)
    draw_logo(cv, LOGO4, LOGO4_W, 192, 8, t=t - 0.2)
    if t > 0.9:
        s = tx("s7b")
        n = int((t - 0.9) * 16)
        text(cv, s[:n], 192 - text_w(s) // 2, 68, c="yellow")
    if t > 1.5:
        cmd = "$ pip install DeepGPR"
        wbox = text_w(cmd) + 20
        panel(cv, 192 - wbox // 2, 88, wbox, 24, fill="ink", edge="lime")
        text(cv, cmd, 192 - wbox // 2 + 10, 92, c="lime", outline=None)
    if t > 2.0:
        text(cv, "github.com/songc0a/DeepGPR", 192, 115, c="white", align="c")
    if t > 2.5:
        text(cv, tx("s7a"), 192, 132, c="fog", align="c")
    if t > 3.0:
        panel(cv, 6, 192, 372, 19, fill="ink", edge="fog")
        small(cv, "CITE", 14, 198, c="yellow", outline=None)
        small(cv, "LIU ET AL., COMPUTERS & GEOSCIENCES (2026) 106101", 370, 198, c="fog", outline=None, align="r")
    return cv


def check_fit():
    assert all(len(c[0]) <= 28 for c in CODE)
    for k in ("s1a", "s1b", "s2a", "s2b", "s4a", "s5a", "s6a"):
        for lang in ("zh", "en"):
            assert text_w(TXT[k][lang]) <= 346, (k, lang, text_w(TXT[k][lang]))


check_fit()


# ------------------------------------------------------------------ timeline
SHOTS = [(0.0, 14.0, shot_field_scene), (14.0, 18.0, shot_logo), (18.0, 26.0, shot_autograd),
         (26.0, 36.0, shot_fwi), (36.0, 44.0, shot_scale), (44.0, 50.0, shot_outro)]
DUR = 50.0
XF = 0.3


def frame_at(f):
    t = f / FPS
    for i, (a, b, fn) in enumerate(SHOTS):
        if a <= t < b:
            cv = fn(t - a, f)
            if i > 1 and t - a < XF:           # dither-dissolve from the previous shot's last frame
                pa, pb, pfn = SHOTS[i - 1]
                prev = pfn(pb - pa - 1.0 / FPS, f)
                cv = dissolve(prev, cv, (t - a) / XF)
            if t > DUR - 0.8:
                cv = dissolve(cv, canvas("ink"), (t - (DUR - 0.8)) / 0.7)
            return cv
    return canvas("ink")


if __name__ == "__main__":
    import subprocess

    from PIL import Image

    mode = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] in ("zh", "en"):
        LANG = sys.argv[2]
    if mode == "stills":
        times = [float(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [1.5, 4.5, 8.5, 12.5, 15.5, 17.5, 21, 24.5, 28, 34, 38, 42.5, 46, 48.5]
        for t in times:
            if t < 14:   # reveal state needs the sequential history
                for ff in range(0, int(t * FPS), 2):
                    for k in range(3):
                        mm = int((ff / FPS - T_SHOT[k]) * MOV_RATE)
                        if mm >= 0:
                            update_reveal(k, min(mm, MOV_END[k] - 1))
            Image.fromarray(to_rgb(frame_at(int(t * FPS)), 3)).save("still_%s_%05.1f.png" % (LANG, t))
    elif mode == "video":
        out = sys.argv[3]
        n = int(DUR * FPS)
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", "%dx%d" % (W * SCALE, H * SCALE), "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                              "-tune", "animation", out], stdin=subprocess.PIPE)
        for f in range(n):
            p.stdin.write(to_rgb(frame_at(f)).tobytes())
            if f % 150 == 0:
                print("frame", f, flush=True)
        p.stdin.close()
        p.wait()
