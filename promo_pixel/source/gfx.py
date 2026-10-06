"""Pixel-art toolkit: one fixed palette, indexed canvases, bitmap fonts, sprites."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SCALE = 384, 216, 30, 5

_PAL = [
    ("ink", "0d0b1a"), ("night", "171a3a"), ("navy", "252d5c"), ("slate", "3b4a7a"),
    ("steel", "6f84a8"), ("fog", "a9bdd1"), ("white", "f2f0e4"),
    ("soil0", "2a1a26"), ("soil1", "4a2b33"), ("soil2", "70413a"), ("soil3", "9c6447"),
    ("soil4", "c9925c"), ("sand", "e8c987"),
    ("red", "d9483b"), ("orange", "f08a3c"), ("yellow", "f7d354"),
    ("teal0", "12505e"), ("teal1", "1f8a8a"), ("cyan", "4fd0c8"), ("ice", "b8f2e6"),
    ("green0", "2f6b3f"), ("green1", "5fae4f"), ("lime", "b5e56b"),
    ("magenta", "b0457a"), ("pink", "f08fa8"), ("purple", "5b3a8c"),
]
C = {n: i for i, (n, _) in enumerate(_PAL)}
LUT = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for _, h in _PAL], np.uint8)
T = 255  # transparent index in sprites

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 + 1 / 32.0
BAYER_FULL = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]


def canvas(color="ink"):
    return np.full((H, W), C[color] if isinstance(color, str) else color, np.uint8)


def to_rgb(idx, scale=SCALE):
    rgb = LUT[idx]
    if scale != 1:
        rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    return rgb


def ci(c):
    return C[c] if isinstance(c, str) else int(c)


def rect(cv, x, y, w, h, c):
    x0, y0, x1, y1 = max(0, x), max(0, y), min(cv.shape[1], x + w), min(cv.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        cv[y0:y1, x0:x1] = ci(c)


def frame(cv, x, y, w, h, c):
    rect(cv, x, y, w, 1, c)
    rect(cv, x, y + h - 1, w, 1, c)
    rect(cv, x, y, 1, h, c)
    rect(cv, x + w - 1, y, 1, h, c)


def panel(cv, x, y, w, h, fill="ink", edge="fog", shadow=True):
    """RPG-style box: 1px outline with clipped corners and an inner bevel."""
    if shadow:
        rect(cv, x + 2, y + 2, w, h, "ink")
    rect(cv, x, y, w, h, fill)
    frame(cv, x, y, w, h, edge)
    for (px, py) in ((x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)):
        if 0 <= px < cv.shape[1] and 0 <= py < cv.shape[0]:
            cv[py, px] = ci("ink")
    for (px, py) in ((x + 1, y + 1), (x + w - 2, y + 1), (x + 1, y + h - 2), (x + w - 2, y + h - 2)):
        if 0 <= px < cv.shape[1] and 0 <= py < cv.shape[0]:
            cv[py, px] = ci(edge)


def blit(cv, spr, x, y, flip=False):
    """Draw an indexed sprite (T = transparent) with clipping."""
    if flip:
        spr = spr[:, ::-1]
    h, w = spr.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cv.shape[1], x + w), min(cv.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s != T
    cv[y0:y1, x0:x1][m] = s[m]


def blit_mask(cv, mask, x, y, c):
    h, w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cv.shape[1], x + w), min(cv.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    cv[y0:y1, x0:x1][m] = ci(c)


def sprite(art, legend):
    """ASCII art -> indexed sprite. '.' is transparent."""
    rows = [r for r in art.strip("\n").split("\n")]
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), T, np.uint8)
    for j, r in enumerate(rows):
        for i, ch in enumerate(r):
            if ch != "." and ch != " ":
                out[j, i] = C[legend[ch]]
    return out


def upscale(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


# ------------------------------------------------------------------ fonts
_UNI = ImageFont.truetype("/usr/share/fonts/opentype/unifont/unifont.otf", 16)
_cache = {}


def text_mask(s):
    """Unifont 16px bitmap mask of a string (ASCII 8px wide, CJK 16px wide)."""
    if s in _cache:
        return _cache[s]
    w = int(_UNI.getlength(s)) + 2
    im = Image.new("1", (max(w, 1), 18), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.text((0, 0), s, font=_UNI, fill=1)
    m = np.array(im, bool)
    cols = np.where(m.any(0))[0]
    m = m[:16, :(cols.max() + 1 if len(cols) else 1)]
    _cache[s] = m
    return m


def text_w(s):
    return int(_UNI.getlength(s))


def outline_of(m):
    p = np.pad(m, 1)
    o = np.zeros_like(p)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            o |= np.roll(np.roll(p, dy, 0), dx, 1)
    return o


def text(cv, s, x, y, c="white", outline="ink", align="l", shadow=None):
    """Draw unifont text. align: l / c / r relative to x."""
    m = text_mask(s)
    w = text_w(s)
    if align == "c":
        x = x - w // 2
    elif align == "r":
        x = x - w
    if outline is not None:
        blit_mask(cv, outline_of(m), x - 1, y - 1, outline)
    if shadow is not None:
        blit_mask(cv, m, x + 1, y + 1, shadow)
    blit_mask(cv, m, x, y, c)
    return w


_F5 = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "111 010 010 010 010 010 111", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "010 110 010 010 010 010 111",
    "2": "01110 10001 00001 00110 01000 10000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "000 000 000 000 000 000 000", ".": "0 0 0 0 0 0 1", ",": "00 00 00 00 00 01 10",
    ":": "0 0 1 0 0 1 0", "-": "0000 0000 0000 1111 0000 0000 0000", "+": "00000 00100 00100 11111 00100 00100 00000",
    "/": "00001 00001 00010 00100 01000 10000 10000", "%": "11001 11001 00010 00100 01000 10011 10011",
    "(": "01 10 10 10 10 10 01", ")": "10 01 01 01 01 01 10", "=": "0000 0000 1111 0000 1111 0000 0000",
    "x": "00000 00000 10001 01010 00100 01010 10001", ">": "1000 0100 0010 0001 0010 0100 1000",
    "<": "0001 0010 0100 1000 0100 0010 0001", "!": "1 1 1 1 1 0 1", "?": "01110 10001 00001 00110 00100 00000 00100",
    "&": "01100 10010 10100 01000 10101 10010 01101",
    "_": "00000 00000 00000 00000 00000 00000 11111", "'": "1 1 0 0 0 0 0", "#": "01010 01010 11111 01010 11111 01010 01010",
    "*": "00000 10101 01110 11111 01110 10101 00000", "~": "00000 00000 01000 10101 00010 00000 00000",
    "e": "00000 00000 01110 10001 11111 10000 01110",  # epsilon-ish / small e
    "s": "00000 00000 01111 10010 10010 10010 01100",  # sigma
    "d": "00110 01000 00100 01110 10001 10001 01110",  # partial-d
    "r": "00000 00000 10110 11000 10000 10000 10000",
    "^": "00100 01110 10101 00100 00100 00100 00100",  # up arrow
    "v": "00100 00100 00100 00100 10101 01110 00100",  # down arrow
    "}": "00100 00010 11111 00010 00100 00000 00000",  # right arrow
}
_F5M = {k: np.array([[c == "1" for c in row] for row in v.split()], bool) for k, v in _F5.items()}


def small_w(s):
    return sum(_F5M[ch].shape[1] + 1 for ch in s) - 1


def small(cv, s, x, y, c="white", outline="ink", align="l"):
    """Tiny 5x7 caps font for HUD labels."""
    w = small_w(s)
    if align == "c":
        x -= w // 2
    elif align == "r":
        x -= w
    m = np.zeros((7, w), bool)
    cx = 0
    for ch in s:
        g = _F5M[ch]
        m[:, cx:cx + g.shape[1]] = g
        cx += g.shape[1] + 1
    if outline is not None:
        blit_mask(cv, outline_of(m), x - 1, y - 1, outline)
    blit_mask(cv, m, x, y, c)
    return w


def dither_ramp(v, ramp):
    """Map v in [0, 1] (2D array at frame coords or given bayer) onto a colour ramp with Bayer dithering."""
    n = len(ramp) - 1
    f = np.clip(v, 0, 1) * n
    lo = np.floor(f).astype(int)
    frac = f - lo
    by = np.tile(BAYER, (v.shape[0] // 4 + 1, v.shape[1] // 4 + 1))[:v.shape[0], :v.shape[1]]
    idx = np.clip(lo + (frac > by), 0, n)
    return np.array([ci(c) for c in ramp], np.uint8)[idx]


def dissolve(a, b, p):
    """Bayer dissolve between two indexed frames, p in [0, 1]."""
    if p <= 0:
        return a
    if p >= 1:
        return b
    m = BAYER_FULL < p
    out = a.copy()
    out[m] = b[m]
    return out


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def hash2(x, y, seed=0):
    """Deterministic per-pixel hash in [0, 1)."""
    n = (x.astype(np.int64) * 374761393 + y.astype(np.int64) * 668265263 + seed * 144665) & 0x7FFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0x7FFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65536.0
