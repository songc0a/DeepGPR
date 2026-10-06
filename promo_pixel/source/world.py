"""World art shared by all shots: night sky, the buried scene, sprites, wave overlay."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import models
from gfx import *  # noqa: F401,F403

GY = 64  # first ground row on screen

# ------------------------------------------------------------------ sky
_yy, _xx = np.mgrid[0:H, 0:W]
_rng = np.random.default_rng(7)
STARS = [(int(_rng.integers(0, W)), int(_rng.integers(0, 52)), int(_rng.integers(0, 64)), int(_rng.integers(0, 3)))
         for _ in range(70)]
_hill1 = (50 + 5 * np.sin(np.arange(W) / 37.0) + 3 * np.sin(np.arange(W) / 13.0 + 2)).astype(int)
_hill2 = (56 + 4 * np.sin(np.arange(W) / 23.0 + 1) + 2 * np.sin(np.arange(W) / 7.0)).astype(int)
MOON = sprite("""
....wwwww....
..wwwwwwwss..
.wwwwwwwwwss.
.wwwswwwwwss.
wwwwwwwwwwsss
wwwwwwwswwsss
wwwwwwwwwwsss
wwswwwwwwssss
wwwwwwwwwssss
.wwwwwwwssss.
.swwwwwsssss.
..sssssssss..
....sssss....
""", {"w": "white", "s": "sand"})
PINE = sprite("""
..g..
..g..
.ggg.
.ggg.
ggggg
.ggg.
ggggg
..k..
""", {"g": "teal0", "k": "ink"})
_PINES = [(int(x), int(_rng.integers(0, 3))) for x in _rng.choice(np.arange(4, W - 8), 26, replace=False)]


def sky(cv, f, y1=GY):
    """Night sky down to row y1 (exclusive)."""
    v = np.clip(_yy[:y1] / 70.0, 0, 1) ** 1.3
    cv[:y1] = dither_ramp(v, ["ink", "ink", "night", "night", "navy", "navy", "slate"])
    for (sx, sy, ph, kind) in STARS:
        if sy >= y1:
            continue
        tw = ((f // 6) + ph) % 16
        if tw == 0:
            continue
        col = ("white", "fog", "steel")[kind] if tw > 2 else "steel"
        cv[sy, sx] = C[col]
        if kind == 0 and tw in (7, 8):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= sx + dx < W and 0 <= sy + dy < y1:
                    cv[sy + dy, sx + dx] = C["steel"]
    blit(cv, MOON, 30, 10)
    cols = np.arange(W)
    for hy, col in ((_hill1, "navy"), (_hill2, "night")):
        m = _yy[:y1] >= hy[None, :]
        cv[:y1][m] = C[col]
    for (px, dy) in _PINES:
        blit(cv, PINE, px, y1 - 8 + dy - 1)


# ------------------------------------------------------------------ ground
IDS = models.scene_ids()[:, models.AIR:]          # (384, 152) [x, depth]
CHEST = sprite("""
kkkkkkkkkkkkkkkkk
koooooooyoooooook
koyyyyyykyyyyyyok
kooooookykooooook
kkkkkkkkykkkkkkkk
kooooookkkooooook
koooooooooooooook
koyoooooooooooyok
koooooooooooooook
koooooooooooooook
kkkkkkkkkkkkkkkkk
""", {"k": "ink", "o": "orange", "y": "yellow"})


def _ground():
    ids = IDS.T  # (depth, x)
    D = ids.shape[0]
    y, x = np.mgrid[0:D, 0:W]
    h = hash2(x, y, 3)
    h2 = hash2(x // 2, y, 9)
    g = np.zeros((D, W), np.uint8)
    spec = {1: ("soil2", "soil3", "soil1"), 2: ("soil1", "soil2", "soil0"), 3: ("soil0", "soil1", "teal0")}
    for k, (base, hi, lo) in spec.items():
        m = ids == k
        g[m] = C[base]
        g[m & (h < 0.09)] = C[hi]
        g[m & (h2 > 0.93)] = C[lo]
    # pebbles in the topsoil, strata lines in clay
    g[(ids == 2) & ((y + (x // 9) % 3) % 13 == 0) & (h > 0.35)] = C["soil0"]
    # layer boundaries: 1px darker lip
    for k, col in ((2, "soil0"), (3, "ink")):
        m = (ids == k)
        top = m & ~np.roll(m, 1, 0)
        top[0] = False
        g[top & (np.roll(ids, 1, 0) == k - 1)] = C[col]
    # boulders
    m = ids == 4
    g[m] = C["slate"]
    inner = binary_erosion(m, iterations=1)
    g[m & ~inner] = C["ink"]
    lit = inner & ~np.roll(np.roll(inner, 2, 0), 2, 1)
    g[lit] = C["steel"]
    sh = inner & ~np.roll(np.roll(inner, -2, 0), -2, 1)
    g[sh] = C["navy"]
    g[inner & (h < 0.05)] = C["fog"]
    # cavity
    m = ids == 6
    g[m] = C["ink"]
    inner = binary_erosion(m, iterations=1)
    g[m & ~inner] = C["soil0"]
    g[inner & (h < 0.04)] = C["night"]
    # stalactite-ish drips at the cavity roof
    roof = inner & ~np.roll(inner, 1, 0)
    g[np.roll(roof, 1, 0) & inner & ((x % 5) == 1)] = C["soil0"]
    # metal pipe (round) and chest
    m = (ids == 5) & (x < 150)
    g[m] = C["steel"]
    inner = binary_erosion(m, iterations=1)
    g[m & ~inner] = C["ink"]
    core = binary_erosion(m, iterations=3)
    g[core] = C["slate"]
    g[binary_erosion(m, iterations=4)] = C["ink"]
    lit = inner & ~np.roll(np.roll(inner, 1, 0), 1, 1) & ~core
    g[lit] = C["white"]
    blit(g, CHEST, 296, 54)
    # grass
    g[0] = C["green1"]
    g[1][h[1] < 0.55] = C["green0"]
    g[1][h[1] >= 0.85] = C["green1"]
    g[0][h[0] > 0.8] = C["lime"]
    g[2][(h[2] < 0.25) & (ids[2] == 1)] = C["green0"]
    return g


GROUND = _ground()
_qm = text_mask("?")
_QPOS = [(int(_rng.integers(6, W - 14)), int(_rng.integers(8, 96))) for _ in range(16)]


def unknown_ground(f):
    """The unsurveyed ground: darkness with faint question marks."""
    D = GROUND.shape[0]
    y, x = np.mgrid[0:D, 0:W]
    g = np.full((D, W), C["ink"], np.uint8)
    g[hash2(x, y, 21) < 0.02] = C["night"]
    g[hash2(x, y, 5) < 0.006] = C["soil0"]
    for i, (qx, qy) in enumerate(_QPOS):
        ph = (f // 10 + i * 3) % 12
        if ph < 7:
            blit_mask(g, _qm, qx, qy + (1 if ph in (2, 3) else 0), "navy" if ph in (0, 6) else "slate")
    g[0] = GROUND[0]
    g[1] = GROUND[1]
    return g


# ------------------------------------------------------------------ sprites
CART = sprite("""
..........kkkkkkk.........
.........kcccccck.........
.........kcicccck.........
.........kccccick.........
..........kkkkkkk.........
...kk.......ks............
....ksk.....ks............
......ksk...ks............
........kskkkskkkkkkkkk...
......kkkkoooooooooooook..
......kyyyoooooooooooook..
......kooooooyyyyyooooook.
......kooooooooooooooook..
......kkkkkkkkkkkkkkkkkk..
.......kkk.........kkk....
......ksfsk.......ksfsk...
......kfsfk.......kfsfk...
.......kkk.........kkk....
""", {"k": "ink", "c": "teal1", "i": "ice", "s": "steel", "f": "fog", "o": "orange", "y": "yellow"})
CART2 = CART.copy()
for _wx in (6, 18):  # second wheel frame: swap hub pixels
    blk = CART2[15:17, _wx + 1:_wx + 4].copy()
    CART2[15:17, _wx + 1:_wx + 4] = blk[::-1]
_PL = {"y": "yellow", "o": "orange", "s": "sand", "k": "ink", "t": "teal1", "d": "teal0", "n": "steel", "w": "white"}
_HEAD = """
...yyyy...
..yyyyyy..
.yyyyyyyy.
..sssks...
..sssss...
...sss....
..tttt....
.tttttd...
.ttttttss.
.tttttd...
.tttttd...
..ttttd...
"""
WALK = [sprite(_HEAD + """
..nnnn....
..nnnn....
..nn.nn...
.nn...nn..
.nn...nn..
kkk...kkk.
""", _PL), sprite(_HEAD + """
..nnnn....
..nnnn....
...nnn....
...nnn....
...nnn....
..kkkk....
""", _PL)]
CART_ANT = 14   # antenna centre x offset inside CART


def draw_cart(cv, x, f, moving=True, ground=GY):
    """Cart with its antenna centred at screen x; operator walks behind."""
    cx = int(x) - CART_ANT
    step = (f // 5) % 2 if moving else 1
    blit(cv, WALK[step], cx - 6, ground - 18 - (0 if step else 0))
    blit(cv, CART2 if (moving and (f // 3) % 2) else CART, cx, ground - 18)


# ------------------------------------------------------------------ waves
def wave_levels(a, gain):
    """Signed field -> small integer levels (-2..2) with dithered band edges."""
    v = a * gain
    by = np.tile(BAYER, (a.shape[0] // 4 + 1, a.shape[1] // 4 + 1))[:a.shape[0], :a.shape[1]]
    av = np.abs(v)
    lvl = np.zeros(a.shape, np.int8)
    lvl[av > 0.10 + 0.10 * by] = 1
    lvl[av > 0.42 + 0.12 * by] = 2
    return lvl * np.sign(v).astype(np.int8)


WAVE_COL = {1: "orange", 2: "yellow", -1: "teal1", -2: "cyan"}


def paint_wave(region, lvl):
    for k, col in WAVE_COL.items():
        region[lvl == k] = C[col]
