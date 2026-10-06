"""Pixel-art assets for the DeepGPR website, drawn with the same palette and sprites as the promo film."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import render as R
from gfx import *  # noqa: F401,F403
from world import CART, CHEST, GROUND, GY, PINE, draw_cart, paint_wave, sky, wave_levels

OUT = sys.argv[1]
PX = os.path.join(OUT, "assets", "images", "px")
os.makedirs(PX, exist_ok=True)
os.makedirs(os.path.join(OUT, "assets", "video"), exist_ok=True)
PAL_FLAT = list(LUT.reshape(-1)) + [0] * (768 - LUT.size)


def save_png(idx, name, transparent=None):
    """Indexed PNG with the film palette (index `transparent` becomes alpha 0)."""
    im = Image.fromarray(idx.astype(np.uint8), "P")
    im.putpalette(PAL_FLAT)
    kw = {}
    if transparent is not None:
        kw["transparency"] = transparent
    im.save(os.path.join(PX, name), optimize=True, **kw)
    print(name, idx.shape, os.path.getsize(os.path.join(PX, name)))


TR = len(LUT)  # spare palette slot used as transparent

# ---------------------------------------------------------------- sky gradient (8 x 160)
h = 160
yy = np.arange(h)[:, None] * np.ones((1, 8))
v = np.clip(yy / (h - 1), 0, 1) ** 1.25
save_png(dither_ramp(v, ["ink", "ink", "night", "night", "navy", "navy", "slate"]), "sky.png")

# ---------------------------------------------------------------- star tile (256 x 256, transparent)
rng = np.random.default_rng(11)
st = np.full((256, 256), TR, np.uint8)
for _ in range(150):
    x, y, k = int(rng.integers(1, 255)), int(rng.integers(1, 255)), int(rng.integers(0, 10))
    st[y, x] = C[("white", "fog", "fog", "steel", "steel", "steel", "slate", "slate", "slate", "slate")[k]]
    if k == 0 and rng.random() < 0.5:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            st[y + dy, x + dx] = C["steel"]
save_png(st, "stars.png", transparent=TR)

# ---------------------------------------------------------------- horizon strip (384 x 40, tileable)
Wt = 384
x = np.arange(Wt)
hz = np.full((40, Wt), TR, np.uint8)
h1 = (16 + 6 * np.sin(2 * np.pi * 2 * x / Wt) + 3 * np.sin(2 * np.pi * 5 * x / Wt + 2)).astype(int)
h2 = (25 + 4 * np.sin(2 * np.pi * 3 * x / Wt + 1) + 2 * np.sin(2 * np.pi * 9 * x / Wt)).astype(int)
ry = np.arange(40)[:, None]
hz[ry >= h1[None, :]] = C["navy"]
hz[ry >= h2[None, :]] = C["night"]
for px_ in rng.choice(np.arange(0, Wt, 6), 30, replace=False):
    dy = int(rng.integers(0, 3))
    for j in range(PINE.shape[0]):
        for i in range(PINE.shape[1]):
            if PINE[j, i] != T:
                hz[40 - 8 + j - dy + (dy if j == 7 else 0) if False else min(39, 40 - 8 + j - dy), (px_ + i) % Wt] = PINE[j, i]
save_png(hz, "horizon.png", transparent=TR)

# ---------------------------------------------------------------- ground strip (384 x 56, tileable)
D = 56
gy, gx = np.mgrid[0:D, 0:Wt]
hh = hash2(gx, gy, 3)
hh2 = hash2(gx // 2, gy, 9)
bound = (38 + 4 * np.sin(2 * np.pi * 2 * x / Wt) + 2 * np.sin(2 * np.pi * 7 * x / Wt + 1)).astype(int)
g = np.full((D, Wt), C["soil2"], np.uint8)
g[hh < 0.07] = C["soil3"]
g[hh2 > 0.94] = C["soil1"]
clay = gy >= bound[None, :]
g[clay] = C["soil1"]
g[clay & (hh < 0.07)] = C["soil2"]
g[clay & (hh2 > 0.94)] = C["soil0"]
g[(gy == bound[None, :])] = C["soil0"]
g[0] = C["green1"]
g[0][hh[0] > 0.8] = C["lime"]
g[1] = np.where(hh[1] < 0.55, C["green0"], C["green1"])
g[2][hh[2] < 0.25] = C["green0"]
save_png(g, "surface.png")
blit(g, GROUND[20:42, 70:92], 60, 14)        # the metal pipe from the film
blit(g, GROUND[16:36, 234:258], 190, 12)     # a boulder
blit(g, CHEST, 300, 20)
save_png(g, "ground.png")
save_png(g[:3], "grass.png")

# ---------------------------------------------------------------- soil textures (64 x 64, tileable)
ty, tx_ = np.mgrid[0:64, 0:64]
t1 = hash2(tx_, ty, 31)
t2 = hash2(tx_ // 2, ty, 37)
for name, (base, hi, lo) in (("soil.png", ("soil2", "soil3", "soil1")), ("clay.png", ("soil1", "soil2", "soil0")),
                             ("deep.png", ("soil0", "soil1", "ink"))):
    s = np.full((64, 64), C[base], np.uint8)
    s[t1 < 0.05] = C[hi]
    s[t2 > 0.955] = C[lo]
    save_png(s, name)

# ---------------------------------------------------------------- logo + favicon (SVG, crisp at any size)


def svg_from_index(idx, scale=1):
    hgt, wid = idx.shape
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" shape-rendering="crispEdges">' % (wid, hgt)]
    by_col = {}
    for j in range(hgt):
        i = 0
        while i < wid:
            c = idx[j, i]
            if c == T:
                i += 1
                continue
            k = i
            while k < wid and idx[j, k] == c:
                k += 1
            by_col.setdefault(int(c), []).append("M%d %dh%dv1h-%dz" % (i, j, k - i, k - i))
            i = k
    for c, ds in by_col.items():
        parts.append('<path fill="#%02x%02x%02x" d="%s"/>' % (*LUT[c], "".join(ds)))
    parts.append("</svg>")
    return "".join(parts)


letters, total = R.logo_letters(4)
lg = np.full((64, total + 8), T, np.uint8)
letters1, _ = R.logo_letters(1)
for spr, lx in letters:
    blit(lg, spr, lx, 0)
rows = np.where((lg != T).any(1))[0]
lg = lg[rows.min():rows.max() + 1]
open(os.path.join(PX, "logo.svg"), "w").write(svg_from_index(lg))
print("logo.svg", lg.shape)

fav = np.full((16, 16), C["ink"], np.uint8)
dspr = letters1[0][0]
dm = dspr[:13, :11]
blit(fav, dm, 3, 3)
for (ax, ay) in ((2, 2), (3, 1), (4, 1), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 1), (11, 1), (12, 2)):
    fav[ay, ax] = C["orange"]
fav[0, 0] = fav[0, 15] = fav[15, 0] = fav[15, 15] = T
open(os.path.join(OUT, "assets", "images", "favicon.svg"), "w").write(
    svg_from_index(fav).replace("<svg ", '<svg role="img" aria-label="DeepGPR" '))

# small cart icon for the header brand mark
ic = np.full((18, 26), T, np.uint8)
blit(ic, CART, 0, 0)
cols = np.where((ic != T).any(0))[0]
open(os.path.join(PX, "cart.svg"), "w").write(svg_from_index(ic[:, cols.min():cols.max() + 1]))

# ---------------------------------------------------------------- hero loop (256 x 192)
FPS_H, DUR_H = 12.5, 8.0
X0, HH = 64, 192


def hero_frame(i):
    t = i / FPS_H
    f = int(t * 30)
    cv = canvas()
    sky(cv, f)
    cv[GY:] = GROUND
    mf = (t - 2.0) * 15.0
    m = int(mf)
    if 0 <= m < 46:
        fade = min(1.0, (46 - mf) / 8.0)
        paint_wave(cv[GY:], wave_levels(R.shot_field(1, m), R.MGAIN[1, m] * fade))
    if t < 1.8:
        cx, mv = 40 + 152 * ease(t / 1.8), True
    elif t < 5.4:
        cx, mv = 192, False
    else:
        cx, mv = 192 + 170 * ((t - 5.4) / 2.6), True
    draw_cart(cv, cx, f, moving=mv)
    if 2.0 <= t < 2.3:
        small(cv, "PING", int(cx) + 16, GY - 30 - int((t - 2.0) * 24), c="yellow")
    return cv[:HH, X0:X0 + 256]


frames = [hero_frame(i) for i in range(int(FPS_H * DUR_H))]
ims = []
for fr in frames:
    im = Image.fromarray(fr, "P")
    im.putpalette(PAL_FLAT)
    ims.append(im)
ims[0].save(os.path.join(PX, "hero-survey.gif"), save_all=True, append_images=ims[1:], duration=80, loop=0,
            optimize=False, disposal=1)
print("hero-survey.gif", os.path.getsize(os.path.join(PX, "hero-survey.gif")))
save_png(frames[int(3.6 * FPS_H)], "hero-survey.png")

# ---------------------------------------------------------------- film: lossless master, poster, og image
if "--video" in sys.argv:
    R.LANG = "en"
    master = "web_master.mkv"
    n = int(R.DUR * FPS)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "%dx%d" % (W * SCALE, H * SCALE), "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p", master],
                         stdin=subprocess.PIPE)
    for f in range(n):
        fr = R.frame_at(f)
        p.stdin.write(to_rgb(fr).tobytes())
        if f == int(34.6 * FPS):
            Image.fromarray(to_rgb(fr)).save(os.path.join(OUT, "assets", "video", "deepgpr-promo-poster.jpg"),
                                             quality=93, subsampling=0)
        if f == int(17.7 * FPS):
            og = np.zeros((630, 1200, 3), np.uint8)
            og[:] = LUT[C["ink"]]
            big = to_rgb(fr, 3)                       # 1152 x 648
            og[:, 24:24 + 1152] = big[9:9 + 630]
            Image.fromarray(og).save(os.path.join(OUT, "assets", "images", "og.png"), optimize=True)
    p.stdin.close()
    p.wait()
    print("master done")
