#!/usr/bin/env python3
"""Render the deterministic DeepGPR signal film and its web deliverables.

Requires Python 3.10+, numpy, Pillow and ffmpeg. No project solver required.
Scientific wavefield graphics are conceptual; the FWI panels are repository
figures. Use --stills first to review the five chapters before encoding.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
from multiprocessing import Pool
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from hologram import draw_hologram

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "website/assets/video"
W, H, FPS, DURATION = 1920, 1080, 30, 32
WHITE = (236, 246, 250)
CYAN = (78, 230, 224)
PINK = (247, 83, 193)
MUTED = (136, 163, 180)
CUTS = (0, 6, 13, 20, 26, 32)
FONT_DIR = Path("/usr/share/fonts/truetype")


@lru_cache(maxsize=50)
def font(size, style="regular"):
    files = {
        "regular": FONT_DIR / "lato/Lato-Regular.ttf",
        "light": FONT_DIR / "lato/Lato-Light.ttf",
        "bold": FONT_DIR / "lato/Lato-Heavy.ttf",
        "black": FONT_DIR / "lato/Lato-Black.ttf",
        "mono": FONT_DIR / "dejavu/DejaVuSansMono.ttf",
    }
    target = files[style]
    if not target.exists():
        target = FONT_DIR / "dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(str(target), size)


def smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


@lru_cache(maxsize=1)
def assets():
    source = ImageOps.fit(Image.open(HERE / "assets/subsurface-keyart.png").convert("RGB"), (W, H), method=Image.Resampling.LANCZOS)
    keyart = Image.new("RGB", (W, H), (2, 6, 12))
    keyart.paste(source.resize((1766, 993), Image.Resampling.LANCZOS), (154, 43))
    # Grade the left side into deep ink while preserving the cinematic geology.
    yy, xx = np.mgrid[:H, :W]
    dark = np.clip(1 - xx / 950, 0, 1) * 0.42
    dark += 0.12 * np.clip((yy - 810) / 270, 0, 1)
    dark += 0.52 * np.clip((185 - yy) / 185, 0, 1)
    keyart = Image.fromarray(np.uint8(np.asarray(keyart) * (1 - dark[:, :, None])))
    x = xx / W
    y = yy / H
    cyan = np.exp(-((x - 0.7) ** 2 / 0.08 + (y - 0.48) ** 2 / 0.14))
    pink = np.exp(-((x - 0.82) ** 2 / 0.06 + (y - 0.78) ** 2 / 0.025))
    a = np.zeros((H, W, 3), dtype=np.float32)
    a[:] = (3, 7, 15)
    a += cyan[:, :, None] * (2, 14, 23)
    a += pink[:, :, None] * (16, 3, 14)
    a += np.random.default_rng(71).normal(0, 0.6, (H, W, 1))
    backdrop = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
    true = Image.open(ROOT / "Fig/2dfwipred.png").convert("RGB")
    initial = Image.open(ROOT / "Fig/2dfwiinit.png").convert("RGB")
    figs = [ImageOps.contain(im, (910, 515), Image.Resampling.LANCZOS) for im in (initial, true)]
    rng = np.random.default_rng(52)
    particles = rng.uniform(0, 1, (80, 4))
    return keyart, backdrop, figs, particles


def text(img, xy, content, size=24, color=WHITE, style="regular", alpha=1, tracking=0):
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size)
    d = ImageDraw.Draw(layer)
    fill = (*color, int(255 * min(1, alpha)))
    if tracking:
        x, y = xy
        for c in content:
            d.text((int(x), int(y)), c, font=font(size, style), fill=fill)
            x += d.textlength(c, font=font(size, style)) + tracking
    else:
        d.text(tuple(int(v) for v in xy), content, font=font(size, style), fill=fill, spacing=int(size * 0.13))
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))


def heading(img, lines, local, size=122, y=336):
    # A quiet upward movement and a left-to-right optical reveal.
    p = smooth(local / 0.95)
    shift = 22 * (1 - p)
    layer = Image.new("RGB", img.size, (0, 0, 0))
    for i, line in enumerate(lines):
        text(layer, (103, y + i * size * 1.08 + shift), line, size, WHITE, "black")
    reveal = Image.new("L", img.size)
    ImageDraw.Draw(reveal).rectangle((90, 240, 100 + int(775 * p), 800), fill=int(255 * p))
    ink = ImageChops.lighter(layer.convert("L"), Image.new("L", img.size))
    # Composite only actual type so the reveal never paints a black rectangle.
    ink = ink.point(lambda x: 255 if x > 0 else 0)
    img.paste(layer, (0, 0), ImageChops.multiply(reveal, ink))


def particles(img, t, strength=1.0):
    d = ImageDraw.Draw(img)
    for px, py, speed, power in assets()[3]:
        x = int(760 + px * 1050 + math.sin(t * 0.19 + py * 9) * 18)
        y = int((160 + py * 740 - t * (5 + speed * 9)) % 820 + 50)
        if 13 <= t < 20 and 890 <= x <= 1834 and 253 <= y <= 883:
            continue
        luminosity = int((0.3 + 0.7 * power) * (0.7 + 0.3 * math.sin(t * 1.4 + px * 13)) * 80 * strength)
        color = (luminosity // 2, luminosity, min(255, luminosity + 13))
        r = 1 if power < 0.8 else 2
        d.ellipse((x - r, y - r, x + r, y + r), fill=color)


def keyart_frame(t):
    base = assets()[0]
    # The camera's trajectory has a 32-second period, so opening/closing join.
    angle = 2 * math.pi * t / DURATION
    scale = 1.018 + 0.009 * math.sin(angle)
    offset_x = 12 * math.sin(angle)
    offset_y = 7 * math.cos(angle)
    transform = (1 / scale, 0, (W - W / scale) / 2 + offset_x,
                 0, 1 / scale, (H - H / scale) / 2 + offset_y)
    return base.transform((W, H), Image.Transform.AFFINE, transform, Image.Resampling.BICUBIC)


def overline(img, number, title, local):
    a = smooth((local + 0.1) / 0.8)
    text(img, (106, 253), number, 20, CYAN, "mono", a)
    ImageDraw.Draw(img).line((157, 268, 194, 268), fill=(38, 102, 108), width=1)
    text(img, (212, 253), title, 20, CYAN, "mono", a, 2)


def chips(img, values, y, local, x=107):
    d = ImageDraw.Draw(img)
    for i, value in enumerate(values):
        a = smooth((local - 0.45 - i * 0.12) / 0.65)
        width = int(d.textlength(value, font=font(19, "mono"))) + 36
        d.rounded_rectangle((x, y, x + width, y + 46), radius=2, outline=(26, int(40 + a * 55), int(50 + a * 62)), width=1)
        text(img, (x + 18, y + 11), value, 19, MUTED, "mono", a)
        x += width + 13


def trace(img, t, x=995, y=890, width=730):
    d = ImageDraw.Draw(img)
    d.line((x, y, x + width, y), fill=(19, 48, 60), width=1)
    for i in range(11):
        dx = x + i * width / 10
        d.line((dx, y - 30, dx, y + 30), fill=(15, 35, 46), width=1)
    points = []
    for i in range(320):
        p = i / 319
        envelope = math.exp(-((p - ((t * 0.17) % 0.75 + 0.12)) / 0.115) ** 2)
        value = 35 * math.cos(p * 65 - t * 8) * envelope
        points.append((x + p * width, y + value))
    d.line(points, fill=CYAN, width=2)
    text(img, (x, y + 42), "RECEIVER WAVEFORM / CONCEPT", 15, MUTED, "mono", tracking=1)


def brand_scene(t, closing=False):
    img = keyart_frame(t)
    particles(img, t, 0.65)
    d = ImageDraw.Draw(img)
    # Thin optical sweep lights move independently of the virtual camera.
    sweep = 1030 + math.sin(2 * math.pi * t / DURATION) * 240
    d.line((sweep, 186, sweep + 120, 231), fill=(51, 130, 148), width=1)
    text(img, (107, 303), "REVEAL THE INVISIBLE", 22, CYAN, "mono", tracking=3)
    text(img, (93, 390), "Deep", 162, WHITE, "black")
    width = ImageDraw.Draw(img).textlength("Deep", font=font(162, "black"))
    text(img, (93 + width, 390), "GPR", 162, CYAN, "black")
    text(img, (108, 600), "Physics. Gradients. Possibilities." if closing else "The science beneath the surface.", 36, WHITE, "light")
    if closing:
        text(img, (110, 700), "$", 28, PINK, "mono")
        text(img, (145, 700), "pip install DeepGPR", 28, WHITE, "mono")
        text(img, (110, 767), "songc0a.github.io/DeepGPR", 21, MUTED, "mono")
    else:
        chips(img, ["2D + 3D", "PYTORCH", "CPU + CUDA"], 709, 2)
        text(img, (110, 802), "Differentiable ground-penetrating radar", 22, MUTED)
    return img


def scientific_scene(scene, t):
    local = t - CUTS[scene]
    img = assets()[1].copy()
    # Far-field blueprint grid and subtle depth cues.
    d = ImageDraw.Draw(img)
    for x in range(880, 1880, 64):
        d.line((x, 158, x, 920), fill=(7, 19, 29), width=1)
    for y in range(158, 921, 64):
        d.line((880, y, 1870, y), fill=(7, 19, 29), width=1)
    if scene == 1:
        rendered = draw_hologram(img, t, mode="scan", box=(815, 160, 1870, 850))
        if rendered is not None:
            img = rendered.convert("RGB")
        overline(img, "01", "PROPAGATE", local)
        heading(img, ["Follow", "the field."], local, size=127)
        text(img, (109, 668), "2D + 3D Maxwell FDTD", 33, WHITE, "light", smooth((local - 0.5) / 0.7))
        text(img, (109, 733), "Electromagnetic waves.\nOne differentiable workflow.", 25, MUTED, "regular", smooth((local - 0.7) / 0.7))
        chips(img, ["FORWARD MODELLING", "CPML"], 840, local)
        trace(img, t)
    elif scene == 2:
        overline(img, "02", "RECONSTRUCT", local)
        heading(img, ["From signal", "to structure."], local, size=106, y=348)
        text(img, (109, 629), "Full-waveform inversion", 32, WHITE, "light", smooth((local - 0.5) / 0.7))
        text(img, (109, 694), "Relative permittivity.\nConductivity. Connected by gradients.", 25, MUTED, "regular", smooth((local - 0.7) / 0.7))
        chips(img, ["PYTORCH AUTOGRAD"], 840, local)
        # Faithful, unmodified repository plots in a scientific white lightbox.
        ix, iy = 906, 316
        fw, fh = assets()[2][1].size
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((ix - 16, iy - 63, ix + fw + 16, iy + fh + 52), radius=12, fill=(8, 17, 28), outline=(34, 75, 87), width=1)
        text(img, (ix + 7, iy - 44), "2D FWI / REPOSITORY EXAMPLE", 18, CYAN, "mono", tracking=1)
        reveal = smooth((local - 1.25) / 2.6)
        fig = Image.blend(assets()[2][0], assets()[2][1], reveal)
        img.paste(fig, (ix, iy))
        marker = int(ix + reveal * fw)
        if 0.0 < reveal < 1:
            d = ImageDraw.Draw(img)
            d.line((marker, iy, marker, iy + fh), fill=CYAN, width=2)
        text(img, (ix + 7, iy + fh + 17), "INITIAL → INVERTED  /  εr + σ", 17, MUTED, "mono")
        # A model/data link sweeps along the connection to the result panel.
        d = ImageDraw.Draw(img)
        d.line((738, 545, 852, 545, 882, 520), fill=(24, 68, 80), width=2)
        dotx = 738 + ((t * 95) % 109)
        d.ellipse((dotx - 3, 542, dotx + 3, 548), fill=CYAN)
    elif scene == 3:
        rendered = draw_hologram(img, t, mode="compute", box=(860, 148, 1880, 860))
        if rendered is not None:
            img = rendered.convert("RGB")
        overline(img, "03", "ACCELERATE", local)
        heading(img, ["Physics.", "In motion."], local, size=122)
        text(img, (109, 667), "CPU + CUDA", 36, CYAN, "bold", smooth((local - 0.5) / 0.7))
        text(img, (109, 733), "Native operators. Automatic gradients.\nControls for large models.", 25, MUTED, "regular", smooth((local - 0.7) / 0.7))
        chips(img, ["CHECKPOINTING", "DDP", "OFFLOAD"], 840, local)
        # Long conducting signal trails add a sense of controlled velocity.
        for i in range(7):
            x = 935 + ((t * (160 + i * 19) + i * 173) % 700)
            y = 903 + (i % 2) * 17
            d = ImageDraw.Draw(img)
            d.line((x - 47, y, x + 15, y), fill=(23, 115, 123), width=1)
            d.line((x + 6, y, x + 15, y), fill=CYAN, width=2)
    particles(img, t)
    return img


def scene_frame(scene, t):
    if scene == 0:
        return brand_scene(t)
    if scene == 4:
        img = brand_scene(t, True)
        # Return to the opening brand layout before the homepage loop boundary.
        if t > 30.75:
            img = Image.blend(img, brand_scene(t), smooth((t - 30.75) / 1.0))
        return img
    return scientific_scene(scene, t)


def hud(img, t, scene):
    d = ImageDraw.Draw(img)
    text(img, (106, 84), "DEEPGPR", 20, WHITE, "mono", tracking=3)
    text(img, (276, 84), "/ SIGNAL FILM", 17, MUTED, "mono", tracking=2)
    d.ellipse((1684, 88, 1692, 96), fill=CYAN)
    text(img, (1709, 81), "32 SEC", 19, MUTED, "mono", tracking=1)
    d.line((106, 131, 1814, 131), fill=(32, 47, 57), width=1)
    # Frame corner marks leave a quiet cinematic safe area.
    for x, y, sx, sy in ((55, 55, 1, 1), (1865, 55, -1, 1), (55, 1025, 1, -1), (1865, 1025, -1, -1)):
        d.line((x, y + sy * 22, x, y, x + sx * 22, y), fill=(60, 113, 123), width=1)
    y = 972
    d.line((106, y, 1814, y), fill=(28, 47, 58), width=1)
    # Chapter labels avoid synthetic performance numbers or solver claims.
    labels = ("REVEAL", "PROPAGATE", "RECONSTRUCT", "ACCELERATE", "EXPLORE")
    if t > 31.75:
        scene = 0
    for i, label in enumerate(labels):
        x = 106 + i * 338
        text(img, (x, 995), f"0{i + 1} / {label}", 16, CYAN if i == scene else (81, 102, 120), "mono", tracking=1)
        if i == scene:
            d.line((x, y, x + 84, y), fill=CYAN, width=2)
    if scene in (1, 3):
        text(img, (1104, 209), "CONCEPTUAL WAVEFIELD VISUALIZATION", 14, MUTED, "mono", tracking=1)
    if scene == 0 or scene == 4:
        text(img, (1462, 932), "SUBSURFACE / CONCEPT", 14, MUTED, "mono", tracking=1)
    return img


def frame(index):
    t = index / FPS
    scene = next(i for i in range(5) if CUTS[i] <= t < CUTS[i + 1])
    img = scene_frame(scene, t)
    for i, boundary in enumerate(CUTS[1:-1], 1):
        if abs(t - boundary) < 0.42:
            blend = smooth((t - boundary + 0.42) / 0.84)
            img = Image.blend(scene_frame(i - 1, t), scene_frame(i, t), blend)
            break
    hud(img, t, scene)
    return img


def frame_bytes(index):
    return frame(index).tobytes()


def stills():
    folder = HERE / "render"
    folder.mkdir(parents=True, exist_ok=True)
    times = (2.5, 9.5, 17.5, 23.0, 28.5)
    shots = []
    for i, t in enumerate(times):
        shot = frame(round(t * FPS))
        shot.save(folder / f"chapter-{i + 1}.jpg", quality=93)
        tile = shot.resize((960, 540), Image.Resampling.LANCZOS)
        shots.append(tile)
    contact = Image.new("RGB", (1920, 1620), (3, 7, 15))
    for i, shot in enumerate(shots):
        contact.paste(shot, ((i % 2) * 960, (i // 2) * 540))
    contact.save(folder / "storyboard.jpg", quality=92)
    OUT.mkdir(parents=True, exist_ok=True)
    frame(round(2.5 * FPS)).resize((1280, 720), Image.Resampling.LANCZOS).save(OUT / "deepgpr-promo-poster.jpg", quality=88, optimize=True)
    print(f"Storyboard: {folder / 'storyboard.jpg'}", flush=True)


def encode(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    HERE.joinpath("render").mkdir(parents=True, exist_ok=True)
    silent = HERE / "render/picture-master.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "4", "-movflags", "+faststart", str(silent)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with Pool(workers) as pool:
            for i, raw in enumerate(pool.imap(frame_bytes, range(DURATION * FPS), chunksize=1)):
                process.stdin.write(raw)
                if i % FPS == 0:
                    print(f"Rendering {i // FPS:02d}/{DURATION}s", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Picture encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    sound = HERE / "render/soundtrack.wav"
    if not sound.exists():
        subprocess.run(["python3", str(HERE / "create_soundtrack.py"), "--output", str(sound)], check=True)
    # The MP4 is both the 1080p downloadable film and browser fallback.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(silent), "-i", str(sound), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(DURATION), "-movflags", "+faststart", str(OUT / "deepgpr-promo.mp4")], check=True)
    # A lighter 720p VP9/Opus source is selected first by capable browsers.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(OUT / "deepgpr-promo.mp4"), "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-row-mt", "1", "-threads", "6", "-cpu-used", "3", "-c:a", "libopus", "-b:a", "112k", str(OUT / "deepgpr-promo.webm")], check=True)
    print(f"Delivered {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stills", action="store_true", help="Render only the poster and chapter contact sheet")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stills()
    if not args.stills:
        encode(args.workers)
