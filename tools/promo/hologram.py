"""Procedural holographic GPR volume for the DeepGPR homepage film.

``draw_hologram(image, time, mode='scan', box=(760,170,1800,850))``
composites into and returns the supplied PIL RGB/RGBA image. ``time`` is
seconds, all animation is deterministic, and ``box`` is an (x0,y0,x1,y1)
design region. Supported modes: scan, invert, compute. There is no text.
Only NumPy and Pillow are needed; bloom is rendered locally at quarter size.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CYAN = (74, 240, 245)
TEAL = (20, 149, 169)
ICE = (176, 253, 249)
PINK = (249, 67, 145)
CORAL = (252, 126, 102)
WHITE = (255, 243, 214)
_SEED = np.random.default_rng(5107)
_PARTICLES = _SEED.uniform(0, 1, (105, 5))


def _terrain(x, z):
    return .058 * np.sin(x * 2.4 + z * .8) + .029 * np.cos(z * 4.1 - x * .55)


def draw_hologram(image: Image.Image, time: float, mode: str = "scan",
                  box: tuple = (760, 170, 1800, 850)) -> Image.Image:
    """Mutate and return ``image`` with a rotating, projected radar volume."""
    if mode not in {"scan", "invert", "compute"}:
        raise ValueError("mode must be scan, invert, or compute")
    bx0, by0, bx1, by1 = map(float, box)
    # Work in a generously padded local viewport, clipped to the source image.
    margin = .11 * (bx1 - bx0)
    ox = max(0, int(bx0 - margin))
    oy = max(0, int(by0 - margin))
    ex = min(image.width, int(bx1 + margin))
    ey = min(image.height, int(by1 + margin))
    w, h = ex - ox, ey - oy
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    bw, bh = bx1 - bx0, by1 - by0
    scale = min(bw / 3.65, bh / 2.16)
    cx = (bx0 + bx1) * .5 - ox
    cy = by0 + bh * .31 - oy
    yaw = .48 + .105 * math.sin(time * .19)
    tilt = .47 + .025 * math.cos(time * .13)
    co, si = math.cos(yaw), math.sin(yaw)
    stroke = max(1, int(bw / 1000))

    def project(points):
        a = np.asarray(points, dtype=float)
        x, y, z = a[..., 0], a[..., 1], a[..., 2]
        return np.stack((cx + scale * (co * x - si * z),
                         cy + scale * (-y + tilt * (si * x + co * z))), axis=-1)

    def line(points, color, alpha=120, width=1):
        p = project(points)
        d.line([tuple(v) for v in p], fill=(*color, int(alpha)),
               width=max(1, int(width * stroke)), joint="curve")

    def polygon(points, color, alpha):
        d.polygon([tuple(v) for v in project(points)], fill=(*color, int(alpha)))

    def dot(point, color, alpha, radius=1.4):
        qx, qy = project(point)
        rr = radius * max(.7, bw / 1040)
        d.ellipse((qx-rr, qy-rr, qx+rr, qy+rr), fill=(*color, int(alpha)))

    # Perspective has an open top and two translucent front-facing cut faces.
    # Their layered geometry supplies the feeling of a real radar tomogram.
    x0, x1, z0, z1, floor = -1.27, 1.27, -.84, .84, -1.23

    def top(x, z):
        return float(_terrain(x, z))

    polygon([(x0,top(x0,z1),z1),(x1,top(x1,z1),z1),
             (x1,floor,z1),(x0,floor,z1)], TEAL, 10)
    polygon([(x1,top(x1,z0),z0),(x1,top(x1,z1),z1),
             (x1,floor,z1),(x1,floor,z0)], CYAN, 7)
    polygon([(x0,floor,z0),(x1,floor,z0),(x1,floor,z1),(x0,floor,z1)], PINK, 7)

    # Rear depth cues drawn before the brighter foreground mesh.
    for dep in np.linspace(-.09, floor, 15):
        color = TEAL if dep > -.74 else PINK
        alpha = 24 if dep > -.74 else 36
        for z in (z0, z1):
            xx = np.linspace(x0, x1, 35)
            yy = dep + .023 * np.sin(xx * 3.8 + z * 3 + time * .12)
            line(np.c_[xx, yy, xx*0+z], color, alpha)
        for x in (x0, x1):
            zz = np.linspace(z0, z1, 29)
            yy = dep + .023 * np.sin(x * 3.8 + zz * 3 + time * .12)
            line(np.c_[zz*0+x, yy, zz], color, alpha)

    for x in np.linspace(x0,x1,20):
        for z in (z0,z1):
            line([(x,top(x,z),z),(x,floor,z)], TEAL, 40)
    for z in np.linspace(z0,z1,14):
        for x in (x0,x1):
            line([(x,top(x,z),z),(x,floor,z)], CYAN, 25)

    # A lower warm fault stratum ripples independently of the surface.
    for j, dep in enumerate((-.80,-.88,-.97)):
        xx = np.linspace(x0,x1,48)
        for z in np.linspace(z0,z1,7):
            yy = dep + .045*np.sin(xx*2.8 + z*2 + .2)
            line(np.c_[xx,yy,xx*0+z], PINK if j < 2 else CORAL, 43 + j*13)
    # Interior lattice: light enough that the anomaly remains legible.
    for dep in (-.29,-.56,-1.18):
        for x in np.linspace(x0,x1,8):
            line([(x,dep,z0),(x,dep,z1)], TEAL if dep>-.7 else PINK, 22)

    # Controlled scan plane, accompanied by wavefront traces in every depth.
    scan_x = x0 + (x1-x0) * (.5 + .48 * math.sin(time * .55 - .8))
    if mode == "scan":
        polygon([(scan_x,top(scan_x,z0)+.05,z0),
                 (scan_x,top(scan_x,z1)+.05,z1),
                 (scan_x,floor,z1),(scan_x,floor,z0)], CYAN, 30)
        for z in np.linspace(z0,z1,14):
            line([(scan_x,top(scan_x,z)+.05,z),(scan_x,floor,z)], CYAN, 61)
        for dep in (0,-.29,-.6,-.91,-1.23):
            line([(scan_x,dep,z0),(scan_x,dep,z1)], ICE, 154, 1.3)
        line([(scan_x,floor,z0),(scan_x,floor,z1)], WHITE, 150, 1.5)

    # Inversion result: an anisotropic object reconstructed with sectional
    # curves; deliberately incomplete early on, with a fine warm inner core.
    if mode in {"invert", "compute"}:
        reveal = .38 + .62 * min(1., max(0., time / 3.5))
        theta = np.linspace(0,math.tau,100)
        ac = np.array([-.10,-.64,.07])
        pulse = 1 + .018 * math.sin(time*1.45)
        rad = np.array([.68,.30,.45]) * pulse
        for lat in np.linspace(-1.30,1.30,12):
            ring = np.c_[rad[0]*np.cos(lat)*np.cos(theta),
                          theta*0+rad[1]*np.sin(lat),
                          rad[2]*np.cos(lat)*np.sin(theta)] + ac
            count = int(len(theta)*reveal)
            line(ring[:count], CORAL if abs(lat)<.44 else PINK,
                 175 if abs(lat)<.44 else 100)
        for lon in np.linspace(0,math.tau,16,endpoint=False):
            a = np.linspace(-math.pi/2,math.pi/2,54)
            section = np.c_[rad[0]*np.cos(a)*np.cos(lon),
                             rad[1]*np.sin(a),
                             rad[2]*np.cos(a)*np.sin(lon)] + ac
            line(section[:int(54*reveal)], PINK, 78)
        for phase in np.linspace(0,math.tau,28,endpoint=False):
            q = ac + rad*np.array([math.cos(phase)*.8,
                                   math.sin(phase*3)*.3,math.sin(phase)*.8])
            dot(q, WHITE, 175, 1.15)

    # Rings are horizontal 3-D wavefronts and pass through the actual volume.
    # Clipping their x/z coordinates gives subtle intersections at cut faces.
    for i in range(5):
        phase = (time*.20 + i*.20) % 1.
        radius = .20 + phase*1.66
        theta = np.linspace(0,math.tau,140)
        xx = -.28 + radius * np.cos(theta)
        zz = .03 + radius * .78 * np.sin(theta)
        yy = -.10 - phase*.86 + .018*np.sin(theta*4-time*.8)
        valid = (xx>x0)&(xx<x1)&(zz>z0)&(zz<z1)
        inds = np.flatnonzero(valid)
        splits = np.split(inds,np.where(np.diff(inds)>1)[0]+1)
        for part in splits:
            if len(part)>2:
                line(np.c_[xx[part],yy[part],zz[part]], CYAN if i%2 else ICE,
                     18 + (1-phase)*42)

    # Organic relief at the open scan surface.
    xx = np.linspace(x0,x1,54)
    for j,z in enumerate(np.linspace(z0,z1,22)):
        yy = _terrain(xx,z)
        base = 96 if j%3 else 145
        line(np.c_[xx,yy,xx*0+z], CYAN, base)
    zz = np.linspace(z0,z1,42)
    for j,x in enumerate(np.linspace(x0,x1,28)):
        yy = _terrain(x,zz)
        line(np.c_[zz*0+x,yy,zz], CYAN if j%4 else ICE, 66 if j%4 else 118)

    # The four major volume corners define a clean silhouette.
    for x,z in ((x0,z0),(x1,z0),(x0,z1),(x1,z1)):
        line([(x,top(x,z),z),(x,floor,z)], CYAN, 91)
        dot((x,top(x,z),z), WHITE, 255, 2.0)
        dot((x,floor,z), CORAL, 210, 1.45)
    for z in (z0,z1):
        line(np.c_[xx,_terrain(xx,z),xx*0+z], ICE, 211, 1.4)
        line([(x0,floor,z),(x1,floor,z)], PINK, 145)
    for x in (x0,x1):
        line(np.c_[zz*0+x,_terrain(x,zz),zz], ICE, 175, 1.4)
        line([(x,floor,z0),(x,floor,z1)], CORAL, 117)

    # Luminous micro-particles inhabit the volume and a narrow halo around it.
    speed = .38 if mode == "compute" else .075
    for n,(a,b,c,e,f) in enumerate(_PARTICLES):
        px = (a*2-1)*1.40
        pz = (b*2-1)*.93
        py = -.08 - ((c + time*speed*(.40+e))%1)*1.25
        opacity = 60 + 135 * (.5+.5*math.sin(time*(1+e)+f*math.tau))
        dot((px,py,pz), WHITE if n%7==0 else CYAN, opacity, .8 if n%7 else 1.5)
        if mode == "compute" and n%3==0:
            line([(px,py,pz),(px,py+.10+.06*e,pz)], CYAN, 65)

    # Floating registration brackets and a fine perimeter survey ring are
    # geometry, not labels, and stay deliberately sparse.
    for x,z in ((x0,z0),(x1,z1)):
        sy = top(x,z) + .15
        line([(x-.10,sy,z),(x+.10,sy,z)], ICE, 115)
        line([(x,sy,z-.08),(x,sy,z+.08)], ICE, 115)
    for k in range(9):
        x = x0 + (x1-x0)*k/8
        dot((x,top(x,z1),z1), WHITE, 235 if k%2==0 else 150, 1.7)
    if mode == "compute":
        pulse_x = x0 + (time*.65 % (x1-x0))
        line([(pulse_x,top(pulse_x,z0)+.008,z0),
              (pulse_x,top(pulse_x,z1)+.008,z1)], WHITE, 198, 1.5)

    # Bloom is added with screen blending, so overlapping fine lines brighten
    # without erasing the clean black surrounding negative space.
    small = layer.resize((max(1,w//4),max(1,h//4)), Image.Resampling.BILINEAR)
    bloom = small.filter(ImageFilter.GaussianBlur(3.3)).resize((w,h),Image.Resampling.BILINEAR)
    bloom.putalpha(bloom.getchannel("A").point(lambda a: min(116, a*2)))
    crop = image.crop((ox,oy,ex,ey)).convert("RGBA")
    crop = Image.alpha_composite(crop,bloom)
    crop = Image.alpha_composite(crop,layer)
    image.paste(crop.convert(image.mode), (ox,oy))
    return image


if __name__ == "__main__":
    import pathlib
    import time as clock
    out = pathlib.Path(__file__).parent / "render"
    out.mkdir(exist_ok=True)
    for mode, t in (("scan",2.8),("invert",9.6),("compute",15.8)):
        im = Image.new("RGB",(1920,1080),(3,8,14))
        start = clock.perf_counter()
        draw_hologram(im,t,mode)
        im.save(out/f"hologram-{mode}.jpg",quality=94)
        print(mode,round(clock.perf_counter()-start,3),"seconds")
