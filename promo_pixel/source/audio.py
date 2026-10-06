"""Original 8-bit style score + sound effects for the DeepGPR promo (pure NumPy synthesis)."""
import wave

import numpy as np

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
DUR = 50.0
N = int(SR * DUR)
L = np.zeros(N)
R = np.zeros(N)
rng = np.random.default_rng(3)

NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(name):
    p = NOTE[name[0]]
    i = 1
    if name[i] in "#b":
        p += 1 if name[i] == "#" else -1
        i += 1
    return 440.0 * 2 ** ((p - 9) / 12 + int(name[i:]) - 4)


def add(sig, t, vol=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N or i < 0:
        return
    sig = sig[:N - i]
    L[i:i + len(sig)] += sig * vol * (1 - max(0, pan))
    R[i:i + len(sig)] += sig * vol * (1 + min(0, pan))


def env(n, a=0.004, d=0.08, s=0.6, r=0.03):
    e = np.full(n, s)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n)
    e[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na)
    e[na:na + nd] = np.linspace(1, s, nd)
    nr = min(nr, n)
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


def pulse(f, dur, duty=0.5, vib=0.0, **kw):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = f * t + (vib * np.sin(2 * np.pi * 5.5 * t) / (2 * np.pi * 5.5) * f if vib else 0)
    w = np.where((ph % 1.0) < duty, 1.0, -1.0) - (2 * duty - 1)
    # soften the edges a little (cheap anti-alias)
    w = np.convolve(w, np.ones(3) / 3, mode="same")
    return w * env(n, **kw)


def tri(f, dur, **kw):
    n = int(dur * SR)
    ph = (f * np.arange(n) / SR) % 1.0
    w = 4 * np.abs(ph - 0.5) - 1
    w = np.round(w * 7.5) / 7.5          # NES-style stepped triangle
    return w * env(n, **kw)


def kick():
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 38)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 20)


def noise(dur, decay, hp=0.0, step=1):
    n = int(dur * SR)
    w = np.repeat(rng.uniform(-1, 1, n // step + 1), step)[:n]
    if hp:
        w = w - np.convolve(w, np.ones(int(hp)) / hp, mode="same")
    return w * np.exp(-np.arange(n) / SR * decay)


CH = {"Am": ["A", "C", "E"], "F": ["F", "A", "C"], "C": ["C", "E", "G"], "G": ["G", "B", "D"],
      "Em": ["E", "G", "B"], "Dm": ["D", "F", "A"]}
ROOT = {"Am": "A2", "F": "F2", "C": "C3", "G": "G2", "Em": "E2", "Dm": "D3"}


def arp_notes(ch, octave=4):
    a, b, c = CH[ch]
    seq = [a + str(octave), b + str(octave + (NOTE[b] < NOTE[a])), c + str(octave + (NOTE[c] < NOTE[a])),
           a + str(octave + 1)]
    return seq


def bar_arp(b, ch, vol=0.10, pat=(0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 3, 1, 2), duty=0.25, octave=4, every=1):
    seq = arp_notes(ch, octave)
    for k in range(0, 16, every):
        add(pulse(hz(seq[pat[k]]), BEAT / 4 * 0.9 * every, duty=duty, d=0.05, s=0.35), b * BAR + k * BEAT / 4, vol,
            pan=0.35 if k % 2 else -0.35)


def bar_bass(b, ch, vol=0.25, pat="x.x.x.xx"):
    f = hz(ROOT[ch])
    for k, c in enumerate(pat):
        if c == "x":
            add(tri(f, BEAT / 2 * 0.9, d=0.02, s=0.9), b * BAR + k * BEAT / 2, vol)
        elif c == "o":
            add(tri(f * 2, BEAT / 2 * 0.9, d=0.02, s=0.9), b * BAR + k * BEAT / 2, vol)


def bar_drums(b, vol=1.0, fill=False, half=False):
    for k in range(4):
        if not half or k % 2 == 0:
            add(kick(), b * BAR + k * BEAT, 0.42 * vol)
        if k % 2 == 1:
            add(noise(0.14, 26, step=3), b * BAR + k * BEAT, 0.16 * vol)
    for k in range(8):
        add(noise(0.04, 90, hp=6), b * BAR + k * BEAT / 2, (0.055 if k % 2 else 0.035) * vol)
    if fill:
        for k in range(4):
            add(noise(0.08, 40, step=2 + k), b * BAR + 3 * BEAT + k * BEAT / 4, 0.13 * vol)


def bar_lead(b, notes, vol=0.17, duty=0.5, octave_shift=0):
    """notes: 8 eighth-note slots: note name, '-' = hold, '.' = rest."""
    k = 0
    while k < 8:
        nme = notes[k]
        if nme in (".", "-"):
            k += 1
            continue
        ln = 1
        while k + ln < 8 and notes[k + ln] == "-":
            ln += 1
        f = hz(nme) * 2 ** octave_shift
        sig = pulse(f, ln * BEAT / 2 * 0.94, duty=duty, vib=0.012 if ln > 1 else 0, d=0.06, s=0.7)
        t = b * BAR + k * BEAT / 2
        add(sig, t, vol)
        add(sig, t + 0.1875, vol * 0.28, pan=0.6)       # echo
        k += ln


MEL_A = [["E5", "-", "A5", "-", "G5", "E5", "C5", "D5"],
         ["C5", "-", "F5", "-", "E5", "C5", "A4", "C5"],
         ["E5", "-", "G5", "-", "E5", "D5", "C5", "E5"],
         ["D5", "-", "-", "B4", "D5", "G5", "-", "-"]]
MEL_B = [["A5", "-", "E5", "A5", "C6", "-", "B5", "A5"],
         ["A5", "-", "F5", "A5", "C6", "-", "A5", "F5"],
         ["G5", "-", "E5", "G5", "C6", "-", "D6", "E6"],
         ["D6", "-", "B5", "G5", "B5", "D6", "-", "-"]]
PROG = ["Am", "F", "C", "G"]

# ---- bars 0-2: quiet intro (night, searching)
for b in range(3):
    ch = PROG[b % 4]
    add(tri(hz(ROOT[ch]), BAR * 0.98, a=0.05, d=0.3, s=0.7, r=0.2), b * BAR, 0.17)
    bar_arp(b, ch, vol=0.06 + 0.012 * b, every=2, duty=0.125)
for k in range(4):   # pickup into the beat
    add(noise(0.08, 40, step=2 + k), 3 * BAR - BEAT + k * BEAT / 4, 0.10)

# ---- bars 3-6: the survey (beat enters)
for i, b in enumerate(range(3, 7)):
    ch = PROG[(b + 1) % 4] if False else PROG[i % 4]
    bar_bass(b, ch)
    bar_drums(b, vol=0.8, fill=(b == 6))
    bar_arp(b, ch, vol=0.075)
    if i >= 2:
        bar_lead(b, MEL_A[i], vol=0.10)

# ---- bars 7-8: logo fanfare
for i, b in enumerate((7, 8)):
    ch = ("C", "G")[i]
    bar_bass(b, ch, pat="x.xxx.xx")
    bar_drums(b, fill=(i == 1))
    bar_arp(b, ch, vol=0.09, pat=(0, 1, 2, 3) * 4, octave=5)
    for nme in arp_notes(ch, 4)[:3]:
        add(pulse(hz(nme), BAR * 0.95, duty=0.5, a=0.01, d=0.4, s=0.55, r=0.2), b * BAR, 0.075)
bar_lead(7, ["C5", "E5", "G5", "C6", "-", "-", "G5", "C6"], vol=0.14)
bar_lead(8, ["D6", "-", "B5", "-", "G5", "B5", "D6", "-"], vol=0.14)

# ---- bars 9-12: autograd (melody A)
for i, b in enumerate(range(9, 13)):
    ch = PROG[i]
    bar_bass(b, ch)
    bar_drums(b, fill=(i == 3))
    bar_arp(b, ch, vol=0.07)
    bar_lead(b, MEL_A[i])

# ---- bars 13-17: inversion (melody B, building) + resolve bar
for i, b in enumerate(range(13, 17)):
    ch = PROG[i]
    bar_bass(b, ch, pat="xoxoxoxo")
    bar_drums(b, fill=(i == 3))
    bar_arp(b, ch, vol=0.07, octave=5 if i >= 2 else 4)
    bar_lead(b, MEL_B[i], vol=0.15)
bar_bass(17, "C", pat="x.x.x.xx")
bar_drums(17, fill=True)
bar_arp(17, "C", vol=0.08, octave=5)
bar_lead(17, ["E6", "-", "-", "C6", "G5", "-", "C6", "-"], vol=0.14)

# ---- bars 18-21: scale (melody A, octave colour)
for i, b in enumerate(range(18, 22)):
    ch = PROG[i]
    bar_bass(b, ch, pat="xoxoxoxo")
    bar_drums(b, fill=(i == 3))
    bar_arp(b, ch, vol=0.07, octave=5)
    bar_lead(b, MEL_A[i], vol=0.14, duty=0.25)

# ---- bars 22-24: outro
for i, b in enumerate((22, 23)):
    ch = ("F", "G")[i]
    bar_bass(b, ch)
    bar_drums(b, fill=(i == 1))
    bar_arp(b, ch, vol=0.08, octave=5)
bar_lead(22, ["A5", "-", "C6", "-", "F5", "A5", "C6", "-"], vol=0.14)
bar_lead(23, ["B5", "-", "D6", "-", "G5", "B5", "D6", "-"], vol=0.14)
for nme in ("C3", "C4", "E4", "G4", "C5"):
    add(pulse(hz(nme), 1.9, duty=0.5, a=0.005, d=0.9, s=0.3, r=0.6), 24 * BAR, 0.09)
add(tri(hz("C2"), 1.9, d=0.4, s=0.6, r=0.6), 24 * BAR, 0.30)
add(kick(), 24 * BAR, 0.5)
for k, nme in enumerate(("C5", "E5", "G5", "C6", "E6", "G6")):
    add(pulse(hz(nme), 0.5, duty=0.25, d=0.2, s=0.3, r=0.2), 24 * BAR + k * BEAT / 4, 0.10, pan=0.4 if k % 2 else -0.4)


# ---- sound effects (synced to the picture)
def ping(t):
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * 1318.5 * tt) + 0.5 * np.sin(2 * np.pi * 1976 * tt)) * np.exp(-tt * 11)
    for k, (dl, v) in enumerate(((0, 0.20), (0.22, 0.09), (0.44, 0.045), (0.66, 0.02))):
        add(s, t + dl, v, pan=(-0.5, 0.5)[k % 2])
    # downward sweep: the pulse going into the ground
    n2 = int(0.35 * SR)
    t2 = np.arange(n2) / SR
    f = 700 * np.exp(-t2 * 6) + 90
    add(np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t2 * 9), t, 0.07)


for t in (2.5, 7.0, 11.0):
    ping(t)


def blip(t, f=880.0, v=0.035, d=0.03):
    add(pulse(f, d, duty=0.5, a=0.001, d=0.01, s=0.5, r=0.01), t, v)


def type_ticks(t0, nchar, cps, f=1200.0, v=0.03, every=2):
    for i in range(0, nchar, every):
        blip(t0 + i / cps, f=f * (1 + 0.06 * ((i * 7) % 3)), v=v, d=0.018)


# caption typing (RPG text blips)
for (t0, n) in ((0.4, 10), (3.0, 16), (6.2, 22), (8.4, 26), (18.6, 30), (26.5, 28), (36.5, 26)):
    type_ticks(t0, n, 26.0, f=660.0, v=0.022)
# terminal typing in the autograd shot
for (t0, n) in ((18.5, 14), (19.3, 28), (21.0, 28), (22.0, 15)):
    type_ticks(t0, n, 30.0, f=1500.0, v=0.03, every=1)
# backward() executes: descending zap, then three OK chimes
tb = 22.0 + 15 / 30.0 + 0.15
n2 = int(0.5 * SR)
t2 = np.arange(n2) / SR
add(np.sign(np.sin(2 * np.pi * np.cumsum(1400 * np.exp(-t2 * 4) + 200) / SR)) * np.exp(-t2 * 5), tb, 0.06)
for i in range(3):
    blip(tb + 0.55 + 0.35 * i, f=hz(("E6", "G6", "C7")[i]), v=0.06, d=0.09)
# logo letters landing
for i in range(7):
    add(kick(), 14.5 + i * 0.07 + 0.22, 0.25)
    blip(14.5 + i * 0.07 + 0.22, f=hz(("C5", "D5", "E5", "G5", "A5", "C6", "E6")[i]), v=0.05, d=0.06)
type_ticks(16.45, 21, 22.0, f=1500.0, v=0.03, every=1)
# FWI converged jingle
for k, nme in enumerate(("C6", "E6", "G6", "C7")):
    blip(26 + 0.8 + 6.4 + k * 0.08, f=hz(nme), v=0.08, d=0.12)
# memory bars + chips + backends
for i, t in enumerate((36.5, 37.3, 38.1)):
    blip(t, f=hz(("G5", "E5", "C5")[i]), v=0.06, d=0.12)
for i in range(4):
    blip(39.0 + 0.35 * i, f=hz("A5") * (1 + 0.12 * i), v=0.045, d=0.05)
for t in (40.6, 41.0):
    blip(t, f=hz("C6"), v=0.05, d=0.07)

# ---- master
mix = np.stack([L, R], 1)
fade = np.ones(N)
nf = int(1.2 * SR)
fade[-nf:] = np.linspace(1, 0, nf) ** 1.5
fade[:int(0.05 * SR)] = np.linspace(0, 1, int(0.05 * SR))
mix *= fade[:, None]
mix = np.tanh(mix * 1.5) / np.tanh(1.5)
mix *= 0.89 / np.abs(mix).max()
with wave.open("music.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print("music.wav", DUR, "s, peak", np.abs(mix).max(), "rms", np.sqrt((mix ** 2).mean()))
