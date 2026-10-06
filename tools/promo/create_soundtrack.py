#!/usr/bin/env python3
"""Render an original, deterministic 32-second DeepGPR cyberpunk score.

Only procedural oscillators and seeded noise are used; there are no samples.
Run: python tools/promo/create_soundtrack.py --output tools/promo/render/soundtrack.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np
from scipy import signal


SAMPLE_RATE = 48_000
DURATION = 32.0
RNG = np.random.default_rng(1042026)


def midi(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12.0)


def lowpass(audio: np.ndarray, cutoff: float, order: int = 3) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, cutoff, fs=SAMPLE_RATE,
                                      output="sos"), audio, axis=0)


def highpass(audio: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, cutoff, btype="highpass",
                                      fs=SAMPLE_RATE, output="sos"), audio, axis=0)


def time_axis(length: float) -> np.ndarray:
    return np.arange(round(length * SAMPLE_RATE)) / SAMPLE_RATE


def taper(length: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(length)
    a = min(length, round(attack * SAMPLE_RATE))
    r = min(length, round(release * SAMPLE_RATE))
    if a:
        env[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if r:
        env[-r:] *= np.sin(np.linspace(np.pi / 2, 0, r)) ** 2
    return env


def add(bus: np.ndarray, source: np.ndarray, start: float,
        gain: float = 1.0, pan: float = 0.0) -> None:
    offset = round(start * SAMPLE_RATE)
    if offset >= len(bus):
        return
    if offset < 0:
        source = source[-offset:]
        offset = 0
    count = min(len(source), len(bus) - offset)
    if source.ndim == 1:
        angle = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        bus[offset:offset + count, 0] += source[:count] * gain * np.cos(angle)
        bus[offset:offset + count, 1] += source[:count] * gain * np.sin(angle)
    else:
        bus[offset:offset + count] += source[:count] * gain


def pad(chord: list[int], duration: float) -> np.ndarray:
    t = time_axis(duration)
    out = np.zeros((len(t), 2))
    for index, note in enumerate(chord):
        frequency = midi(note)
        for channel, detune in enumerate((-0.0018, 0.0018)):
            phase = 2 * np.pi * frequency * (1 + detune) * t
            tone = (np.sin(phase) + 0.23 * np.sin(2 * phase + 0.3)
                    + 0.08 * np.sin(3 * phase + 0.7))
            # Independent, very slow movement leaves the mix open and spacious.
            motion = 0.82 + 0.18 * np.sin(2 * np.pi * (0.105 + index * 0.015)
                                        * t + index + channel)
            out[:, channel] += tone * motion / len(chord)
    return lowpass(out, 1700) * taper(len(t), 1.0, 1.6)[:, None]


def pluck(note: int, length: float = 1.8, brightness: float = 1.0) -> np.ndarray:
    t = time_axis(length)
    frequency = midi(note)
    phase = 2 * np.pi * frequency * t
    # A soft FM attack becomes a pure bell-like tone, with no copied melody.
    modulation = 1.25 * brightness * np.exp(-t / 0.085) * np.sin(phase * 2)
    tone = (np.sin(phase + modulation)
            + 0.28 * np.sin(phase * 2.004) * np.exp(-t / 0.3)
            + 0.05 * np.sin(phase * 4.01) * np.exp(-t / 0.07))
    env = np.exp(-t / 0.48) * taper(len(t), 0.005, 0.13)
    return tone * env


def bass(note: int, duration: float) -> np.ndarray:
    t = time_axis(duration)
    phase = 2 * np.pi * midi(note) * t
    tone = (0.82 * np.sin(phase) + 0.44 * np.sin(phase / 2)
            + 0.13 * np.sin(phase * 2) + 0.045 * np.sin(phase * 3))
    return tone * np.exp(-t / 0.7) * taper(len(t), 0.028, 0.16)


def kick() -> np.ndarray:
    t = time_axis(0.5)
    frequency = 47.0 + 95.0 * np.exp(-t / 0.026)
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    tone = np.sin(phase) * np.exp(-t / 0.105)
    air = highpass(RNG.standard_normal(len(t)), 2800)
    return (tone + 0.055 * air * np.exp(-t / 0.008)) * taper(len(t), 0.001, 0.055)


def snare() -> np.ndarray:
    t = time_axis(0.27)
    noise = highpass(lowpass(RNG.standard_normal(len(t)), 7500), 1400)
    tone = np.sin(2 * np.pi * 174 * t) * np.exp(-t / 0.037)
    return (0.56 * noise * np.exp(-t / 0.055) + 0.22 * tone) * taper(len(t), 0.0015, 0.04)


def hat() -> np.ndarray:
    t = time_axis(0.12)
    noise = highpass(lowpass(RNG.standard_normal(len(t)), 13000), 6800)
    return noise * np.exp(-t / 0.022) * taper(len(t), 0.001, 0.016)


def whoosh(length: float = 1.65) -> np.ndarray:
    t = time_axis(length)
    n = RNG.standard_normal((len(t), 2))
    n = highpass(lowpass(n, 6500), 430)
    envelope = np.sin(np.pi * t / length) ** 2
    texture = 0.85 + 0.15 * np.sin(2 * np.pi * (18 * t + 19 * t**2))
    return n * (envelope * texture)[:, None]


def impact() -> np.ndarray:
    t = time_axis(2.4)
    phase = 2 * np.pi * (37 * t + 4.0 * (1 - np.exp(-t / 0.1)))
    sub = np.sin(phase) * np.exp(-t / 0.68)
    grit = lowpass(RNG.standard_normal((len(t), 2)), 350)
    return (sub[:, None] * 0.73 + grit * np.exp(-t / 0.48)[:, None]) * taper(len(t), 0.009, 0.35)[:, None]


def spatial_tail(source: np.ndarray) -> np.ndarray:
    """Light stereo echoes and diffuse early reflections without muddy sub bass."""
    source = highpass(source, 280)
    tail = np.zeros_like(source)
    for delay, level in ((0.1875, 0.17), (0.375, 0.14), (0.5625, 0.09),
                         (0.75, 0.065), (0.9375, 0.035), (1.3125, 0.02)):
        d = round(delay * SAMPLE_RATE)
        tail[d:] += lowpass(source[:-d, ::-1], 4900 / (1 + delay)) * level
    for index, delay in enumerate((0.053, 0.071, 0.101, 0.137, 0.163, 0.211)):
        d = round(delay * SAMPLE_RATE)
        tail[d:] += source[:-d, ::-1 if index % 2 else 1] * 0.021
    return tail


def render(output: Path) -> None:
    n = round(DURATION * SAMPLE_RATE)
    harmonic = np.zeros((n, 2))
    drums = np.zeros_like(harmonic)
    effects = np.zeros_like(harmonic)
    chords = [
        [50, 57, 60, 65, 69], [46, 53, 57, 62, 65],
        [43, 50, 53, 58, 62], [45, 52, 55, 60, 64],
        [50, 57, 60, 65, 69], [46, 53, 57, 62, 65],
        [45, 52, 55, 60, 64], [50, 57, 60, 65, 69],
    ]
    roots = [38, 34, 31, 33, 38, 34, 33, 38]
    for index, chord in enumerate(chords):
        add(harmonic, pad(chord, 6.2), index * 4 - 0.25,
            0.225 if index < 2 else 0.25)

    # Intro: cinematic weight and sparse glass that makes room for the reveal.
    add(effects, impact(), 0.4, 0.29)
    for start, note, pan in ((1.1, 81, -0.5), (2.3, 77, 0.5),
                             (3.7, 84, -0.2), (4.65, 81, 0.6)):
        add(harmonic, pluck(note, 2.2, 0.7), start, 0.055, pan)

    # Original non-vocal ostinato, following each chord rather than a stock melody.
    pattern = [2, 4, 3, 1, 4, 2, 3, 4]
    for step, start in enumerate(np.arange(6.0, 26.0, 0.25)):
        chord = chords[min(7, int(start // 4))]
        # Deliberately leave breaths in the inversion sequence.
        if start < 20 and step % 2 == 1:
            continue
        note = chord[pattern[step % len(pattern)]] + 12
        volume = 0.043 if start < 20 else 0.051
        if step % 8 in (0, 3):
            volume *= 1.16
        add(harmonic, pluck(note, brightness=0.9), float(start), volume,
            0.53 * np.sin(step * 1.6))

    # Warm pulse with sidechain-like breathing that keeps the low end clear.
    for step, start in enumerate(np.arange(6.0, 28.0, 0.5)):
        root = roots[min(7, int(start // 4))]
        add(harmonic, bass(root, 0.49), float(start) + 0.09,
            0.145 if start < 20 else 0.165)
        add(drums, kick(), float(start), 0.235 if step % 4 == 0 else 0.19)
        if step % 4 in (2,):
            add(drums, snare(), float(start), 0.087, 0.03)
        add(drums, hat(), float(start) + 0.25, 0.055 if step % 2 else 0.037,
            0.28 if step % 2 else -0.28)
        if 20 <= start < 25.5:
            add(drums, hat(), float(start) + 0.125, 0.024, -0.4)

    # Transition swells land on the film's exact chapter boundaries.
    for boundary in (6.0, 13.0, 20.0, 26.0):
        add(effects, whoosh(), boundary - 1.22, 0.075)
        add(effects, impact(), boundary, 0.18 if boundary < 26 else 0.29)
        add(harmonic, pluck(86 if boundary != 26 else 81, 2.7, 0.45),
            boundary + 0.035, 0.047, -0.18)

    # Closing chord and slow descending lights resolve the last brand card.
    for start, note, pan in ((26.4, 81, 0.4), (27.4, 77, -0.4),
                             (28.4, 74, 0.2), (29.4, 69, -0.25)):
        add(harmonic, pluck(note, 2.5, 0.55), start, 0.067, pan)
    add(harmonic, bass(38, 3.6), 28.0, 0.14)

    # Subtle electronic atmosphere throughout, naturally narrowing at the end.
    t = np.arange(n) / SAMPLE_RATE
    texture = highpass(lowpass(RNG.standard_normal((n, 2)), 4200), 1700)
    texture_env = (0.35 + 0.65 * np.sin(np.pi * np.minimum(t / 32, 1)) ** 2)
    effects += texture * texture_env[:, None] * 0.008
    harmonic += spatial_tail(harmonic) * 0.75
    effects += spatial_tail(effects) * 0.38
    mix = harmonic + drums + effects
    mix = highpass(mix, 25)

    # Gentle analog saturation; preserve transients and prevent hard clipping.
    mix = np.tanh(mix * 1.15) / 1.15
    mix *= taper(n, 0.25, 1.15)[:, None]
    mix -= mix.mean(axis=0, keepdims=True)
    # Ceiling -1.1 dBFS, after the final DC correction.
    peak = float(np.max(np.abs(mix)))
    mix *= 10 ** (-1.1 / 20) / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(mix, -1, 1) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())
    peak_db = 20 * np.log10(np.max(np.abs(pcm.astype(float))) / 32768)
    rms_db = 20 * np.log10(np.sqrt(np.mean((pcm.astype(float) / 32768) ** 2)))
    print(f"Rendered: {output}")
    print(f"Duration: {len(pcm) / SAMPLE_RATE:.3f}s | 48 kHz stereo 16-bit PCM")
    print(f"Peak: {peak_db:.2f} dBFS | RMS: {rms_db:.2f} dBFS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "render" / "soundtrack.wav")
    render(parser.parse_args().output)
