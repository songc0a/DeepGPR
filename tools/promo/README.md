# DeepGPR signal film

A 32-second cyberpunk brand film for the homepage. The visual language is
obsidian geology, cyan electromagnetic contours, magenta subsurface layers,
restrained bloom, and large, clear typography. The film uses English to match
the existing website.

## Deliverables

- `website/assets/video/deepgpr-promo.mp4`: 1920×1080, 30 fps, H.264/AAC,
  stereo original score, fast-start metadata, 32 seconds.
- `website/assets/video/deepgpr-promo.webm`: 1280×720, 30 fps, VP9/Opus,
  approximately 5 MB; the homepage's preferred source.
- `website/assets/video/deepgpr-promo-poster.jpg`: opening brand frame.

The homepage loads sources when the film becomes visible, starts muted, loops,
and offers play/pause and sound controls. Reduced-motion preferences keep the
poster visible until Play is pressed. The section pauses when it leaves the
viewport or the browser tab is hidden.

## Edit structure

| Time | Chapter | Visual |
| --- | --- | --- |
| 0–6 s | Reveal | Cinematic geological key art, virtual camera, brand title |
| 6–13 s | Propagate | Animated 3D scan, terrain grid, receiver waveform |
| 13–20 s | Reconstruct | Repository's initial and inverted 2D FWI figures |
| 20–26 s | Accelerate | Moving computation trails, subsurface volume |
| 26–32 s | Explore | Brand resolve, installation command, website address |

Wavefield and geological effects are conceptual brand graphics; the film marks
them as such. The real scientific panels come from `Fig/2dfwiinit.png` and
`Fig/2dfwipred.png`, with their original labels and color scales. The smooth
transition between those endpoints is illustrative, not recorded solver
iterations. Product capabilities are taken from this checkout's README; the
film makes no quantitative speed or accuracy claims.

## Source assets and soundtrack

The key art in `assets/subsurface-keyart.png` was generated with the **built-in
imagegen tool**, using the `imagegen` skill. The final generation prompt is in
`assets/keyart-prompt.txt`; no API/CLI fallback was used. Typography, movement,
the 3D mesh, scan plane, particles, waveform, and transitions are rendered by
the local scripts. The original soundtrack is synthesized from oscillators and
seeded noise, with no external music or samples. Its transitions land at
6, 13, 20, and 26 seconds.

## Rebuild

Use Python 3.10+ with Pillow, NumPy, and SciPy, plus ffmpeg with libx264,
libvpx-vp9, AAC, and libopus. On Linux the renderer uses installed Lato and
DejaVu fonts under `/usr/share/fonts/truetype`, with a DejaVu fallback.

```bash
python3 tools/promo/create_soundtrack.py
python3 tools/promo/render_film.py --stills
python3 tools/promo/render_film.py --workers 6
```

Generated intermediate files and chapter previews go into the ignored
`render/` folder. Final website files and the key art are retained in the
repository. No DeepGPR solver, CUDA installation, or external service is
required for rebuilding from the saved asset.

Validation: all five chapter previews inspected; both final videos fully
decoded without media errors; dimensions, frame rate, audio channels and
duration checked; homepage playback and controls checked in the browser.
