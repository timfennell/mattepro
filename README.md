# MattePro

Triangulation matting and alpha-aware focus stacking for macro specimen capture.

Built for photographing pinned insect specimens on a motion rig and producing
clean, background-free stacks for photogrammetry / 3D Gaussian Splatting.

## What it does

Each frame is shot three times — once against a black backdrop, once against white, and once on grey.
Triangulation matting recovers a true alpha channel from that set, including
genuine translucency such as insect wings, which a chroma key cannot.

1. **Matte** — RAW black/white pairs to 16-bit RGBA TIFFs, with optional
   per-pixel background calibration, alpha curve, CA correction and defringing.
2. **Stack** — alpha-aware focus stacking via
   [shinestacker](https://github.com/lucalista/shinestacker)'s pyramid fusion.
3. **Export** — COLMAP-ready image + mask pairs, images written RGBA so
   downstream 3DGS trainers can use the matte directly.

A batch mode runs matte → stack → COLMAP across every rotation of an orbit one
at a time, reclaiming each rotation's mattes as it goes, so peak disk stays at
roughly one rotation instead of the whole capture.

## Notes on the imaging pipeline

- Alpha is solved from unscaled linear RAW. Exposure is a display/output control
  and never enters the matte, or the background stops resolving to alpha 0.
- Stacked output is un-premultiplied before fusion and re-premultiplied after,
  so `max(RGB) <= A` holds and edges do not glow.
- Fused alpha is capped against a per-frame percentile, gated on whether any
  frame shows the pixel genuinely opaque — this suppresses defocus spread
  without eroding thin structures like antennae.
- Frame registration is available for focus breathing, which changes
  magnification across a bracket.

## Requirements

Python 3, NiceGUI, numpy, rawpy, tifffile, opencv-python, scipy, Pillow.
Stacking additionally needs shinestacker on the path.

## Build

    pyinstaller MattePro.spec
