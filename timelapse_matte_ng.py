#!/usr/bin/env python3
"""
MattePro — NiceGUI edition (same framework + CSS as Smalti)
"""

import asyncio
import base64
import gc
import io
import logging
import os
import queue
import re
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageFilter

from nicegui import ui, app

# shinestacker supplies the alpha-aware focus stacker (PyramidAutoStack).
# It is a normal installed dependency:  pip install "shinestacker>=1.17.0"
#
# 1.17.0 is the first release that carries an alpha channel through the
# Laplacian pyramid. Older versions still stack, but silently discard the
# matte — which looks like a MattePro bug rather than a version problem, so
# the version is checked up front and reported in the UI.
SHINESTACKER_MIN = "1.17.0"


def _ver_tuple(s):
    """Parse '1.17.0' / '1.17.0rc1' into a comparable 3-tuple."""
    out = []
    for part in str(s).split(".")[:3]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        out.append(int(digits or 0))
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def _shinestacker_status():
    """(available, version, problem) — never imports shinestacker.

    Importing it pulls in scipy and matplotlib, which is slow enough to be
    visible at startup, so availability is probed via the module finder.
    find_spec works inside a PyInstaller bundle; dist metadata may not be
    present there, so a missing version is treated as "bundled and fine"
    rather than as an error.
    """
    import importlib.util
    import importlib.metadata as _md
    if importlib.util.find_spec("shinestacker") is None:
        return False, None, ("shinestacker is not installed — "
                             f"pip install \"shinestacker>={SHINESTACKER_MIN}\"")
    try:
        v = _md.version("shinestacker")
    except Exception:
        return True, None, None
    if _ver_tuple(v) < _ver_tuple(SHINESTACKER_MIN):
        return False, v, (f"shinestacker {v} is installed, but {SHINESTACKER_MIN} or "
                          "newer is required to carry alpha through the stack")
    return True, v, None

# ── Kelvin → linear-light RGB ─────────────────────────────────────────────────

def _kelvin_srgb255(k):
    t = k / 100
    if t <= 66:
        r = 255.0
        g = max(0.0, min(255.0, 99.4708025861 * np.log(t) - 161.1195681661))
        b = 0.0 if t <= 19 else max(0.0, min(255.0,
                138.5177312231 * np.log(t - 10) - 305.0447927307))
    else:
        r = max(0.0, min(255.0, 329.698727446 * (t - 60) ** -0.1332047592))
        g = max(0.0, min(255.0, 288.1221695283 * (t - 60) ** -0.0755148492))
        b = 255.0
    return r, g, b

def kelvin_to_linear(k):
    r, g, b = _kelvin_srgb255(k)
    def srgb_inv(v): return (v / 255.0 / 12.92) if v/255.0 <= 0.04045 else ((v/255.0 + 0.055) / 1.055) ** 2.4
    return np.array([srgb_inv(r), srgb_inv(g), srgb_inv(b)], dtype=np.float32)

@dataclass
class MattePair:
    black: Path
    white: Path
    grey:  Optional[Path]
    out_path: Path

def _frame_key(p):
    m = re.search(r"rot[+\-]?\d+\.\d+", p.as_posix(), re.IGNORECASE)
    return m.group(0).lower() if m else p.parent.name

def _read_raw_linear(path, half_size=False, ca_scale=(1.0, 1.0)):
    """Return scene-linear [0,1] float32 RGB. rawpy gamma=(1,1) = linear output; no gamma decode needed."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            gamma=(1, 1),
            bright=1.0,
            half_size=half_size,
        )
    arr = rgb.astype(np.float32) / 65535.0
    if ca_scale != (1.0, 1.0):
        arr[:, :, 0] = np.clip(arr[:, :, 0] * ca_scale[0], 0, 1)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * ca_scale[1], 0, 1)
    return arr

def _apply_defringe(img, strength):
    if strength <= 0: return img
    from PIL import Image as PILImage
    pil = PILImage.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    import cv2
    hsv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    purple_mask = (((h >= 130) & (h <= 170)) | ((h >= 280) & (h <= 320)) |
                   ((h >= 10) & (h <= 40))) & (s > 30)
    alpha = (strength / 100.0) * purple_mask.astype(np.float32)
    grey = np.mean(img, axis=2, keepdims=True)
    grey3 = np.repeat(grey, 3, axis=2)
    return (img * (1 - alpha[:,:,None]) + grey3 * alpha[:,:,None]).clip(0, 1)

def _quarter_size(arr):
    h, w = arr.shape[:2]
    return arr[:h//2*2:2, :w//2*2:2]

def _resolve_grey_bg(bg_white, grey_brightness, cal_grey):
    if cal_grey is not None: return cal_grey
    return bg_white * (grey_brightness / 100.0)

def _lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x**(1/2.4) - 0.055)

def _srgb_to_lin(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _checkerboard(h, w, size=16):
    rows = np.arange(h) // size; cols = np.arange(w) // size
    dark = np.array([153, 153, 153], dtype=np.uint8)
    light = np.array([204, 204, 204], dtype=np.uint8)
    mask = ((rows[:,None] + cols[None,:]) % 2 == 0)
    out = np.where(mask[:,:,None], dark, light)
    return out.astype(np.uint8)

@dataclass
class CalImages:
    black_full: Optional[np.ndarray] = None
    white_full: Optional[np.ndarray] = None
    grey_full:  Optional[np.ndarray] = None
    black_half: Optional[np.ndarray] = None
    white_half: Optional[np.ndarray] = None
    grey_half:  Optional[np.ndarray] = None

    @property
    def available(self):
        return self.black_full is not None and self.white_full is not None

_RAW_EXTS = frozenset(("arw","cr3","cr2","nef","raf","dng"))

def _find_raws(directory: Path):
    """Return sorted RAW files in directory (scandir, no extra stat calls)."""
    try:
        entries = sorted(os.scandir(str(directory)), key=lambda e: e.name)
    except OSError:
        return []
    # Skip macOS resource forks ("._name.ARW"): they carry a RAW extension but are
    # 4KB metadata stubs. Critically, "._" sorts before real filenames, so leaving
    # them in would also shift black/white pairing out of alignment by one.
    return [Path(e.path) for e in entries
            if e.is_file(follow_symlinks=False)
            and not e.name.startswith(".")
            and e.name.rsplit(".", 1)[-1].lower() in _RAW_EXTS]

def _scan_project(proj: Path):
    """Single os.walk pass — returns (pairs, matte_dirs, stacked_files, cal_paths).
    cal_paths is dict with optional keys 'cal_black_path', 'cal_white_path', 'cal_grey_path'.
    """
    pairs         = []
    matte_dirs    = []
    stacked_files = []
    rot_dirs_seen = set()
    # Cal frame dirs: slot_A_black_flat / slot_B_white_flat / slot_C_grey_flat
    # (first found wins; also checks project-root-level flat_black etc.)
    cal_black_raw = None
    cal_white_raw = None
    cal_grey_raw  = None

    # Canonical flat folder name → which cal slot
    _CAL_BLACK_NAMES = frozenset(("slot_a_black_flat","flat_black","black_flat",
                                  "flats_black","cal_black","black_cal"))
    _CAL_WHITE_NAMES = frozenset(("slot_b_white_flat","flat_white","white_flat",
                                  "flats_white","cal_white","white_cal"))
    _CAL_GREY_NAMES  = frozenset(("slot_c_grey_flat","flat_grey","grey_flat",
                                  "flats_grey","cal_grey","grey_cal",
                                  "slot_c_gray_flat","flat_gray","gray_flat",
                                  "cal_gray","gray_cal"))

    # Root-level file stems → which cal slot
    _ROOT_BLACK = frozenset(("cal_black","black_cal","flat_black","black_flat"))
    _ROOT_WHITE = frozenset(("cal_white","white_cal","flat_white","white_flat"))
    _ROOT_GREY  = frozenset(("cal_grey","grey_cal","flat_grey","grey_flat",
                              "cal_gray","gray_cal","flat_gray","gray_flat"))

    for dirpath, dirnames, filenames in os.walk(str(proj)):
        dirnames.sort()
        dname      = os.path.basename(dirpath)
        dname_low  = dname.lower()

        # Check root-level loose RAW files (e.g. cal_black.ARW next to project folder)
        if dirpath == str(proj):
            for fname in sorted(filenames):
                stem, _, ext = fname.rpartition(".")
                if ext.lower() not in _RAW_EXTS:
                    continue
                stem_low = stem.lower()
                if stem_low in _ROOT_BLACK and cal_black_raw is None:
                    cal_black_raw = os.path.join(dirpath, fname)
                elif stem_low in _ROOT_WHITE and cal_white_raw is None:
                    cal_white_raw = os.path.join(dirpath, fname)
                elif stem_low in _ROOT_GREY and cal_grey_raw is None:
                    cal_grey_raw  = os.path.join(dirpath, fname)

        if dname == "slot_A_black":
            rot_dir = Path(dirpath).parent
            if rot_dir not in rot_dirs_seen:
                rot_dirs_seen.add(rot_dir)
                white_dir = rot_dir / "slot_B_white"
                grey_dir  = rot_dir / "slot_C_grey"
                if white_dir.is_dir():
                    blacks = _find_raws(Path(dirpath))
                    whites = _find_raws(white_dir)
                    greys  = _find_raws(grey_dir) if grey_dir.is_dir() else []
                    if blacks and whites:
                        out_dir = rot_dir / "matte"
                        for i, (b, w) in enumerate(zip(blacks, whites)):
                            g = greys[i] if i < len(greys) else None
                            pairs.append(MattePair(b, w, g,
                                out_dir / f"{b.stem}_matte.tif"))

        elif dname_low in _CAL_BLACK_NAMES and cal_black_raw is None:
            raws = _find_raws(Path(dirpath))
            if raws: cal_black_raw = str(raws[0])

        elif dname_low in _CAL_WHITE_NAMES and cal_white_raw is None:
            raws = _find_raws(Path(dirpath))
            if raws: cal_white_raw = str(raws[0])

        elif dname_low in _CAL_GREY_NAMES and cal_grey_raw is None:
            raws = _find_raws(Path(dirpath))
            if raws: cal_grey_raw = str(raws[0])

        elif dname == "matte":
            try:
                has_tif = any(
                    e.name.lower().endswith((".tif", ".tiff"))
                    and not e.name.startswith(".")
                    for e in os.scandir(dirpath)
                    if e.is_file(follow_symlinks=False)
                )
            except OSError:
                has_tif = False
            if has_tif:
                matte_dirs.append(Path(dirpath))

        elif dname == "stacked":
            try:
                # Exclude macOS resource forks ("._name.tif"). They are created on
                # external/FAT volumes, match the extension test, and are not TIFFs —
                # collecting them makes the COLMAP stage fail on an unreadable file.
                tifs = sorted(
                    Path(e.path) for e in os.scandir(dirpath)
                    if e.is_file(follow_symlinks=False)
                    and e.name.lower().endswith((".tif", ".tiff"))
                    and not e.name.startswith(".")
                )
            except OSError:
                tifs = []
            stacked_files.extend(tifs)

    cal_paths = {}
    if cal_black_raw: cal_paths["cal_black_path"] = cal_black_raw
    if cal_white_raw: cal_paths["cal_white_path"] = cal_white_raw
    if cal_grey_raw:  cal_paths["cal_grey_path"]  = cal_grey_raw

    return pairs, sorted(matte_dirs), sorted(stacked_files), cal_paths


# Keep old names for any callers that use them directly
def discover_pairs(proj): return _scan_project(proj)[0]
def discover_matte_dirs(proj): return _scan_project(proj)[1]
def discover_stacked_files(proj): return _scan_project(proj)[2]
def discover_cal_paths(proj): return _scan_project(proj)[3]

# ── Processing helpers ─────────────────────────────────────────────────────────

_IDENTITY_LUT = np.linspace(0, 1, 256).astype(np.float32)

def _monotone_cubic_lut(points):
    pts = sorted(points, key=lambda p: p[0])
    xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(xs)
    if n < 2:
        return _IDENTITY_LUT.copy()
    d = [(ys[i+1]-ys[i])/(xs[i+1]-xs[i]) for i in range(n-1)]
    m = [0.0]*n
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1]*d[i] <= 0: m[i] = 0.0
        else: m[i] = (d[i-1]+d[i])/2.0
    for i in range(n-1):
        if abs(d[i]) < 1e-9: m[i]=m[i+1]=0.0
        else:
            a=m[i]/d[i]; b=m[i+1]/d[i]
            if a**2+b**2 > 9: s=3/np.hypot(a,b); m[i]=s*a*d[i]; m[i+1]=s*b*d[i]
    t = np.linspace(0,1,256)
    lut = np.zeros(256, dtype=np.float32)
    for j,tv in enumerate(t):
        if tv <= xs[0]:   lut[j]=ys[0]; continue
        if tv >= xs[-1]:  lut[j]=ys[-1]; continue
        for i in range(n-1):
            if xs[i] <= tv <= xs[i+1]:
                h=(tv-xs[i])/(xs[i+1]-xs[i])
                # Cubic Hermite basis. h00 is h^2*(2h-3)+1 == 2h^3-3h^2+1.
                # It was h**3*(2*h-3)+1 (a quartic, one factor of h too many),
                # which breaks partition of unity: h00+h01 peaked at 1.26, so
                # the curve was pulled up to 26% high between control points
                # while still passing exactly through them.
                dx = xs[i+1]-xs[i]
                h00 = h**2*(2*h-3)+1
                h10 = h**3-2*h**2+h
                h01 = h**2*(3-2*h)
                h11 = h**3-h**2
                lut[j] = h00*ys[i] + h10*dx*m[i] + h01*ys[i+1] + h11*dx*m[i+1]
                break
    return lut

def _apply_alpha_lut(alpha, lut, solidify=0):
    idx = np.clip((alpha * 255).astype(np.int32), 0, 255)
    out = lut[idx]
    if solidify > 0:
        thresh = 1.0 - solidify / 100.0
        out = np.where(alpha >= thresh, 1.0, out)
    return out.clip(0, 1).astype(np.float32)

def _apply_sharpen(rgb8, amount):
    if amount <= 0: return rgb8
    from PIL import Image as PILImage, ImageFilter as IF
    pil = PILImage.fromarray(rgb8)
    blurred = pil.filter(IF.GaussianBlur(radius=1.5))
    sharp = PILImage.fromarray(np.clip(np.array(pil).astype(np.int16)*2 - np.array(blurred).astype(np.int16), 0, 255).astype(np.uint8))
    blend = PILImage.blend(pil, sharp, amount/200.0)
    return np.array(blend)

def _sharpen_linear(img_f32, amount):
    """USM sharpening on a float32 [0,1] array (works on linear-light data)."""
    if amount <= 0: return img_f32
    from PIL import Image as PILImage, ImageFilter as IF
    u8 = (np.clip(img_f32, 0, 1) * 255).astype(np.uint8)
    blurred = np.array(PILImage.fromarray(u8).filter(IF.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
    hp = img_f32.astype(np.float32) - blurred
    return np.clip(img_f32 + hp * (amount / 200.0), 0, 1)

def _match_size(arr, target):
    """Resize arr to match target's (h,w) if they differ — handles rawpy half_size rounding."""
    th, tw = target.shape[:2]
    ah, aw = arr.shape[:2]
    if ah == th and aw == tw:
        return arr
    from PIL import Image as _PIL
    # Use uint8 — PIL handles 3-channel uint8 correctly; uint16 silently collapses to grayscale
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    resized = np.array(_PIL.fromarray(u8).resize((tw, th), _PIL.BILINEAR), dtype=np.float32) / 255.0
    return resized

def _apply_ca_spatial(img_f32, ca):
    """Lateral CA correction: spatially scale R and B channels about the image center.
    ca=(r_scale, b_scale); >1 expands the channel, <1 contracts it.
    Apply consistently to all images (subject + cal) so alpha computation is correct."""
    if ca == (1.0, 1.0):
        return img_f32
    from PIL import Image as PILImage
    h, w = img_f32.shape[:2]
    result = img_f32.copy()
    for ch_idx, scale in [(0, ca[0]), (2, ca[1])]:
        if abs(scale - 1.0) < 1e-6:
            continue
        u8 = (np.clip(img_f32[:,:,ch_idx], 0, 1) * 255).astype(np.uint8)
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        resized = np.array(PILImage.fromarray(u8).resize((nw, nh), PILImage.BILINEAR), dtype=np.float32) / 255.0
        if scale > 1.0:
            left = (nw - w) // 2
            top  = (nh - h) // 2
            result[:,:,ch_idx] = resized[top:top+h, left:left+w]
        else:
            ch_out = np.zeros((h, w), dtype=np.float32)
            left = (w - nw) // 2
            top  = (h - nh) // 2
            ch_out[top:top+nh, left:left+nw] = resized
            result[:,:,ch_idx] = ch_out
    return result

def compute_preview(cb, cw, cg, bg_white, alpha_min, cal, grey_brightness, alpha_lut, sharpen=0, solidify=0, defringe=0, ca=(1.0, 1.0), exposure=0):
    # Exposure is a BRIGHTNESS control and must not enter the alpha solve.
    # Alpha is a ratio against the calibration reference:
    #     alpha = 1 - (cw - cb) / (cal_w - cal_b)
    # Scaling only the subject captures by s leaves the reference unscaled, so
    # s stops cancelling and the ratio becomes s -> background alpha = 1 - s
    # (at -1 EV the background would be 50% opaque). Exposure is therefore
    # folded into the display scale below, after the matte is solved.
    ev_scale = 2.0 ** (exposure / 100.0) if exposure else 1.0
    # CA spatial correction and sharpening DO belong before alpha: both change
    # channel registration / edge detail that the solve should see.
    if ca != (1.0, 1.0):
        cb = _apply_ca_spatial(cb, ca); cw = _apply_ca_spatial(cw, ca)
        if cg is not None: cg = _apply_ca_spatial(cg, ca)
    if sharpen > 0:
        cb = _sharpen_linear(cb, sharpen)
        cw = _sharpen_linear(cw, sharpen)
        if cg is not None: cg = _sharpen_linear(cg, sharpen)

    if cal and cal.available:
        cal_b = _match_size(cal.black_half, cb) if cal.black_half is not None else np.zeros_like(cb)
        cal_w = _match_size(cal.white_half, cb) if cal.white_half is not None else bg_white
        # Apply same spatial CA correction to cal images — spatially aligning all channels
        # consistently is required so the alpha ratio (cw-cb)/(cal_w-cal_b) is correct.
        if ca != (1.0, 1.0):
            cal_b = _apply_ca_spatial(cal_b, ca); cal_w = _apply_ca_spatial(cal_w, ca)
        div = np.maximum(cal_w - cal_b, 0.01)
        raw_alpha = 1.0 - (cw - cb) / div
        raw_alpha = raw_alpha.mean(axis=2)
        raw_alpha_01 = np.clip(raw_alpha, 0, 1)
        # Normalize raw_alpha to [0,1] relative to the [alpha_min, 1] range before
        # applying the LUT. This means curve input 0 == alpha_min and input 1 == 1.
        # Combined with lut[0]=0 enforced in the event handler, calibration-noise
        # pixels just above alpha_min have nearly-zero normalized input and can never
        # be raised to high alpha by the curve.
        _span = max(1.0 - alpha_min, 1e-4)
        alpha_norm = np.clip((raw_alpha_01 - alpha_min) / _span, 0.0, 1.0)
        alpha = _apply_alpha_lut(alpha_norm, alpha_lut, solidify)
        alpha = np.where(raw_alpha_01 < alpha_min, 0.0, alpha)
        # Foreground extraction: use smooth blend between cb (stable at low alpha)
        # and the triangulation formula (correct at high alpha). A hard threshold
        # creates a visible ring artifact in the composite where the formula switches.
        a3 = raw_alpha_01[:,:,None]
        safe_denom = np.maximum(a3, 0.10)  # clamp denominator to reduce amplification
        fg_extracted = np.clip((cb - cal_b * (1 - a3)) / safe_denom, 0, 1)
        # Smoothstep blend: 0→cb at a≤alpha_min, 1→extracted at a≥0.40
        blend_lo = float(alpha_min) + 0.01
        blend_hi = 0.40
        t = np.clip((a3 - blend_lo) / max(blend_hi - blend_lo, 1e-4), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        fg_lin = np.where(a3 < alpha_min,
                          0.0,
                          t * fg_extracted + (1.0 - t) * cb).astype(np.float32)
    else:
        diff = np.clip(cw - cb, 0, None)
        raw_alpha = 1.0 - (diff / np.maximum(bg_white, 1e-6)).mean(axis=2)
        raw_alpha_01 = np.clip(raw_alpha, 0, 1)
        _span = max(1.0 - alpha_min, 1e-4)
        alpha_norm = np.clip((raw_alpha_01 - alpha_min) / _span, 0.0, 1.0)
        alpha = _apply_alpha_lut(alpha_norm, alpha_lut, solidify)
        alpha = np.where(raw_alpha_01 < alpha_min, 0.0, alpha)
        fg_lin = cb.copy()
    if defringe > 0:
        fg_lin = _apply_defringe(fg_lin, defringe)

    # Exposure normalization for display: camera exposed conservatively so linear
    # values rarely reach 1.0. Scale so the white reference p99 maps to 1.0.
    white_ref = (cal.white_half if (cal and cal.white_half is not None) else cw)
    wp = float(np.percentile(white_ref, 99.5))
    disp_scale = min(1.0 / wp, 10.0) if wp > 0.05 else 1.0
    # User exposure rides on the normalisation as a single multiply, so there is
    # only one clip and the alpha channel is untouched.
    disp_scale *= ev_scale
    def _ds(x): return np.clip(x * disp_scale, 0, 1)

    fg_disp = _ds(fg_lin)
    fg8 = (_lin_to_srgb(fg_disp) * 255).clip(0, 255).astype(np.uint8)
    # No post-sharpen of fg8 — sharpening was already applied to source images above.
    alpha8 = (alpha * 255).clip(0, 255).astype(np.uint8)
    checker = _checkerboard(fg8.shape[0], fg8.shape[1])
    a = alpha[:,:,None]
    comp = (fg8.astype(np.float32) * a + checker.astype(np.float32) * (1-a)).clip(0,255).astype(np.uint8)
    grey_lin = _resolve_grey_bg(cw, grey_brightness, cg if (cal and cal.grey_half is not None) else None)
    grey8 = (_lin_to_srgb(_ds(grey_lin)) * 255).clip(0, 255).astype(np.uint8)
    return {"black": (_lin_to_srgb(_ds(cb))*255).clip(0,255).astype(np.uint8),
            "white": (_lin_to_srgb(_ds(cw))*255).clip(0,255).astype(np.uint8),
            "grey":  grey8,
            "alpha": np.stack([alpha8,alpha8,alpha8],axis=2),
            "composite": comp,
            "fg8": fg8,          # sharpened gamma-encoded foreground for compositing
            "fg_lin": fg_disp, "alpha_raw": alpha}

def derive_calibration(pairs, max_frames=30, on_log=None, should_run=None):
    """Reconstruct calibration frames from the captures themselves.

    A calibration shot is an empty frame: it records what the background looks
    like with no specimen — lens vignetting, phone-screen falloff, the sensor
    noise floor. Without one the solve falls back to a flat constant for white
    and exact zero for black, so

        raw_alpha = 1 - (cw - cb) / (cal_w - cal_b)

    lands slightly under 1 across the background instead of exactly 1. That is
    the faint haze that no amount of raising the opacity floor removes cleanly,
    because the floor hides the error rather than correcting it, and lifting it
    far enough eats the soft edges — antennae, wing margins — that the matte
    exists to preserve.

    A synthetic flat image cannot help: it is arithmetically identical to the
    fallback. But the real background IS recoverable from the captures, because
    the specimen occupies a small part of the frame and ROTATES. Sample frames
    spread across rotations and any given background pixel is unobstructed in
    most of them, so a per-pixel median rejects the specimen as an outlier and
    leaves the true background.

    Frames are sampled across rotations rather than within one, so the subject
    lands somewhere different in each. Half-size is deliberate: it matches what
    compute_preview solves at, and keeps a 30-frame stack near 1 GB instead of
    tens.
    """
    log = on_log or (lambda m: None)
    alive = should_run or (lambda: True)
    if not pairs:
        return None, "no pairs to derive from"

    # Spread the sample across rotations. Consecutive frames in one rotation
    # differ only by focus, so the subject sits in the same place and the
    # median would keep it.
    by_rot = {}
    for pr in pairs:
        by_rot.setdefault(_frame_key(pr.black), []).append(pr)
    rots = sorted(by_rot)
    if len(rots) < 3:
        return None, (f"only {len(rots)} rotation(s) found — the subject barely moves "
                      "between them, so a median cannot separate it from the background")

    picks = []
    r = 0
    while len(picks) < max_frames and r < max_frames * 3:
        rot = rots[r % len(rots)]
        grp = by_rot[rot]
        idx = (r // len(rots))
        if idx < len(grp):
            picks.append(grp[idx])
        r += 1
    if len(picks) < 5:
        return None, f"only {len(picks)} usable frames"

    def _stack_background(get_path, label, bright_bg):
        """Recover the background plane for one slot.

        Two stages, and the second is not optional.

        Per-pixel PERCENTILE across frames, not median. The specimen is pinned
        at frame centre and rotates in place — it does not translate — so the
        central pixels are covered in most frames and a median returns the
        specimen there. A high percentile on a white background (the specimen is
        darker) or a low percentile on black (the specimen is brighter) survives
        the specimen covering a pixel in most, though not all, frames.

        Then FIT A SMOOTH SURFACE and use the fit everywhere. Whatever the
        percentile does, pixels the specimen never uncovers cannot be measured
        at all, and using a measured value there would make the specimen itself
        transparent. Vignetting and screen falloff are low-frequency, so a
        quadratic in x and y is a good model: it is fitted on the periphery,
        where the background genuinely is visible, and evaluated across the
        middle. That also smooths away sensor noise, which is the other thing a
        calibration frame should not carry.
        """
        frames, chroma = [], []
        for i, pr in enumerate(picks):
            if not alive():
                return None
            path = get_path(pr)
            if path is None or not Path(path).exists():
                continue
            try:
                a = _read_raw_linear(Path(path), half_size=True)
            except Exception as e:
                log(f"    skip {Path(path).name}: {e}")
                continue
            frames.append((a * 65535.0).clip(0, 65535).astype(np.uint16))
            m = a.reshape(-1, 3).mean(axis=0)
            chroma.append(float(m[0] / max(m[2], 1e-6)))
            if (i + 1) % 5 == 0:
                log(f"    {label}: read {len(frames)}/{len(picks)}")
        if len(frames) < 5:
            return None

        # Drop frames whose background colour disagrees with the rest. A screen
        # that drifted partway through a scan would otherwise poison every
        # frame after it.
        med_c = float(np.median(chroma))
        keep = [f for f, c in zip(frames, chroma)
                if med_c > 0 and abs(math.log2(max(c, 1e-6) / med_c)) <= 0.15]
        if 5 <= len(keep) < len(frames):
            log(f"    {label}: dropped {len(frames)-len(keep)} frame(s) whose background "
                f"colour disagreed (drifted screen?)")
            frames = keep

        stack = np.stack(frames, axis=0)
        h, w = stack.shape[1], stack.shape[2]
        pct = 90.0 if bright_bg else 10.0
        meas = np.empty(stack.shape[1:], np.float32)
        band = max(64, h // 12)
        for y in range(0, h, band):
            meas[y:y+band] = np.percentile(stack[:, y:y+band], pct, axis=0).astype(np.float32)
        del stack
        meas /= 65535.0

        # Fit a quadratic surface per channel on the periphery, where the
        # background is actually visible, then evaluate it everywhere.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
        ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
        edge = (np.abs(nx) > 0.55) | (np.abs(ny) > 0.55)       # outer frame only
        A_all = np.stack([np.ones_like(nx), nx, ny, nx*nx, ny*ny, nx*ny], axis=-1)
        A_fit = A_all[edge]
        out = np.empty_like(meas)
        coefs = []
        for c in range(3):
            coef, *_ = np.linalg.lstsq(A_fit, meas[edge][:, c], rcond=None)
            coefs.append(coef)
            out[:, :, c] = A_all @ coef
        resid = float(np.abs((A_all[edge] @ np.linalg.lstsq(
            A_fit, meas[edge][:, 1], rcond=None)[0]) - meas[edge][:, 1]).mean())
        log(f"    {label}: fitted background surface, edge residual {resid:.5f}")
        return np.clip(out, 0.0, 1.0), coefs

    def _eval_surface(coefs, shape_hw):
        """Evaluate a fitted background surface at any resolution.

        The fit is a quadratic in normalised coordinates, so the full-size plane
        is evaluated exactly rather than upsampled from the half-size one.
        Both are needed: the solve reads *_half, but CalImages.available — which
        gates whether the solve uses the calibration at all — requires *_full.
        """
        h, w = shape_hw
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
        ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
        A = np.stack([np.ones_like(nx), nx, ny, nx*nx, ny*ny, nx*ny], axis=-1)
        out = np.empty((h, w, 3), np.float32)
        for c in range(3):
            out[:, :, c] = A @ coefs[c]
        return np.clip(out, 0.0, 1.0)

    log(f"  deriving calibration from {len(picks)} frames across {len(rots)} rotations…")
    cal = CalImages()
    rw = _stack_background(lambda pr: pr.white, "white", bright_bg=True)
    if rw is None:
        return None, "could not read enough white frames"
    rb = _stack_background(lambda pr: pr.black, "black", bright_bg=False)
    if rb is None:
        return None, "could not read enough black frames"
    rg = (_stack_background(lambda pr: pr.grey, "grey", bright_bg=True)
          if any(pr.grey for pr in picks) else None)
    w, w_coef = rw
    b, b_coef = rb
    g, g_coef = (rg if rg is not None else (None, None))

    cal.white_half, cal.black_half = w, b
    if g is not None:
        cal.grey_half = g

    # Full-size planes as well. The solve reads *_half, but it is gated behind
    # CalImages.available, which requires black_full AND white_full — so setting
    # only the half planes produced a correct calibration that the solve then
    # ignored entirely, with no visible effect and an empty status line.
    try:
        full_hw = _read_raw_linear(picks[0].white, half_size=False).shape[:2]
        cal.white_full = _eval_surface(w_coef, full_hw)
        cal.black_full = _eval_surface(b_coef, full_hw)
        if g_coef is not None:
            cal.grey_full = _eval_surface(g_coef, full_hw)
    except Exception as e:
        log(f"    full-size planes skipped: {e}")
    rng = float(np.mean(w) - np.mean(b))
    return cal, (f"derived from {len(picks)} frames · mean white {np.mean(w):.4f}, "
                 f"black {np.mean(b):.4f}, separation {rng:.4f}")


class MattePipeline:
    def __init__(self, pairs, bg_white, alpha_min, cal=None, grey_brightness=50,
                 alpha_lut=None, sharpen=0, solidify=0, ca=(1.0,1.0), defringe=0,
                 exposure=0,
                 on_progress=None, on_log=None, on_done=None, on_preview=None):
        self._pairs=pairs; self._bg=bg_white; self._amin=alpha_min
        self._cal=cal; self._grey_br=grey_brightness
        self._lut=alpha_lut if alpha_lut is not None else _IDENTITY_LUT.copy()
        self._sharpen=sharpen; self._solidify=solidify; self._ca=ca; self._defringe=defringe
        self._exposure=exposure
        self._on_prog=on_progress or (lambda n,t:None)
        self._on_log=on_log or (lambda m:None)
        self._on_done=on_done or (lambda ok,m:None)
        self._on_prev=on_preview or (lambda i,imgs,r:None)
        self._cancelled=False

    def cancel(self): self._cancelled=True

    def run(self):
        try:
            n=len(self._pairs)
            for i, pair in enumerate(self._pairs):
                if self._cancelled: self._on_done(False,"Cancelled"); return
                self._on_log(f"Frame {i+1}/{n}: {pair.black.name}")
                self._on_prog(i,n)
                try:
                    cb = _read_raw_linear(pair.black, ca_scale=self._ca)
                    cw = _read_raw_linear(pair.white, ca_scale=self._ca)
                    cg = _read_raw_linear(pair.grey) if pair.grey else None
                    # Exposure is brightness only — it must NOT enter the alpha
                    # solve, or the background stops resolving to alpha 0.
                    # See the note in compute_preview; applied with disp_scale below.
                    ev_scale = 2.0 ** (self._exposure / 100.0) if self._exposure else 1.0
                    # Linear sharpening on source images (matches compute_preview)
                    if self._sharpen > 0:
                        cb = _sharpen_linear(cb, self._sharpen)
                        cw = _sharpen_linear(cw, self._sharpen)
                        if cg is not None: cg = _sharpen_linear(cg, self._sharpen)
                    if self._cal and self._cal.available:
                        cal_b=self._cal.black_full; cal_w=self._cal.white_full
                        div=np.maximum(cal_w-cal_b,0.01)
                        raw_a=(1.0-(cw-cb)/div).mean(axis=2)
                        raw_a01=np.clip(raw_a,0,1)
                        _span=max(1.0-self._amin,1e-4)
                        alpha_norm=np.clip((raw_a01-self._amin)/_span,0.0,1.0)
                        alpha=_apply_alpha_lut(alpha_norm,self._lut,self._solidify)
                        alpha=np.where(raw_a01<self._amin,0.0,alpha)
                        a3=raw_a01[:,:,None]
                        safe_denom=np.maximum(a3,0.10)
                        fg_extracted=np.clip((cb-cal_b*(1-a3))/safe_denom,0,1)
                        blend_lo=float(self._amin)+0.01; blend_hi=0.40
                        t=np.clip((a3-blend_lo)/max(blend_hi-blend_lo,1e-4),0.0,1.0)
                        t=t*t*(3.0-2.0*t)
                        fg_lin=np.where(a3<self._amin,0.0,t*fg_extracted+(1.0-t)*cb).astype(np.float32)
                    else:
                        diff=np.clip(cw-cb,0,None)
                        raw_a=1.0-(diff/np.maximum(self._bg,1e-6)).mean(axis=2)
                        raw_a01=np.clip(raw_a,0,1)
                        _span=max(1.0-self._amin,1e-4)
                        alpha_norm=np.clip((raw_a01-self._amin)/_span,0.0,1.0)
                        alpha=_apply_alpha_lut(alpha_norm,self._lut,self._solidify)
                        alpha=np.where(raw_a01<self._amin,0.0,alpha)
                        fg_lin=cb.copy()
                    if self._defringe>0: fg_lin=_apply_defringe(fg_lin,self._defringe)
                    # White-point normalization. MUST use the same reference image as
                    # compute_preview (white_half), not white_full: half-size RAW demosaic
                    # averages pixels down, lowering the p99.5 and so raising disp_scale.
                    # Using white_full here made every export darker than the preview.
                    white_ref=None
                    if self._cal:
                        white_ref=(self._cal.white_half if self._cal.white_half is not None
                                   else self._cal.white_full)
                    if white_ref is None: white_ref=cw
                    wp=float(np.percentile(white_ref,99.5))
                    disp_scale=min(1.0/wp,10.0) if wp>0.05 else 1.0
                    disp_scale*=ev_scale        # exposure rides here, not on the solve
                    fg_norm=np.clip(fg_lin*disp_scale,0,1)
                    fg_srgb=_lin_to_srgb(fg_norm)
                    a_f=alpha[:,:,None]
                    a16=(alpha*65535).clip(0,65535).astype(np.uint16)
                    # Pre-multiplied (associated) alpha: Affinity and most compositors
                    # detect ASSOCALPHA reliably and open the file as a transparent layer.
                    # Straight alpha (UNASSALPHA) was being read as an opaque Background layer.
                    rgb16=(fg_srgb*a_f*65535).clip(0,65535).astype(np.uint16)
                    rgba=np.concatenate([rgb16,a16[:,:,None]],axis=2)
                    pair.out_path.parent.mkdir(exist_ok=True)
                    tifffile.imwrite(str(pair.out_path),rgba,photometric='rgb',compression='deflate',
                                     extrasamples=(1,))
                    if i<3:
                        # cb/cw/cg are already sharpened, so pass sharpen=0. Exposure is
                        # NOT baked into them any more, so pass the real value through.
                        qcb=_quarter_size(cb); qcw=_quarter_size(cw)
                        qcg=_quarter_size(cg) if cg is not None else None
                        imgs=compute_preview(qcb,qcw,qcg,self._bg,self._amin,self._cal,self._grey_br,self._lut,0,self._solidify,self._defringe,exposure=self._exposure)
                        self._on_prev(i,imgs,(qcb,qcw,qcg))
                except Exception as e:
                    self._on_log(f"  ERROR: {e}")
                    logging.exception(e)
            self._on_prog(n,n)
            self._on_log(f"Done — {n} mattes written.")
            self._on_done(True,"")
        except Exception as e:
            self._on_done(False,str(e))

class _SSProcess:
    """Minimal shinestacker `process` shim.

    BaseStackAlgo calls back into a host process object for progress and for
    cancellation. Only CHECK_RUNNING is load-bearing: returning False aborts
    the run. Everything else is progress chatter we forward to the log.
    """
    def __init__(self, on_log, is_running, input_path, output_path, on_frac=None):
        self.id = 0
        self.name = "mattepro"
        self.input_path = input_path
        self.output_path = output_path
        self._log = on_log
        self._running = is_running
        # Reports 0..1 progress WITHIN one stack, parsed from shinestacker's own
        # frame messages. Without it the bar sits at 0/1 for the whole run (the
        # outer counter only ticks per directory) and looks frozen.
        self._on_frac = on_frac or (lambda f: None)

    def callback(self, kind, *a, **kw):
        try:
            from shinestacker.config.constants import constants
            if kind == constants.CALLBACK_CHECK_RUNNING:
                return bool(self._running())
        except Exception:
            pass
        return None

    _FRAME_RE = re.compile(r"frame\s+(\d+)\s*/\s*(\d+)")

    def sub_message_r(self, msg, level=None):
        try:
            txt = re.sub(r"\x1b\[[0-9;]*m", "", str(msg)).strip()
            if not txt:
                return
            self._log(f"    {txt}")
            m = self._FRAME_RE.search(txt)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if tot > 0:
                    # Frame loading is roughly the first 70% of a stack; the
                    # pyramid fuse/collapse tail is the rest.
                    self._on_frac(min(0.70, 0.70 * cur / tot))
            elif "fusing pyramids" in txt:
                self._on_frac(0.80)
            elif "pyramids fusion completed" in txt:
                self._on_frac(0.90)
            elif "collapsing pyramid" in txt:
                self._on_frac(0.95)
        except Exception:
            pass


class StackPipeline:
    def __init__(self, matte_dirs, on_progress=None, on_log=None, on_done=None, on_preview=None,
                 denoise=0, sharpen=0, sharpen_radius=1.0, kernel_size=5, gen_kernel=0.4, min_size=32,
                 alpha_lut=None, halo_fix=True, halo_pct=75, align=True,
                 defocus_filter=True, defocus_keep=0.15,
                 defocus_scan="ends"):
        self._dirs=matte_dirs; self._on_prog=on_progress or (lambda n,t:None)
        self._on_log=on_log or (lambda m:None); self._on_done=on_done or (lambda ok,m:None)
        self._on_prev=on_preview or (lambda imgs:None)
        self._denoise=denoise; self._sharpen=sharpen; self._shr=sharpen_radius
        self._kernel=kernel_size; self._genk=gen_kernel; self._minsize=min_size
        self._lut=alpha_lut if alpha_lut is not None else _IDENTITY_LUT.copy()
        self._halo_fix=halo_fix
        self._halo_pct=int(halo_pct)
        self._align=align
        self._defocus_filter=bool(defocus_filter)
        self._defocus_keep=float(defocus_keep)
        self._defocus_scan=str(defocus_scan)
        self._alpha_stack=None
        self._cancelled=False

    # ── Defocused-frame rejection ────────────────────────────────────────────
    _FOCUS_WIN  = 15       # px; local sharpness window — small, to catch a wing tip
    _FOCUS_TOPN = 40       # average this many best pixels, so one hot pixel cannot win

    def _focus_score(self, path):
        """Sharpness of the SHARPEST SMALL PATCH inside the subject.

        The question is not "is this frame sharp" but "does this frame contain
        any real detail worth fusing". A stack frame typically starts with just
        the tip of a wing or a leg in focus and everything else soft — that
        frame is valuable, and any statistic that averages over the subject
        throws it away.

        So: local sharpness energy over a small window, then the mean of the
        highest-responding pixels. A tiny in-focus tip lights up its own window
        and survives; a frame whose focus plane missed the specimen entirely has
        no window anywhere above the noise floor.

        Scored INSIDE THE MATTE ONLY. These frames are RGBA with a transparent
        background, so a whole-frame measure is dominated by the cutout edge — a
        hard step from subject to nothing — and every frame scores high whatever
        its focus. The alpha mask is eroded so that edge is excluded too.

        Returns a raw score; the keep/drop decision is relative to the rest of
        the stack (see _filter_defocused), because absolute sharpness depends on
        subject, magnification and lighting.
        """
        try:
            import cv2
            arr = tifffile.imread(str(path))
        except Exception:
            return None
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return None
        a  = arr.astype(np.float32)
        mx = 65535.0 if arr.dtype == np.uint16 else 255.0
        rgb = a[:, :, :3] / mx
        lum = (0.2126*rgb[:, :, 0] + 0.7152*rgb[:, :, 1] + 0.0722*rgb[:, :, 2])

        if arr.shape[2] >= 4:
            alpha = a[:, :, 3] / mx
            mask  = (alpha > 0.9).astype(np.uint8)
            mask  = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=2)
        else:
            mask = np.ones(lum.shape, np.uint8)
        if int(mask.sum()) < 256:
            return 0.0          # nothing opaque enough to judge

        # Local sharpness energy: Laplacian squared, averaged over a small
        # window. The window smooths away single-pixel noise spikes without
        # diluting a genuinely sharp edge.
        lap = cv2.Laplacian(lum, cv2.CV_32F, ksize=3)
        eng = cv2.boxFilter(lap * lap, -1, (self._FOCUS_WIN, self._FOCUS_WIN),
                            normalize=True)

        # Only consider windows fully inside the subject.
        inner = cv2.erode(mask, np.ones((self._FOCUS_WIN, self._FOCUS_WIN), np.uint8))
        vals  = eng[inner > 0]
        if vals.size == 0:
            return 0.0

        n = min(self._FOCUS_TOPN, vals.size)
        return float(np.mean(np.partition(vals, -n)[-n:]))

    def _filter_defocused(self, tiffs):
        """Drop frames with no in-focus region. Returns (kept, dropped, scores).

        The rail travels the same start->end on every rotation, but a specimen
        is not the same depth from every angle, so some frames land with the
        focus plane entirely off the subject. Those carry no detail for the
        fusion to select — only grain, which a Laplacian pyramid is happy to
        treat as structure.

        Scans INWARD FROM EACH END rather than scoring every frame. The rail
        sweeps monotonically through the specimen, so the focus plane can only
        miss it at the extremes: once a frame at the near end shows detail,
        every frame deeper in does too. Scoring is not cheap — 0.92s on a
        6000x4000 RGBA frame, which is 69 minutes for a 100-rotation batch at
        45 frames each — so walking in from the ends and stopping at the first
        keeper takes that to about 15 minutes.

        Set defocus_scan="all" to score every frame instead. That is the only
        way to catch a bad frame in the MIDDLE of a stack (a matting failure, a
        subject that shifted), which the ends-inward walk assumes cannot happen.

        The threshold is a fraction of the sharpest frame seen, since absolute
        sharpness is meaningless across subjects and magnifications. A floor of
        three frames is always kept: a stack that scores flat means the measure
        is unreliable there, not that every frame is worthless.
        """
        n = len(tiffs)
        scores = [None] * n

        def score(i):
            if scores[i] is None:
                scores[i] = self._focus_score(tiffs[i])
            return scores[i] or 0.0

        # ── Reference peak ────────────────────────────────────────────────────
        # Probe the middle, where the focus plane is inside the specimen by
        # assumption, to learn what "sharp" is worth for this stack.
        if getattr(self, "_defocus_scan", "ends") == "ends" and n >= 8:
            probes = sorted({n // 2, n // 3, (2 * n) // 3})
            peak = max(score(i) for i in probes)
            if peak <= 0.0:
                # Middle looks flat — the assumption does not hold here, so fall
                # back to scoring everything rather than guessing.
                for i in range(n):
                    score(i)
                return self._decide(tiffs, scores)

            thr = peak * self._defocus_keep
            # Never eat more than this much of the stack from one end; beyond it
            # something other than defocus is going on.
            limit = max(1, int(n * 0.4))

            lead = 0
            while lead < limit and score(lead) < thr:
                lead += 1
            tail = 0
            while tail < limit and score(n - 1 - tail) < thr:
                tail += 1

            if lead >= limit or tail >= limit:
                self._on_log("  focus scan inconclusive from the ends — "
                             "scoring every frame")
                for i in range(n):
                    score(i)
                return self._decide(tiffs, scores)

            drop = [tiffs[i] for i in range(lead)] + \
                   [tiffs[n - 1 - i] for i in range(tail)]
            keep = [t for t in tiffs if t not in drop]
            if len(keep) < 3:
                return list(tiffs), [], scores
            return keep, drop, scores

        # ── Exhaustive ────────────────────────────────────────────────────────
        for i in range(n):
            score(i)
        return self._decide(tiffs, scores)

    def _decide(self, tiffs, scores):
        """Keep/drop from a fully-scored stack, threshold relative to its peak."""
        valid = [s for s in scores if s is not None and s > 0]
        if len(valid) < 3:
            return list(tiffs), [], scores
        thr = max(valid) * self._defocus_keep
        keep, drop = [], []
        for t, s in zip(tiffs, scores):
            (keep if (s is None or s >= thr) else drop).append(t)
        if len(keep) < 3:
            return list(tiffs), [], scores
        return keep, drop, scores

    _ALIGN_DS = 4      # match features at 1/4 res; the transform scales back up

    def _align_gray(self, img):
        """8-bit downsampled luminance for feature matching."""
        import cv2
        g=(img[:,:,:3].mean(axis=2)*255).clip(0,255).astype(np.uint8)
        return cv2.resize(g,None,fx=1.0/self._ALIGN_DS,fy=1.0/self._ALIGN_DS,
                          interpolation=cv2.INTER_AREA)

    def _align_transform(self, detector, matcher, ref_kp, ref_des, img):
        """Similarity transform mapping img onto the reference, at full res.

        Partial-affine (translation + rotation + uniform scale) is the right model
        for focus breathing; a full homography would over-fit on a subject with
        real depth.
        """
        import cv2
        try:
            kp,des=detector.detectAndCompute(self._align_gray(img),None)
            if des is None or ref_des is None or len(kp)<10: return None
            pairs=matcher.knnMatch(des,ref_des,k=2)
            good=[a for a,b in pairs if a.distance<0.75*b.distance]
            if len(good)<12: return None
            src=np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst=np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M,inl=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                                              ransacReprojThreshold=3)
            if M is None or inl is None or int(inl.sum())<10: return None
            # Rescale the translation from the downsampled frame to full res.
            M=M.copy(); M[:,2]*=self._ALIGN_DS
            return M
        except Exception:
            return None

    def _to_straight(self, tiffs, tmpdir):
        """Un-premultiply + edge-extend the mattes for fusion.

        The pyramid fuses RGB and alpha independently, so premultiplied input
        comes out violating max(RGB) <= A: fused RGB keeps energy from frames
        where the subject was opaque while fused alpha lands low, and dividing
        gives a huge overbright halo (measured: 48% of semi-transparent pixels
        invalid, median 5.5x too bright). Fusing STRAIGHT colour and
        re-premultiplying afterwards makes the invariant hold by construction.

        Colour is flood-extended past the matte edge so the pyramid never sees a
        hard colour cliff there; beyond the extended band alpha is 0, so the
        extended values vanish when we re-premultiply.
        """
        from scipy import ndimage
        out=[]
        # Collect per-frame alpha (uint8 is ample for a percentile ceiling) while
        # we are already reading each frame — no extra I/O.
        self._alpha_stack=[]
        # Focus bracketing changes magnification ("focus breathing"): on this data
        # the frames span an 8.8% scale change with ~170px of translation, so the
        # union of silhouettes is much wider than any single frame and the fusion
        # inherits that as a halo. Register every frame to the middle one first.
        ref_kp=ref_des=None; detector=matcher=None
        if self._align and len(tiffs)>2:
            try:
                import cv2
                detector=cv2.AKAZE_create()
                matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
                ri=len(tiffs)//2
                rimg=tifffile.imread(str(tiffs[ri])).astype(np.float32)/65535.0
                rg=self._align_gray(rimg)
                ref_kp,ref_des=detector.detectAndCompute(rg,None)
                self._on_log(f"  aligning to frame {ri+1}/{len(tiffs)} "
                             f"({0 if ref_kp is None else len(ref_kp)} keypoints)")
                del rimg,rg
            except Exception as e:
                self._on_log(f"  align unavailable ({e}) — continuing unaligned")
                ref_kp=ref_des=None
        n_aligned=0
        for k,t in enumerate(tiffs):
            if self._cancelled: break
            img=tifffile.imread(str(t)).astype(np.float32)/65535.0
            if img.ndim!=3 or img.shape[2]!=4:
                out.append(str(t)); continue
            if ref_des is not None and k!=len(tiffs)//2:
                M=self._align_transform(detector,matcher,ref_kp,ref_des,img)
                if M is not None:
                    import cv2
                    h,w=img.shape[:2]
                    img=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                    n_aligned+=1
            if k==len(tiffs)-1 and ref_des is not None:
                self._on_log(f"  aligned {n_aligned}/{len(tiffs)-1} frames")
            self._alpha_stack.append((img[:,:,3]*255).clip(0,255).astype(np.uint8))
            A=img[:,:,3]; a3=np.maximum(A,1e-4)[:,:,None]
            straight=np.divide(img[:,:,:3],a3,where=a3>1e-4,
                               out=np.zeros_like(img[:,:,:3]))
            straight=np.clip(straight,0,1)
            known=A>0.05
            if known.any() and not known.all():
                idx=ndimage.distance_transform_edt(
                    ~known,return_distances=False,return_indices=True)
                straight=straight[idx[0],idx[1]]
            p=Path(tmpdir)/f"s{k:04d}.tif"
            tifffile.imwrite(str(p),
                np.concatenate([(straight*65535).clip(0,65535).astype(np.uint16),
                                (A*65535).clip(0,65535).astype(np.uint16)[:,:,None]],axis=2),
                photometric='rgb',compression='deflate',extrasamples=(1,))
            out.append(str(p))
        return out

    def cancel(self): self._cancelled=True

    def run(self):
        try:
            from shinestacker import PyramidAutoStack
            n=len(self._dirs)
            n_ok=0; n_fail=0
            for i,d in enumerate(self._dirs):
                if self._cancelled: self._on_done(False,"Cancelled"); return
                self._on_log(f"Stacking {d.parent.name} ({i+1}/{n})…")
                self._on_prog(i,n)
                tiffs=sorted(t for t in d.glob("*.tif*") if not t.name.startswith("."))
                if not tiffs: self._on_log("  No TIFFs — skip"); continue

                if self._defocus_filter and len(tiffs) >= 6:
                    self._on_log("  scoring focus…")
                    _kept, _dropped, _scores = self._filter_defocused(tiffs)
                    if _dropped:
                        _sv = [x for x in _scores if x]
                        self._on_log(
                            f"  dropped {len(_dropped)} defocused frame(s) "
                            f"(peak {max(_sv):.4g}, threshold "
                            f"{max(_sv)*self._defocus_keep:.4g})")
                        for _d in _dropped[:8]:
                            self._on_log(f"    – {_d.name}")
                        if len(_dropped) > 8:
                            self._on_log(f"    … and {len(_dropped)-8} more")
                        tiffs = _kept
                    else:
                        self._on_log("  all frames carry in-focus detail")

                self._on_log(f"  {len(tiffs)} frame(s) to fuse")
                # Sub-progress within this directory, so a single long stack does
                # not leave the bar pinned at 0%.
                def _frac(f, _i=i, _n=n):
                    self._on_prog(min(_n, _i + f), _n)
                tmpdir=None
                try:
                    out_dir=d.parent/"stacked"; out_dir.mkdir(exist_ok=True)
                    out_path=out_dir/f"{d.parent.name}_stacked.tif"
                    inputs=[str(t) for t in tiffs]
                    if self._halo_fix:
                        import tempfile as _tf
                        tmpdir=_tf.mkdtemp(prefix="mattepro-straight-")
                        self._on_log("  un-premultiplying + edge-extending (halo fix)…")
                        inputs=self._to_straight(tiffs,tmpdir)
                        if self._cancelled:
                            self._on_done(False,"Cancelled"); return
                    # shinestacker reads the frames itself: init(filenames) then
                    # focus_stack(). PyramidAutoStack.supports_alpha is True, so the
                    # RGBA matte alpha is carried through fusion.
                    stk=PyramidAutoStack(kernel_size=self._kernel,gen_kernel=self._genk,
                                         min_size=self._minsize)
                    # Order matters: set_process must precede init() (init_implementation
                    # raises if process is None), and set_output_filename must follow it
                    # (it delegates to the implementation created inside init()).
                    stk.set_process(_SSProcess(self._on_log,lambda: not self._cancelled,
                                               str(d),str(out_dir),on_frac=_frac))
                    stk.init(inputs)
                    stk.set_output_filename(str(out_path))
                    result=stk.focus_stack()
                    if result is None: raise RuntimeError("focus_stack returned no data")
                    if result.dtype!=np.uint16:
                        result=np.clip(result,0,65535).astype(np.uint16)
                    if result.ndim==3 and result.shape[2]==3:
                        a=np.full(result.shape[:2]+(1,),65535,np.uint16)
                        result=np.concatenate([result,a],axis=2)
                    # shinestacker is OpenCV-based and works in BGR/BGRA: read_img()
                    # does RGBA->BGRA on load, and its own writer does BGR->RGB on
                    # save. We save with tifffile directly, so we must undo it here
                    # or red and blue come out swapped (warm subjects render blue).
                    result = result[:, :, [2, 1, 0, 3]]
                    if self._halo_fix:
                        rf=result.astype(np.float32)/65535.0
                        af=rf[:,:,3]
                        # Alpha is Laplacian-fused too, which unions the sharp
                        # silhouette with the WIDER defocused ones and overshoots
                        # past every input frame.
                        #
                        # Cap fused alpha at a per-frame percentile, EXCEPT where
                        # some frame shows the pixel genuinely opaque.
                        #
                        # A thin appendage is opaque, so in the 2-3 frames where it
                        # is sharp its alpha exceeds 0.9. Defocus spread averages
                        # toward zero and essentially never does. Measured on real
                        # data with spatially-defined regions: requiring alpha>0.9
                        # in at least one frame keeps 100% of appendage pixels while
                        # protecting only 18% of halo pixels.
                        #
                        # The ramp MUST start high. An earlier version ramped from
                        # 0.5, but the halo's mean per-frame maximum is 0.65, so it
                        # sat mid-ramp and kept ~67% of the halo — that regressed
                        # halo suppression badly. Row-blocked to bound peak memory.
                        st=getattr(self,"_alpha_stack",None)
                        if st and self._halo_pct < 100:
                            H=af.shape[0]; step=max(64,H//16)
                            for y0 in range(0,H,step):
                                y1=min(H,y0+step)
                                blk=np.stack([a[y0:y1] for a in st]).astype(np.float32)/255.0
                                lo=np.percentile(blk,self._halo_pct,axis=0)
                                mx=blk.max(axis=0)
                                del blk
                                # opaque-somewhere gate: 0 below 0.85, 1 above 0.95
                                t=np.clip((mx-0.85)/0.10,0,1); t=t*t*(3.0-2.0*t)
                                cur=af[y0:y1]
                                af[y0:y1]=np.minimum(cur,lo)*(1.0-t)+cur*t
                        # Fusion ran on straight colour; restore premultiplied
                        # output. max(RGB) <= A now holds by construction.
                        a4=af[:,:,None]
                        rgb=np.clip(rf[:,:,:3],0,1)*a4
                        result=np.concatenate([
                            (rgb*65535).clip(0,65535).astype(np.uint16),
                            (a4*65535).clip(0,65535).astype(np.uint16)],axis=2)
                    self._alpha_stack=None
                    if self._denoise>0 or self._sharpen>0:
                        # Data is sRGB-encoded 16-bit; work in sRGB 8-bit, round-trip through 16-bit.
                        rgb=result[:,:,:3].astype(np.float32)/65535.0
                        rgb8=(rgb*255).clip(0,255).astype(np.uint8)
                        if self._denoise>0:
                            import cv2
                            rgb8=cv2.fastNlMeansDenoisingColored(rgb8,None,self._denoise/10,self._denoise/10,7,21)
                        if self._sharpen>0:
                            rgb8=_apply_sharpen(rgb8,self._sharpen)
                        result[:,:,:3]=(rgb8.astype(np.float32)/255.0*65535).clip(0,65535).astype(np.uint16)
                    # Apply stack alpha curve LUT if non-identity
                    if not np.allclose(self._lut, _IDENTITY_LUT):
                        a_f = result[:,:,3].astype(np.float32) / 65535.0
                        idx = np.clip((a_f * 255).astype(np.int32), 0, 255)
                        result[:,:,3] = (self._lut[idx] * 65535).clip(0, 65535).astype(np.uint16)
                    tifffile.imwrite(str(out_path),result,photometric='rgb',compression='deflate',
                                     extrasamples=(1,))
                    # Full path: the output sits several levels down beside matte/,
                    # so the bare filename alone is not enough to find it.
                    self._on_log(f"  Saved → {out_path}")
                    # preview — data is sRGB-encoded, pre-multiplied alpha.
                    # Un-premultiply to recover straight colour for the readout/composite.
                    rgba_f=result.astype(np.float32)/65535.0
                    a=rgba_f[:,:,3:4]
                    fg_straight=np.divide(rgba_f[:,:,:3],np.maximum(a,1e-4),
                                          where=a>1e-4,out=np.zeros_like(rgba_f[:,:,:3]))
                    fg8=(np.clip(fg_straight,0,1)*255).astype(np.uint8)
                    checker=_checkerboard(fg8.shape[0],fg8.shape[1])
                    comp=(fg8.astype(np.float32)*a+checker.astype(np.float32)*(1-a)).clip(0,255).astype(np.uint8)
                    alpha_raw=rgba_f[:,:,3]
                    a8=(alpha_raw*255).clip(0,255).astype(np.uint8)
                    self._on_prev({"composite":comp,"fg8":fg8,"alpha_raw":alpha_raw,
                                   "alpha":np.stack([a8,a8,a8],axis=2)})
                    n_ok+=1
                except Exception as e:
                    n_fail+=1
                    self._on_log(f"  ERROR: {e}"); logging.exception(e)
                finally:
                    if tmpdir:
                        import shutil as _sh
                        _sh.rmtree(tmpdir, ignore_errors=True)
            self._on_prog(n,n)
            # Report honestly: a run where every stack errored is a failure, not a
            # success. Previously this always reported ok=True.
            if n_fail and not n_ok:
                self._on_log("Stacking FAILED — no stacks produced.")
                self._on_done(False,f"all {n_fail} stack(s) failed — see log")
            elif n_fail:
                self._on_log(f"Stacking finished with errors: {n_ok} ok, {n_fail} failed.")
                self._on_done(False,f"{n_fail} of {n_ok+n_fail} stack(s) failed — see log")
            else:
                self._on_log(f"Stacking complete — {n_ok} stack(s) written.")
                self._on_done(True,"")
        except Exception as e:
            logging.exception(e)
            self._on_done(False,str(e))

class ColmapMaskPipeline:
    """Emit a COLMAP-ready image/mask pair set from the stacked mattes.

    COLMAP mask convention: the mask filename is the IMAGE filename with '.png'
    appended, mirroring the image sub-folder layout — image 'v001.png' pairs with
    mask 'v001.png.png'. Mask pixels of 0 are ignored by the feature extractor;
    any non-zero value is kept, so the mask is binarised rather than left soft.
    """
    def __init__(self, project_dir, stacked_files, on_progress=None, on_log=None,
                 on_done=None, mask_threshold=128, soft_alpha=False,
                 rgba_images=True):
        self._proj=project_dir; self._files=stacked_files
        self._thr=int(mask_threshold); self._soft=soft_alpha
        self._rgba=rgba_images
        self._on_prog=on_progress or (lambda n,t:None)
        self._on_log=on_log or (lambda m:None); self._on_done=on_done or (lambda ok,m:None)
        self._cancelled=False

    def cancel(self): self._cancelled=True

    def run(self):
        try:
            imgs_dir=self._proj/"colmap_images"; masks_dir=self._proj/"colmap_masks"
            imgs_dir.mkdir(exist_ok=True); masks_dir.mkdir(exist_ok=True)
            # Optional soft alpha for downstream 3DGS/NeRF supervision. It is a
            # SEPARATE output: COLMAP's mask semantics are binary (verified —
            # a mask value of 128 yields byte-identical keypoints to 255), so a
            # soft mask there would admit every faint halo pixel as fully valid.
            soft_dir=self._proj/"alpha_soft"
            if self._soft: soft_dir.mkdir(exist_ok=True)
            n=len(self._files); n_ok=0; n_fail=0
            for i,f in enumerate(self._files):
                if self._cancelled: self._on_done(False,"Cancelled"); return
                self._on_prog(i,n)
                try:
                    arr=tifffile.imread(str(f))
                    if arr.ndim!=3 or arr.shape[2] not in (3,4):
                        raise ValueError(f"unexpected shape {arr.shape}")
                    mx=65535.0 if arr.dtype==np.uint16 else 255.0
                    if arr.shape[2]==4:
                        rgb=arr[:,:,:3].astype(np.float32)/mx
                        a  =arr[:,:,3].astype(np.float32)/mx
                    else:
                        rgb=arr.astype(np.float32)/mx
                        a  =np.ones(arr.shape[:2],np.float32)
                    # Stacked TIFFs carry pre-multiplied alpha. Un-premultiply so the
                    # COLMAP image holds true surface colour; leaving it premultiplied
                    # darkens subject edges toward black and biases feature matching.
                    a3=a[:,:,None]
                    rgb=np.divide(rgb,np.maximum(a3,1e-4),where=a3>1e-4,
                                  out=np.zeros_like(rgb))
                    rgb8=(np.clip(rgb,0,1)*255).round().astype(np.uint8)
                    # Binary mask: COLMAP keeps any non-zero pixel, so a soft edge
                    # would admit background. Threshold to a hard subject mask.
                    mask8=np.where(a*255.0>=self._thr,255,0).astype(np.uint8)
                    stem=f.parent.parent.name
                    img_name=f"{stem}.png"
                    if self._rgba:
                        # Brush reads transparency from the IMAGE (its
                        # --alpha-loss-weight is documented as applying "if input
                        # view has transparency") and ignores any separate mask
                        # folder, so the matte has to travel in the alpha channel.
                        # Straight (un-premultiplied) RGB + alpha, which is the
                        # PNG convention. COLMAP reads these fine — verified.
                        a8=(np.clip(a,0,1)*255).round().astype(np.uint8)
                        Image.fromarray(np.dstack([rgb8,a8]),"RGBA").save(str(imgs_dir/img_name))
                    else:
                        Image.fromarray(rgb8,"RGB").save(str(imgs_dir/img_name))
                    # NOTE the doubled extension - this is the COLMAP pairing rule.
                    Image.fromarray(mask8,"L").save(str(masks_dir/f"{img_name}.png"))
                    if self._soft:
                        soft8=(np.clip(a,0,1)*255).round().astype(np.uint8)
                        Image.fromarray(soft8,"L").save(str(soft_dir/img_name))
                    cov=float((mask8>0).mean())*100.0
                    self._on_log(f"  {img_name}  (mask {cov:.1f}% subject)")
                    n_ok+=1
                except Exception as e:
                    n_fail+=1
                    self._on_log(f"  ERROR {f.name}: {e}")
            self._on_prog(n,n)
            if n_fail and not n_ok:
                self._on_done(False,f"all {n_fail} file(s) failed — see log"); return
            # macOS writes "._name" AppleDouble stubs alongside real files on
            # volumes without native xattr support (exFAT etc). COLMAP reads every
            # file in --image_path, so a stray stub makes feature extraction fail
            # on an unreadable image. Remove them so the set is genuinely usable.
            swept = 0
            for dd in (imgs_dir, masks_dir):
                try:
                    for f in dd.iterdir():
                        if f.name.startswith("._") and f.is_file():
                            try: f.unlink(); swept += 1
                            except OSError: pass
                except OSError:
                    pass
            if swept:
                self._on_log(f"  removed {swept} macOS ._ stub(s) from the COLMAP set")
            self._on_log(f"COLMAP set ready: {n_ok} image(s) → {imgs_dir}")
            self._on_log(f"                  {n_ok} mask(s)  → {masks_dir}")
            self._on_log("Run: colmap feature_extractor "
                         f"--image_path {imgs_dir} --mask_path {masks_dir}")
            if n_fail:
                self._on_done(False,f"{n_fail} of {n} file(s) failed — see log")
            else:
                self._on_done(True,"")
        except Exception as e:
            logging.exception(e)
            self._on_done(False,str(e))

class BatchPipeline:
    """Matte -> stack -> COLMAP for every rotation, ONE rotation at a time.

    Mattes are the disk problem: 200 rotations x 50 frames x ~28MB is ~280GB if
    they are all generated up front. Doing a full matte/stack/export cycle per
    rotation and discarding that rotation's mattes keeps peak usage at roughly
    one rotation's worth.

    Resumable: a rotation whose stacked TIFF and COLMAP pair already exist is
    skipped, so an interrupted run can simply be restarted.
    """
    def __init__(self, pairs, stacks, proj_dir, matte_kw, stack_kw,
                 delete_mattes=True, mask_threshold=128, soft_alpha=False,
                 rgba_images=True,
                 pipelined=True,
                 reprocess=True, clean_colmap=True,
                 on_progress=None, on_log=None, on_done=None, on_preview=None,
                 on_stage=None, on_stats=None):
        self._pairs=pairs; self._stacks=stacks; self._proj=Path(proj_dir)
        self._mkw=dict(matte_kw); self._skw=dict(stack_kw)
        self._delete=delete_mattes; self._thr=mask_threshold
        self._soft_alpha=soft_alpha
        self._rgba_images=rgba_images
        self._pipelined=pipelined; self._reprocess=reprocess
        self._clean_colmap=clean_colmap
        self._on_prog=on_progress or (lambda n,t:None)
        self._on_log=on_log or (lambda m:None)
        self._on_done=on_done or (lambda ok,m:None)
        self._on_prev=on_preview or (lambda imgs:None)
        self._on_stage=on_stage or (lambda st,lab,f:None)
        self._on_stats=on_stats or (lambda d:None)
        self._t0=None; self._freed=0
        self._c={'done':0,'skipped':0,'failed':0,'total':0}
        self._last_emit=0.0
        self._cancelled=False

    def cancel(self): self._cancelled=True

    def _sub(self, i, n, lo, hi, stage=None, label=""):
        """Map a sub-stage's 0..1 progress into this rotation's slice of the bar,
        and drive that stage's own lane in the batch dashboard."""
        def cb(a, b):
            f = (a / b) if b else 0.0
            self._on_prog(i + lo + (hi - lo) * f, n)
            if stage: self._on_stage(stage, label, f)
            self._tick()
        return cb

    def _stats(self, done=None, skipped=None, failed=None, n=None):
        """Push the counters to the dashboard.

        Also called (throttled) from the stage callbacks: a rotation can take
        many minutes, and without this the whole panel — overall, elapsed,
        counts — sits frozen until the first rotation finishes, which reads as
        "nothing happened".
        """
        c=self._c
        if done is not None: c['done']=done
        if skipped is not None: c['skipped']=skipped
        if failed is not None: c['failed']=failed
        if n is not None: c['total']=n
        el = 0.0 if self._t0 is None else (time.time() - self._t0)
        finished = c['done'] + c['skipped'] + c['failed']
        tot = c['total']
        eta = (el / finished) * (tot - finished) if finished and tot > finished else None
        self._last_emit = time.time()
        self._on_stats(dict(done=c['done'], skipped=c['skipped'], failed=c['failed'],
                            elapsed=el, eta=eta, freed=self._freed,
                            finished=finished, total=tot))

    def _tick(self):
        """Refresh elapsed/overall at most twice a second."""
        if time.time() - self._last_emit >= 0.5:
            self._stats()

    # ── stages ────────────────────────────────────────────────────────
    def _plan(self, i, group):
        """Resolve the paths for one rotation, or None if it has no pairs."""
        gp=[self._pairs[k] for k in group]
        if not gp: return None
        mdir=gp[0].out_path.parent
        rot=mdir.parent
        return dict(i=i, pairs=gp, mdir=mdir, rot=rot,
                    out=rot/"stacked"/f"{rot.name}_stacked.tif",
                    img=self._proj/"colmap_images"/f"{rot.name}.png")

    @staticmethod
    def _purge(paths):
        n=0
        for p in paths:
            try:
                if p.is_file(): p.unlink(); n+=1
            except OSError: pass
        return n

    def _clear_stale(self, job):
        """Remove any prior output for this rotation before regenerating it.

        Leftovers are not merely untidy. StackPipeline globs the whole matte
        directory, so mattes from an earlier test run — a different frame count,
        different settings, possibly a partial write — would be fused in
        alongside the new ones. The matte directory must contain exactly this
        run's frames and nothing else.
        """
        mdir=job["mdir"]
        n=self._purge(list(mdir.glob('*.tif*'))) if mdir.is_dir() else 0
        if n: self._on_log(f"   cleared {n} stale matte(s)")
        return n

    def _clear_outputs(self, job):
        """Drop this rotation's previous stacked TIFF and COLMAP pair."""
        gone=self._purge([job["out"], job["img"],
                          self._proj/"colmap_masks"/f"{job['img'].name}.png"])
        # AppleDouble stubs beside them on external volumes
        for d,stem in ((job["out"].parent,job["out"].name),
                       (job["img"].parent,job["img"].name)):
            gone+=self._purge([d/f"._{stem}"])
        return gone

    def _matte_one(self, job, prog):
        """Produce this rotation's mattes. Returns the files it created."""
        mdir=job["mdir"]; mdir.mkdir(parents=True,exist_ok=True)
        self._clear_stale(job)
        MattePipeline(pairs=job["pairs"], on_progress=prog,
            on_log=lambda m: self._on_log("   "+m.strip()),
            on_done=lambda ok,msg: None, on_preview=lambda *a: None,
            **self._mkw).run()
        created=[p for p in mdir.glob('*.tif*') if not p.name.startswith('.')]
        if not created:
            raise RuntimeError("no mattes produced")
        return created

    def _stack_and_export(self, job, created, prog_stack, prog_colmap):
        """Stack, write the COLMAP pair, then reclaim the mattes."""
        StackPipeline(matte_dirs=[job["mdir"]], on_progress=prog_stack,
            on_log=lambda m: self._on_log("   "+m.strip()),
            on_done=lambda ok,msg: None,
            on_preview=lambda imgs: self._on_prev(imgs),
            **self._skw).run()
        if not job["out"].exists():
            raise RuntimeError("stack produced no output")
        ColmapMaskPipeline(project_dir=self._proj, stacked_files=[job["out"]],
            mask_threshold=self._thr, soft_alpha=self._soft_alpha,
            rgba_images=self._rgba_images,
            on_progress=prog_colmap,
            on_log=lambda m: self._on_log("   "+m.strip()),
            on_done=lambda ok,msg: None).run()
        if not job["img"].exists():
            raise RuntimeError("COLMAP export produced no image")
        # Only now is it safe to drop the inputs.
        if self._delete:
            freed=0
            for p in (created or list(job["mdir"].glob('*.tif*'))):
                try: freed+=p.stat().st_size; p.unlink()
                except OSError: pass
            try: job["mdir"].rmdir()
            except OSError: pass
            self._freed += freed
            self._on_log(f"   freed {freed/1e9:.2f} GB of mattes")

    def _empty_colmap_dirs(self):
        """Clear colmap_images/ and colmap_masks/ so the set holds only this run.

        Guarded to overwrite mode by the caller: emptying these while RESUMING
        would delete the very outputs the resume is built on, and would also
        discard views contributed by any other orbit.

        Files only, non-recursive — a nested layout is never touched.
        """
        n=0; freed=0
        for d in (self._proj/"colmap_images", self._proj/"colmap_masks",
                  self._proj/"alpha_soft"):
            if not d.is_dir(): continue
            for f in d.iterdir():
                try:
                    if f.is_file():
                        freed+=f.stat().st_size; f.unlink(); n+=1
                except OSError: pass
        if n:
            self._on_log(f"Emptied COLMAP dirs: removed {n} file(s), "
                         f"{freed/1e6:.1f} MB")
        return n

    def _jobs(self):
        """Rotations still needing work, in order."""
        out=[]
        for i,group in enumerate(self._stacks):
            j=self._plan(i,group)
            if j is None: continue
            out.append(j)
        return out

    # ── drivers ───────────────────────────────────────────────────────
    def run(self):
        try:
            jobs=self._jobs(); n=len(jobs)
            if not n:
                self._on_done(False,"No rotations found"); return
            self._t0=time.time()
            self._c={'done':0,'skipped':0,'failed':0,'total':len(self._jobs())}
            self._stats()
            # Once, up front — and only when overwriting. During a resume these
            # directories hold the completed work we are resuming from.
            if self._clean_colmap and self._reprocess:
                self._empty_colmap_dirs()
            mode=("pipelined (matting runs ahead of stacking)"
                  if self._pipelined else "one at a time")
            self._on_log(f"Batch: {n} rotation(s), {mode}"
                         f"{', deleting mattes after each' if self._delete else ''}.")
            r=self._run_pipelined(jobs,n) if self._pipelined else self._run_serial(jobs,n)
            done,skipped,failed=r
            self._on_prog(n,n)
            if self._cancelled:
                self._on_log("Cancelled."); self._on_done(False,"Cancelled"); return
            msg=f"{done} completed, {skipped} skipped, {failed} failed"
            self._on_log(f"Batch finished — {msg}.")
            self._on_log(f"COLMAP set: {self._proj/'colmap_images'} + "
                         f"{self._proj/'colmap_masks'}")
            self._on_done(failed==0, "" if failed==0 else msg)
        except Exception as e:
            logging.exception(e)
            self._on_done(False,str(e))

    def _run_serial(self, jobs, n):
        done=skipped=failed=0
        for k,job in enumerate(jobs):
            if self._cancelled: break
            self._on_prog(k,n)
            if (not self._reprocess) and job["out"].exists() and job["img"].exists():
                skipped+=1
                self._on_log(f"[{k+1}/{n}] {job['rot'].name} — already complete, skipping")
                self._stats(done,skipped,failed,n)
                continue
            self._on_log(f"[{k+1}/{n}] {job['rot'].name} — {len(job['pairs'])} frame(s)")
            try:
                created=[]
                rn=job["rot"].name
                if self._reprocess:
                    g=self._clear_outputs(job)
                    if g: self._on_log(f"   replaced {g} previous output file(s)")
                if not job["out"].exists():
                    created=self._matte_one(job,self._sub(k,n,0.00,0.55,"matte",rn))
                    if self._cancelled: break
                    self._on_stage("matte","",0.0)
                self._stack_and_export(job,created,
                                       self._sub(k,n,0.55,0.90,"stack",rn),
                                       self._sub(k,n,0.90,0.98,"colmap",rn))
                self._on_stage("stack","",0.0); self._on_stage("colmap","",0.0)
                done+=1
            except Exception as e:
                failed+=1; self._on_log(f"   ERROR: {e}"); logging.exception(e)
            self._stats(done,skipped,failed,n)
        return done,skipped,failed

    def _run_pipelined(self, jobs, n):
        """Matte rotation i+1 while rotation i stacks.

        Bounded queue of 1 means at most two rotations of mattes exist at once —
        the one being written and the one waiting to stack. Stacking stays
        single-file on purpose: it is already internally threaded and CPU-bound,
        so running two at once would only contend.
        """
        import queue as _q
        q=_q.Queue(maxsize=1)
        state={"done":0,"skipped":0,"failed":0}
        lock=threading.Lock()

        def produce():
            try:
                for k,job in enumerate(jobs):
                    if self._cancelled: break
                    if (not self._reprocess) and job["out"].exists() and job["img"].exists():
                        with lock: state["skipped"]+=1
                        self._on_log(f"[{k+1}/{n}] {job['rot'].name} — already complete, skipping")
                        self._stats(state["done"],state["skipped"],state["failed"],n)
                        continue
                    self._on_log(f"[{k+1}/{n}] {job['rot'].name} — matting "
                                 f"{len(job['pairs'])} frame(s)")
                    try:
                        if self._reprocess:
                            g=self._clear_outputs(job)
                            if g: self._on_log(f"   replaced {g} previous output file(s)")
                        created=[] if job["out"].exists() else self._matte_one(
                            job, self._sub(k,n,0.00,0.45,"matte",job["rot"].name))
                        self._on_stage("matte","",0.0)
                        while not self._cancelled:
                            try: q.put((k,job,created),timeout=0.5); break
                            except _q.Full: continue
                    except Exception as e:
                        with lock: state["failed"]+=1
                        self._on_log(f"   ERROR (matte {job['rot'].name}): {e}")
                        logging.exception(e)
                        self._stats(state["done"],state["skipped"],state["failed"],n)
            finally:
                while not self._cancelled:
                    try: q.put(None,timeout=0.5); break
                    except _q.Full: continue

        t=threading.Thread(target=produce,daemon=True); t.start()
        while True:
            if self._cancelled: break
            try: item=q.get(timeout=0.5)
            except Exception: continue
            if item is None: break
            k,job,created=item
            self._on_log(f"   stacking {job['rot'].name}")
            try:
                rn=job["rot"].name
                self._stack_and_export(job,created,
                                       self._sub(k,n,0.45,0.90,"stack",rn),
                                       self._sub(k,n,0.90,0.98,"colmap",rn))
                self._on_stage("stack","",0.0); self._on_stage("colmap","",0.0)
                with lock: state["done"]+=1
            except Exception as e:
                with lock: state["failed"]+=1
                self._on_log(f"   ERROR: {e}"); logging.exception(e)
            self._on_prog(min(n,k+1),n)
            self._stats(state["done"],state["skipped"],state["failed"],n)
        t.join(timeout=5)
        return state["done"],state["skipped"],state["failed"]


# ── CSS (Smalti design system + MattePro additions) ───────────────────────────

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0C0D10;--surface:#13151A;--card:#1A1D25;--card2:#1F2330;
  --border:#2A2E3D;--accent:#7C6EFA;--accent2:#FA6E9E;--green:#4ADE80;
  --text:#E8EAF0;--sub:#717898;--rad:14px;--rad-sm:8px;
}
html,body{background:var(--bg)!important;color:var(--text)!important;
  font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.5;
  height:100%;margin:0;overflow:hidden}
/* Strip every Quasar container that could cap width */
.q-layout,.q-layout__section--marginal,
.q-page-container,.q-page,
.nicegui-content{
  padding:0!important;margin:0!important;
  height:100%!important;min-height:0!important;
  width:100%!important;max-width:none!important;
  box-sizing:border-box}

/* ── Layout ─────────────────────────────────────────────────────── */
.mp-shell{display:grid;grid-template-columns:420px 1fr;grid-template-rows:56px 1fr;
  grid-template-areas:"hdr hdr""left right";
  height:100vh;width:100vw;overflow:hidden;position:fixed;top:0;left:0}
.mp-header{grid-area:hdr;background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 24px;gap:14px;flex-shrink:0}
.mp-left{grid-area:left;background:var(--bg);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;min-width:0}
.mp-right{grid-area:right;background:var(--bg);overflow:hidden;
  display:flex;flex-direction:column;min-width:0}

/* ── Tab bar ─────────────────────────────────────────────────────── */
.mp-tabs{display:flex;background:var(--surface);border-bottom:1px solid var(--border);
  padding:8px 12px;gap:6px;flex-shrink:0}
button.mp-tab{
  padding:7px 22px!important;border-radius:var(--rad-sm)!important;
  background:var(--card2)!important;color:var(--sub)!important;
  font-size:11px!important;font-weight:700!important;
  letter-spacing:0.08em!important;text-transform:uppercase!important;
  border:1px solid var(--border)!important}
button.mp-tab:hover{background:#272b3a!important;color:var(--text)!important}
button.mp-tab.active{background:var(--accent)!important;color:#fff!important;border-color:var(--accent)!important}

/* ── Scroll panels ───────────────────────────────────────────────── */
.mp-scroll{flex:1;overflow-y:auto;overflow-x:hidden}
.mp-scroll::-webkit-scrollbar{width:0}
.mp-panel{padding:16px 16px 32px;display:flex;flex-direction:column;gap:10px}

/* ── Cards ───────────────────────────────────────────────────────── */
.card{background:var(--card);border:1px solid var(--border);
  border-radius:var(--rad);padding:16px 18px}
.card-title{font-size:10px;font-weight:700;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--sub);margin-bottom:14px;display:block}

/* ── Param rows ──────────────────────────────────────────────────── */
/* Stacked param row: caption line (label left, value right) above a slider that
   spans the full panel width. Two benefits over the old side-by-side layout:
   the track is ~2.5x longer so fine adjustment is far easier, and the track
   width no longer depends on the value text. Previously "0 EV" -> "+0.01 EV"
   widened the value column, shrank the track, and made the thumb jump — read
   as snapping, worst at the centre where the zero form flips to signed.
   (The old width:56px never applied anyway: .param-value is an inline <span>,
   and width is ignored on non-replaced inline elements.) */
.param-row{display:block;margin-bottom:14px}
.param-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:8px;margin-bottom:2px}
/* ui.html() wraps each span in a div; size those wrappers, not just the spans. */
.param-head>div:first-child{flex:1 1 auto;min-width:0}
.param-head>div:last-child{flex:0 0 auto}
.param-label{font-size:12px;color:var(--text);display:block;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.param-value{font-size:12px;color:var(--accent);display:block;text-align:right;
  font-variant-numeric:tabular-nums;font-weight:500;white-space:nowrap}

/* ── Batch progress dashboard ─────────────────────────────────────── */
.bstat-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}
.bstat-k{font-size:12px;color:var(--text)}
.bstat-v{font-size:11px;color:var(--sub);font-variant-numeric:tabular-nums;
  text-align:right;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bbar{height:6px;border-radius:3px;background:var(--card2);overflow:hidden}
.bbar-fill{height:100%;width:0%;border-radius:3px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width .18s linear}
.bbar-fill.alt{background:var(--accent);opacity:.75}
.bgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px 8px;margin-top:18px;
  padding-top:16px;border-top:1px solid var(--border)}
.bnum{font-size:17px;font-weight:600;color:var(--text);
  font-variant-numeric:tabular-nums;line-height:1.1}
.bnum.ok{color:var(--green)}
.bnum.err{color:var(--accent2)}
.blab{font-size:10px;color:var(--sub);letter-spacing:.08em;text-transform:uppercase;
  margin-top:3px}

/* ── Quasar slider overrides ─────────────────────────────────────── */
/* Full panel width inside a stacked .param-row. */
.param-row .q-slider{display:block!important;width:100%!important}
.q-slider{flex:1!important;min-width:0!important}
.q-slider .q-slider__track-container{opacity:1!important}
.q-slider .q-slider__track{background:var(--border)!important}
.q-slider .q-slider__selection,.q-slider .q-slider__inner{background:var(--accent)!important}
.q-slider .q-slider__thumb{color:var(--accent)!important}
.q-slider .q-slider__thumb-container .q-slider__thumb{color:var(--accent)!important}
.q-slider__focus-ring{opacity:0!important}

/* ── Buttons — reset then restyle so browser defaults can't win ──── */
button.mp-btn-primary,button.mp-btn-secondary,button.mp-btn-sm,button.mp-tab{
  all:unset;box-sizing:border-box;cursor:pointer;display:inline-block;
  font-family:'Inter',system-ui,sans-serif;
  transition:opacity 0.15s,background 0.15s,color 0.15s}
button.mp-btn-primary{
  display:block;width:100%;padding:13px 0;border-radius:var(--rad-sm);
  background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
  font-weight:700;font-size:14px;letter-spacing:0.04em;text-align:center}
button.mp-btn-primary:hover{opacity:0.85}
button.mp-btn-primary:disabled{opacity:0.35;cursor:not-allowed;pointer-events:none}
button.mp-btn-secondary{
  display:block;width:100%;padding:10px 0;border-radius:var(--rad-sm);
  background:var(--card2);color:var(--text);
  font-weight:600;font-size:13px;letter-spacing:0.03em;text-align:center;
  border:1px solid var(--border)}
button.mp-btn-secondary:hover{background:#272b3a}
button.mp-btn-secondary:disabled{opacity:0.35;cursor:not-allowed;pointer-events:none}
button.mp-btn-sm{
  display:inline-block;padding:5px 14px;border-radius:var(--rad-sm);
  white-space:nowrap;background:var(--card2);color:var(--sub);
  font-size:11px;font-weight:600;flex-shrink:0;border:1px solid var(--border)}
button.mp-btn-sm:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn-row{display:flex;gap:8px}
.btn-row button.mp-btn-secondary{flex:1}

/* ── File rows ───────────────────────────────────────────────────── */
.file-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.file-label{font-size:12px;color:var(--sub);width:72px;min-width:72px;flex-shrink:0}
.file-input{flex:1;min-width:0;background:var(--card2);border:1px solid var(--border);
  border-radius:var(--rad-sm);padding:7px 10px;color:var(--text);font-size:12px;
  outline:none;transition:border-color 0.15s;font-family:'Inter',system-ui,sans-serif}
.file-input:focus{border-color:var(--accent)}
.file-input::placeholder{color:var(--sub);opacity:0.6}

/* ── Progress ────────────────────────────────────────────────────── */
.prog-track{width:100%;height:4px;border-radius:2px;background:var(--border);
  overflow:hidden;margin:10px 0 6px}
.prog-fill{height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width 0.3s ease;width:0}

/* ── Log ─────────────────────────────────────────────────────────── */
.mp-log-box{background:var(--card2);border:1px solid var(--border);
  border-radius:var(--rad-sm);font-family:'Menlo','Monaco',monospace;
  font-size:11px;color:var(--sub);padding:10px;overflow-y:auto;
  height:200px;white-space:pre-wrap;line-height:1.65}

/* ── Preview ─────────────────────────────────────────────────────── */
.preview-bar{display:flex;align-items:center;gap:10px;padding:10px 18px;
  background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}
/* Alpha curve: full panel width, fixed aspect-ish height. The backing store is
   sized to devicePixelRatio in JS so strokes stay crisp on Retina. */
.curve-canvas{display:block;width:100%;height:200px;border-radius:10px;
  cursor:crosshair;background:#161922;border:1px solid var(--border)}
.curve-hint{font-size:11px;color:var(--sub);letter-spacing:.01em}
.curve-hint b{color:#8E86FF;font-weight:600}
.preview-area{flex:1;overflow:hidden;background:#0a0b0e;position:relative;
  touch-action:none;cursor:grab}
.preview-area.mp-panning{cursor:grabbing}
/* Transform-driven zoom/pan: GPU compositing, no reflow, no server round-trip. */
#preview-img-el{position:absolute;top:0;left:0;transform-origin:0 0;
  max-width:none;max-height:none;will-change:transform;
  image-rendering:auto;user-select:none;-webkit-user-drag:none}
#preview-img-el.mp-snap{transition:transform 160ms cubic-bezier(.22,.61,.36,1)}
/* Crisp pixels once magnified past 1:1 so pixel-peeping is honest. */
#preview-img-el.mp-crisp{image-rendering:pixelated}
.seg-ctrl{display:inline-flex;background:var(--card2);border:1px solid var(--border);
  border-radius:var(--rad-sm);overflow:hidden}
button.seg-btn{
  all:unset;box-sizing:border-box;cursor:pointer;
  padding:5px 13px;font-size:11px;font-weight:700;
  color:var(--sub);letter-spacing:0.05em;
  font-family:'Inter',system-ui,sans-serif;
  transition:background 0.15s,color 0.15s}
button.seg-btn:hover{color:var(--text)}
button.seg-btn.active{background:var(--accent);color:#fff}

/* ── Status ──────────────────────────────────────────────────────── */
.status-ok{font-size:11px;color:var(--green)}
.status-muted{font-size:11px;color:var(--sub)}
.status-err{font-size:11px;color:#FF5F6D}
</style>
"""

# ── Curve editor (HTML5 canvas, same logic as tkinter version) ─────────────────

CURVE_JS = """
<script>
(function(){
  function initCurve(canvasId, callbackName, yMin, yMax) {
    var cv = document.getElementById(canvasId);
    if (!cv) return;
    // Idempotent: a second init on the same canvas would create a rival
    // instance with its own pts and listeners, and would clobber the shared
    // _reset/_getlut globals — desyncing the drawn curve from the applied LUT.
    if (cv._mpCurveInit) return;
    cv._mpCurveInit = true;
    var ctx = cv.getContext('2d');
    // W/H are CSS pixels; the backing store is scaled by devicePixelRatio so
    // strokes are rendered at native resolution instead of being upscaled.
    var W = 0, H = 0, dpr = 1;
    var PAD = 10;              // inset so end handles aren't clipped at the edge
    var pts = [[0,0],[1,1]];
    var dragging = -1;

    function resize(){
      var r = cv.getBoundingClientRect();
      var cw = Math.max(1, Math.round(r.width));
      var ch = Math.max(1, Math.round(r.height));
      dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(cw*dpr) || cv.height !== Math.round(ch*dpr)) {
        cv.width  = Math.round(cw*dpr);
        cv.height = Math.round(ch*dpr);
      }
      W = cw; H = ch;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    // Plot area is inset by PAD so the 0 and 1 handles sit fully inside the canvas.
    function plotW(){ return Math.max(1, W - PAD*2); }
    function plotH(){ return Math.max(1, H - PAD*2); }
    function toC(px, py) {
      return [PAD + px*plotW(),
              PAD + plotH() - (py - yMin)/(yMax - yMin)*plotH()];
    }
    function toP(cx, cy) {
      return [(cx - PAD)/plotW(),
              yMin + (PAD + plotH() - cy)/plotH()*(yMax - yMin)];
    }

    function nearest(cx, cy) {
      var best=-1, bd=1e9;
      for(var i=0;i<pts.length;i++){
        var c=toC(pts[i][0],pts[i][1]);
        var d=Math.hypot(cx-c[0],cy-c[1]);
        if(d<12&&d<bd){bd=d;best=i;}
      }
      return best;
    }

    function hermiteLut() {
      var sorted=pts.slice().sort((a,b)=>a[0]-b[0]);
      var n=sorted.length;
      var xs=sorted.map(p=>p[0]),ys=sorted.map(p=>p[1]);
      var d=[];
      for(var i=0;i<n-1;i++) d.push((ys[i+1]-ys[i])/(xs[i+1]-xs[i]));
      var m=new Array(n).fill(0);
      // d has n-1 slopes (0..n-2). The last one is d[n-2], NOT d[n-1] — that
      // index is out of bounds and yields undefined, which turns every interior
      // LUT sample into NaN. Canvas silently drops NaN path segments, so the
      // curve simply vanished (worst at the 2-point identity default, where the
      // very first tangent lookup fails). Python's _monotone_cubic_lut uses
      // d[-1], which is the last element there — the literal JS translation of
      // that idiom is the bug.
      m[0]=d[0]; m[n-1]=d[n-2];
      for(var i=1;i<n-1;i++){
        if(d[i-1]*d[i]<=0) m[i]=0;
        else m[i]=(d[i-1]+d[i])/2;
      }
      for(var i=0;i<n-1;i++){
        if(Math.abs(d[i])<1e-9){m[i]=0;m[i+1]=0;}
        else{var a=m[i]/d[i],b=m[i+1]/d[i];
          var s2=a*a+b*b;if(s2>9){var s=3/Math.sqrt(s2);m[i]=s*a*d[i];m[i+1]=s*b*d[i];}}
      }
      var lut=new Array(256);
      for(var j=0;j<256;j++){
        var tv=j/255;
        if(tv<=xs[0]){lut[j]=ys[0];continue;}
        if(tv>=xs[n-1]){lut[j]=ys[n-1];continue;}
        for(var i=0;i<n-1;i++){
          if(xs[i]<=tv&&tv<=xs[i+1]){
            var h=(tv-xs[i])/(xs[i+1]-xs[i]);
            // Cubic Hermite basis. h00 is h*h*(2h-3)+1 == 2h^3-3h^2+1; it was
            // h*h*h*(2h-3)+1, a quartic with one factor of h too many, which
            // broke partition of unity (h00+h01 peaked at 1.26) and bulged the
            // curve up to 26% high between control points.
            var dx=xs[i+1]-xs[i];
            var h00=h*h*(2*h-3)+1, h10=h*h*h-2*h*h+h;
            var h01=h*h*(3-2*h),   h11=h*h*h-h*h;
            lut[j]=h00*ys[i]+h10*dx*m[i]+h01*ys[i+1]+h11*dx*m[i+1];
            break;
          }
        }
      }
      return lut;
    }

    // Snap a coordinate to a half-pixel so 1px strokes land on a pixel centre
    // instead of straddling two — otherwise every grid line renders as 2px grey.
    function crisp(v){ return Math.round(v) + 0.5; }

    function draw() {
      if (!W || !H) return;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle='#161922'; ctx.fillRect(0,0,W,H);

      var x0=toC(0,0)[0], x1=toC(1,1)[0];
      var yTop=PAD, yBot=PAD+plotH();

      // Grid — quarters across the plot area.
      ctx.lineWidth=1; ctx.strokeStyle='#242838';
      ctx.beginPath();
      for(var i=0;i<=4;i++){
        var gx=crisp(x0+(x1-x0)*i/4), gy=crisp(yTop+plotH()*i/4);
        ctx.moveTo(gx,yTop); ctx.lineTo(gx,yBot);
        ctx.moveTo(x0,gy);   ctx.lineTo(x1,gy);
      }
      ctx.stroke();

      // Identity reference.
      ctx.strokeStyle='#333850'; ctx.setLineDash([3,4]); ctx.lineWidth=1;
      var c0=toC(0,0),c1=toC(1,1);
      ctx.beginPath();ctx.moveTo(c0[0],c0[1]);ctx.lineTo(c1[0],c1[1]);ctx.stroke();
      ctx.setLineDash([]);

      // Curve. Sample per device pixel so the polyline is smooth at any width,
      // rather than 256 fixed samples stretched across a wide canvas.
      var lut=hermiteLut();
      var n=Math.max(64, Math.ceil(plotW()*dpr));
      // Build the polyline, dropping any non-finite sample. A NaN would be
      // silently skipped by canvas, leaving the fill unbounded while the stroke
      // disappears — the fill and stroke must run over the identical points.
      var xy=[];
      for(var j=0;j<=n;j++){
        var t=j/n;
        var f=t*255, i0=Math.floor(f), i1=Math.min(255,i0+1), fr=f-i0;
        var a0=lut[i0], a1=lut[i1];
        if(!isFinite(a0)||!isFinite(a1)) continue;
        var v=a0*(1-fr)+a1*fr;                // interpolate, no stair-stepping
        var cy=PAD+plotH()-(v-yMin)/(yMax-yMin)*plotH();
        if(!isFinite(cy)) continue;
        xy.push([x0+(x1-x0)*t, cy]);
      }
      if(xy.length>1){
        var last=xy.length-1;
        // Soft fill under the curve — bounded by exactly the stroked points.
        var grad=ctx.createLinearGradient(0,yTop,0,yBot);
        grad.addColorStop(0,'rgba(124,110,250,0.22)');
        grad.addColorStop(1,'rgba(124,110,250,0.02)');
        ctx.beginPath();
        ctx.moveTo(xy[0][0],yBot);
        for(var j=0;j<=last;j++) ctx.lineTo(xy[j][0],xy[j][1]);
        ctx.lineTo(xy[last][0],yBot);
        ctx.closePath();
        ctx.fillStyle=grad; ctx.fill();

        // Stroke: round joins/caps and a faint glow keep the line clean.
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(xy[0][0],xy[0][1]);
        for(var j=1;j<=last;j++) ctx.lineTo(xy[j][0],xy[j][1]);
        ctx.lineJoin='round'; ctx.lineCap='round';
        ctx.shadowColor='rgba(124,110,250,0.45)'; ctx.shadowBlur=6;
        ctx.strokeStyle='#7C6EFA'; ctx.lineWidth=2;
        ctx.stroke();
        ctx.restore();
      }

      // Control points.
      var sorted=pts.slice().sort(function(a,b){return a[0]-b[0];});
      for(var i=0;i<sorted.length;i++){
        var c=toC(sorted[i][0],sorted[i][1]);
        ctx.beginPath();ctx.arc(c[0],c[1],5.5,0,Math.PI*2);
        ctx.fillStyle='#0F1118'; ctx.fill();
        ctx.lineWidth=2.5; ctx.strokeStyle='#8E86FF'; ctx.stroke();
        ctx.beginPath();ctx.arc(c[0],c[1],2,0,Math.PI*2);
        ctx.fillStyle='#EDEBFF'; ctx.fill();
      }
    }

    function fire() {
      // Send raw control points to Python instead of the computed float LUT —
      // JavaScript sparse arrays produce undefined holes that become NaN when
      // numpy deserializes the JSON. Python's _monotone_cubic_lut is authoritative.
      emitEvent(callbackName, {pts: pts.map(function(p){return [p[0],p[1]];})});
    }

    cv.addEventListener('mousedown', function(e){
      var r=cv.getBoundingClientRect();
      var cx=e.clientX-r.left, cy=e.clientY-r.top;
      if(e.button===2){
        var n=nearest(cx,cy);
        if(n>0&&n<pts.length-1){pts.splice(n,1);draw();fire();}
        return;
      }
      var n=nearest(cx,cy);
      if(n>=0){dragging=n;}
      else {
        var p=toP(cx,cy);
        if(p[0]>0.02&&p[0]<0.98){pts.push(p);pts.sort((a,b)=>a[0]-b[0]);
          dragging=pts.findIndex(pp=>Math.abs(pp[0]-p[0])<1e-6&&Math.abs(pp[1]-p[1])<1e-6);}
      }
      draw();
    });
    cv.addEventListener('mousemove', function(e){
      if(dragging<0) return;
      var r=cv.getBoundingClientRect();
      var cx=e.clientX-r.left, cy=e.clientY-r.top;
      var p=toP(cx,cy);
      if(dragging===0){p[0]=0;}
      else if(dragging===pts.length-1){p[0]=1;}
      else{p[0]=Math.max(0.01,Math.min(0.99,p[0]));}
      p[1]=Math.max(yMin,Math.min(yMax,p[1]));
      pts[dragging]=p; draw();
    });
    cv.addEventListener('mouseup', function(e){
      if(dragging>=0){dragging=-1;fire();}
    });
    cv.addEventListener('contextmenu',function(e){e.preventDefault();});
    // Track panel width changes (and DPR changes on monitor switch).
    if (window.ResizeObserver) {
      var ro = new ResizeObserver(function(){ resize(); });
      ro.observe(cv);
    }
    window.addEventListener('resize', resize);
    // First layout can report width 0 if the card is still being laid out.
    resize();
    if (!W || W < 2) setTimeout(resize, 60);
    window[callbackName+'_reset']=function(){pts=[[0,0],[1,1]];draw();fire();};
    window[callbackName+'_getlut']=function(){return hermiteLut();};
  }

  window.initCurve=initCurve;
  // NiceGUI renders the body after DOMContentLoaded, so poll briefly until the
  // canvases exist. initCurve is idempotent, so repeats are harmless.
  function _mpInitCurves(tries){
    var a=document.getElementById('curve-matte');
    var b=document.getElementById('curve-stack');
    if(a) initCurve('curve-matte','curve_matte_changed',-0.5,1.5);
    if(b) initCurve('curve-stack','curve_stack_changed',-0.5,1.5);
    if((!a||!b) && (tries||0)<40) setTimeout(function(){_mpInitCurves((tries||0)+1);},150);
  }
  document.addEventListener('DOMContentLoaded', function(){ _mpInitCurves(0); });
  _mpInitCurves(0);
})();
</script>
"""

# ── App state (no tkinter) ────────────────────────────────────────────────────

class MatteApp:
    def __init__(self):
        # Project state
        self.proj_path: str = ""
        self.pairs: list[MattePair] = []
        self.matte_dirs: list[Path] = []
        self.stacked_files: list[Path] = []
        # Last stacked-result preview, kept so the post-stack project rescan
        # cannot replace it with a matte frame.
        self._stack_prev_imgs = None
        self._keep_stack_preview = False
        self.cal_black_path: str = ""
        self.cal_white_path: str = ""
        self.cal_grey_path: str = ""
        self.cal: Optional[CalImages] = None
        self.cal_loading: bool = False

        # Matting params
        self.kelvin: int = 5500
        self.alpha_min: float = 0.02
        self.grey_brightness: int = 50
        self.sharpen: int = 0
        self.solidify: int = 0
        self.ca_red: int = 0
        self.ca_blue: int = 0
        self.defringe: int = 0
        self.exposure: int = 0  # EV × 100 (−200..+200 = −2..+2 stops)

        # Alpha curve LUTs
        self.alpha_lut: np.ndarray = _IDENTITY_LUT.copy()
        self.stack_alpha_lut: np.ndarray = _IDENTITY_LUT.copy()

        # Stack params
        self.stack_denoise: int = 0
        self.stack_sharpen: int = 0
        self.stack_sharpen_r: float = 1.0
        self.stack_kernel: int = 5
        self.stack_genk: float = 0.4
        self.stack_minsize: int = 32
        # Fuse straight (un-premultiplied) colour then re-premultiply. Costs an
        # extra pass over the frames but removes the edge halo entirely.
        # Percentile ceiling applied to fused alpha (100 = off).
        self.stack_halo_pct: int = 75
        # Register frames before fusion (focus breathing shifts/scales them).
        self.stack_align: bool = True
        # Batch: reclaim each rotation's mattes once it is fully exported.
        self.batch_delete_mattes: bool = True
        # Matte the next rotation while the current one stacks.
        self.batch_pipelined: bool = True
        # Replace prior outputs (test runs) rather than skipping them.
        self.batch_reprocess: bool = True
        # Start the COLMAP dirs empty so the set holds only this batch.
        self.batch_clean_colmap: bool = True
        # Binary cut for the COLMAP mask (0..255).
        self.mask_threshold: int = 128
        # Also emit soft greyscale alpha for 3DGS supervision.
        self.batch_soft_alpha: bool = False
        # Embed the matte in the image alpha — required by Brush.
        self.rgba_images: bool = True
        self._batch_pipe = None

        # Processing
        self._pipeline: Optional[MattePipeline] = None
        self._stack_pipe: Optional[StackPipeline] = None
        self._colmap_pipe: Optional[ColmapMaskPipeline] = None
        self._q: queue.Queue = queue.Queue()

        # Preview state
        self.pair_idx: int = 0
        self._stacks: list = []    # list of [pair_indices] per stack/rotation
        self.stack_idx: int = 0
        self.frame_idx: int = 0
        self.view_mode: str = "composite"
        self._comp_bg: str = "checker"
        self._zoom: str = "fit"
        self._raw_cache: OrderedDict = OrderedDict()
        self._loading_idx: Optional[int] = None
        self._prev_imgs: dict = {}
        self._preview_debounce_timer: Optional[threading.Timer] = None

        # UI element references (set after ui build)
        self.ui = {}  # holds refs to UI elements by name

    def get_ca(self):
        # Slider units are hundredths of a percent (-300..300 == +/-3.00%).
        return (1.0 + self.ca_red / 10000.0, 1.0 + self.ca_blue / 10000.0)

    def request_preview_refresh(self):
        """Debounced: recompute preview 350ms after the last slider/param change."""
        if self._preview_debounce_timer:
            self._preview_debounce_timer.cancel()
        def _fire():
            try:
                if self.pairs:
                    threading.Thread(target=_load_pair_preview_bg,
                                     args=(self, self.pair_idx), daemon=True).start()
            except Exception:
                import traceback
                self._q.put(("log", "FIRE ERR: " + traceback.format_exc()))
        self._preview_debounce_timer = threading.Timer(0.35, _fire)
        self._preview_debounce_timer.start()

    def is_running(self):
        return (self._pipeline is not None or self._stack_pipe is not None
                or self._colmap_pipe is not None or self._batch_pipe is not None)

    def start_matte(self):
        if not self.pairs:
            ui.notify("Load a project folder with black/white pairs first.", type="warning"); return
        self._clear_log()
        self._pipeline = MattePipeline(
            pairs=self.pairs, bg_white=kelvin_to_linear(self.kelvin),
            alpha_min=self.alpha_min, cal=self.cal,
            grey_brightness=self.grey_brightness, alpha_lut=self.alpha_lut,
            sharpen=self.sharpen, solidify=self.solidify, ca=self.get_ca(),
            defringe=self.defringe, exposure=self.exposure,
            on_progress=lambda n,t: self._q.put(("progress",n,t)),
            on_log=lambda m: self._q.put(("log",m)),
            on_done=lambda ok,m: self._q.put(("done",ok,m)),
            on_preview=lambda i,imgs,r: self._q.put(("batch_preview",i,imgs,r)),
        )
        self._set_ui_running(True)
        threading.Thread(target=self._pipeline.run, daemon=True).start()

    def start_batch(self):
        """Full matte -> stack -> COLMAP over every rotation, one at a time."""
        if not self.pairs or not self._stacks:
            ui.notify("Load a project folder first.", type="warning"); return
        self._clear_log()
        matte_kw = dict(
            bg_white=kelvin_to_linear(self.kelvin), alpha_min=self.alpha_min,
            cal=self.cal, grey_brightness=self.grey_brightness,
            alpha_lut=self.alpha_lut, sharpen=self.sharpen,
            solidify=self.solidify, ca=self.get_ca(), defringe=self.defringe,
            exposure=self.exposure)
        stack_kw = dict(
            denoise=self.stack_denoise, sharpen=self.stack_sharpen,
            sharpen_radius=self.stack_sharpen_r, kernel_size=self.stack_kernel,
            gen_kernel=self.stack_genk, min_size=self.stack_minsize,
            alpha_lut=self.stack_alpha_lut, halo_fix=True,
            halo_pct=self.stack_halo_pct, align=self.stack_align)
        self._batch_pipe = BatchPipeline(
            pairs=self.pairs, stacks=self._stacks, proj_dir=Path(self.proj_path),
            matte_kw=matte_kw, stack_kw=stack_kw,
            delete_mattes=self.batch_delete_mattes,
            pipelined=self.batch_pipelined,
            reprocess=self.batch_reprocess,
            clean_colmap=self.batch_clean_colmap,
            mask_threshold=self.mask_threshold,
            soft_alpha=self.batch_soft_alpha,
            rgba_images=self.rgba_images,
            on_progress=lambda n,t: self._q.put(("progress",n,t)),
            on_log=lambda m: self._q.put(("log",m)),
            on_done=lambda ok,m: self._q.put(("done",ok,m)),
            on_preview=lambda imgs: self._q.put(("stack_preview",imgs)),
            on_stage=lambda st,lab,f: self._q.put(("batch_stage",st,lab,f)),
            on_stats=lambda d: self._q.put(("batch_stats",d)),
        )
        self._set_ui_running(True)
        threading.Thread(target=self._batch_pipe.run, daemon=True).start()

    def start_stacking(self):
        if self.proj_path:
            self.matte_dirs = discover_matte_dirs(Path(self.proj_path))
        if not self.matte_dirs:
            ui.notify("No matte/ folders found. Run 'Export Mattes' first.", type="warning"); return
        self._clear_log()
        self._stack_pipe = StackPipeline(
            matte_dirs=self.matte_dirs,
            on_progress=lambda n,t: self._q.put(("progress",n,t)),
            on_log=lambda m: self._q.put(("log",m)),
            on_done=lambda ok,m: self._q.put(("done",ok,m)),
            on_preview=lambda imgs: self._q.put(("stack_preview",imgs)),
            denoise=self.stack_denoise, sharpen=self.stack_sharpen,
            sharpen_radius=self.stack_sharpen_r, kernel_size=self.stack_kernel,
            gen_kernel=self.stack_genk, min_size=self.stack_minsize,
            alpha_lut=self.stack_alpha_lut, halo_fix=True,
            halo_pct=self.stack_halo_pct, align=self.stack_align,
        )
        self._set_ui_running(True)
        threading.Thread(target=self._stack_pipe.run, daemon=True).start()

    def start_colmap(self):
        if not self.stacked_files:
            ui.notify("No stacked TIFFs found. Run 'Stack Mattes' first.", type="warning"); return
        self._clear_log()
        self._colmap_pipe = ColmapMaskPipeline(
            project_dir=Path(self.proj_path), stacked_files=self.stacked_files,
            mask_threshold=self.mask_threshold, soft_alpha=self.batch_soft_alpha,
            rgba_images=self.rgba_images,
            on_progress=lambda n,t: self._q.put(("progress",n,t)),
            on_log=lambda m: self._q.put(("log",m)),
            on_done=lambda ok,m: self._q.put(("done",ok,m)),
        )
        self._set_ui_running(True)
        threading.Thread(target=self._colmap_pipe.run, daemon=True).start()

    def cancel(self):
        for p in [self._pipeline, self._stack_pipe, self._colmap_pipe,
                  self._batch_pipe]:
            if p: p.cancel()

    def show_output(self):
        if not self.pairs:
            ui.notify("No project loaded.", type="warning"); return
        os.system(f'open "{self.pairs[0].out_path.parent}"')

    @staticmethod
    def _js_status(el_id: str, text: str, css_class: str = "status-muted"):
        """Update a status span in the DOM via JS — works across all NiceGUI clients."""
        safe = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        ui.run_javascript(
            f'(function(){{var e=document.getElementById("{el_id}");'
            f'if(e){{e.textContent="{safe}";e.className="{css_class}";}}}})();'
        )

    def reload_project(self):
        p = self.proj_path.strip()
        if not p: return
        if getattr(self, "_scanning", False): return   # don't start a second scan
        proj = Path(p)
        if not proj.is_dir():
            self._js_status("pairs-status", f"Folder not found: {p[:40]}", "status-err")
            return
        self._raw_cache.clear()
        self._scanning = True
        self._js_status("pairs-status", "Scanning project…")

        def _scan():
            try:
                pairs, matte_dirs, stacked, cal_paths = _scan_project(proj)
                self._q.put(("project_scanned", pairs, matte_dirs, stacked, proj, cal_paths))
            except Exception as exc:
                self._q.put(("project_scan_err", str(exc)))
            finally:
                self._scanning = False

        threading.Thread(target=_scan, daemon=True).start()

    def _auto_find_flats(self, proj: Path):
        """Search project folder for flat/calibration RAW files and populate cal fields."""
        RAW_EXTS = ("*.ARW", "*.arw", "*.CR3", "*.cr3", "*.CR2", "*.cr2",
                    "*.NEF", "*.nef", "*.RAF", "*.raf", "*.DNG", "*.dng")

        def _first_raw(directory: Path):
            if not directory.is_dir(): return ""
            for ext in RAW_EXTS:
                # Skip macOS resource forks ("._name.ARW"). They match the glob on
                # external volumes and sort BEFORE real files, so an unfiltered
                # files[0] would hand back a 4KB non-RAW stub as the calibration.
                files = sorted(f for f in directory.glob(ext)
                               if not f.name.startswith("."))
                if files: return str(files[0])
            return ""

        def _root_file(stems):
            """Look for a single RAW file in proj root matching any of the given stems."""
            for ext in ("ARW", "arw", "CR3", "cr3", "CR2", "cr2",
                        "NEF", "nef", "RAF", "raf", "DNG", "dng"):
                for stem in stems:
                    p = proj / f"{stem}.{ext}"
                    if p.exists(): return str(p)
            return ""

        def _find_cal(root_stems, folder_names):
            # 1. Root-level single file (e.g. cal_black.ARW)
            hit = _root_file(root_stems)
            if hit: return hit
            # 2. Named subdirectory
            for name in folder_names:
                hit = _first_raw(proj / name)
                if hit: return hit
            # 3. Inside a 'flats' parent folder
            flats = proj / "flats"
            if flats.is_dir():
                for name in folder_names:
                    hit = _first_raw(flats / name)
                    if hit: return hit
            return ""

        found_black = _find_cal(
            ["cal_black", "black_cal", "flat_black", "black_flat"],
            ["flat_black", "black_flat", "black", "slot_A_black_flat", "flats_black"])
        found_white = _find_cal(
            ["cal_white", "white_cal", "flat_white", "white_flat"],
            ["flat_white", "white_flat", "white", "slot_B_white_flat", "flats_white"])
        found_grey  = _find_cal(
            ["cal_grey", "grey_cal", "flat_grey", "grey_flat",
             "cal_gray", "gray_cal", "flat_gray", "gray_flat"],
            ["flat_grey", "grey_flat", "grey", "slot_C_grey_flat",
             "flats_grey", "gray", "flat_gray"])

        updates = {}
        for attr, found in [
            ("cal_black_path", found_black),
            ("cal_white_path", found_white),
            ("cal_grey_path",  found_grey),
        ]:
            if found and not getattr(self, attr):
                updates[attr] = found

        if updates:
            self._q.put(("flats_found", updates))

    def _set_ui_running(self, running: bool):
        rd = 'true' if running else 'false'
        cd = 'false' if running else 'true'
        ui.run_javascript(f"""
          ['run-btn','stack-run-btn','colmap-btn','batch-btn'].forEach(function(id){{
            var el=document.getElementById(id); if(el) el.disabled={rd};
          }});
          ['cancel-btn','stack-cancel-btn','batch-cancel-btn'].forEach(function(id){{
            var el=document.getElementById(id); if(el) el.disabled={cd};
          }});
        """)
        if not running:
            self._pipeline = None
            self._stack_pipe = None
            self._colmap_pipe = None
            self._batch_pipe = None

    def _clear_log(self):
        for key in ["matte_log", "stack_log", "log", "batch_log"]:
            el = self.ui.get(key)
            if el: el.clear()

    def _append_log(self, msg):
        for key in ["matte_log", "log", "batch_log"]:
            el = self.ui.get(key)
            if el: el.push(msg)
        # Surface errors as toast notifications so they're never missed
        if any(msg.startswith(p) for p in ("PREVIEW ERR", "DISPLAY ERR", "FIRE ERR", "SLIDER ERR", "CURVE ERR")):
            ui.notify(msg[:120], type="negative", timeout=8000)

    def poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                k = item[0]
                if k == "log":
                    self._append_log(item[1])
                elif k == "progress":
                    _, n, t = item
                    pct = n/t*100
                    ui.run_javascript(
                        f'["prog-fill","stack-prog-fill"].forEach(function(id){{'
                        f'var f=document.getElementById(id);if(f)f.style.width="{pct:.1f}%";}});'
                    )
                    for pl_key in ["prog_label", "stack_prog_label"]:
                        pl = self.ui.get(pl_key)
                        if pl: pl.set_text(f"{n:.0f} / {t}  ({pct:.0f}%)")
                elif k == "batch_stage":
                    _, stage, label, frac = item
                    txt = label if label else "—"
                    ui.run_javascript(
                        f'(function(){{'
                        f'var t=document.getElementById("b-{stage}");'
                        f'var f=document.getElementById("b-{stage}-fill");'
                        f'if(t)t.textContent={txt!r};'
                        f'if(f)f.style.width="{frac*100:.1f}%";}})()'
                    )
                elif k == "batch_stats":
                    d = item[1]
                    def _hms(x):
                        if x is None: return "—"
                        x=int(x); return f"{x//3600}:{(x%3600)//60:02d}:{x%60:02d}" if x>=3600 else f"{x//60}:{x%60:02d}"
                    pct = 100.0*d["finished"]/d["total"] if d["total"] else 0.0
                    ui.run_javascript(
                        f'(function(){{var S=function(i,v){{var e=document.getElementById(i);if(e)e.textContent=v;}};'
                        f'S("b-done","{d["done"]}");S("b-skip","{d["skipped"]}");S("b-fail","{d["failed"]}");'
                        f'S("b-elapsed","{_hms(d["elapsed"])}");S("b-eta","{_hms(d["eta"])}");'
                        f'S("b-freed","{d["freed"]/1e9:.1f} GB");'
                        f'S("b-overall","{d["finished"]} / {d["total"]} rotations");'
                        f'var f=document.getElementById("b-overall-fill");'
                        f'if(f)f.style.width="{pct:.1f}%";}})()'
                    )
                elif k == "done":
                    _, ok, msg = item
                    self._set_ui_running(False)
                    if ok:
                        ui.run_javascript(
                            '["prog-fill","stack-prog-fill"].forEach(function(id){'
                            'var f=document.getElementById(id);if(f)f.style.width="100%";});'
                        )
                    for pl_key in ["prog_label", "stack_prog_label"]:
                        pl = self.ui.get(pl_key)
                        if pl:
                            pl.set_text("COMPLETE ✓" if ok else "FAILED ✗")
                            pl.classes(remove="status-ok status-err", add="status-ok" if ok else "status-err")
                    if not ok and msg:
                        ui.notify(f"Error: {msg}", type="negative", timeout=8000)
                    if ok and self.proj_path:
                        self.reload_project()
                elif k == "batch_preview":
                    _, idx, imgs, raws = item
                    self.pair_idx = idx
                    self._prev_imgs = imgs
                    self._update_preview_image()
                elif k == "stack_preview":
                    # Retain it: the rescan triggered by "done" would otherwise
                    # replace this with a matte-frame preview a moment later.
                    self._stack_prev_imgs = item[1]
                    self._keep_stack_preview = True
                    self._prev_imgs = item[1]
                    self._update_preview_image()
                elif k == "preview":
                    self._prev_imgs = item[1]
                    self._update_preview_image()
                elif k == "preview_err":
                    pass
                elif k == "project_scanned":
                    _, pairs, matte_dirs, stacked, proj, cal_paths = item
                    self.pairs         = pairs
                    self.matte_dirs    = matte_dirs
                    self.stacked_files = stacked
                    # Group pairs into stacks by rotation directory
                    from collections import OrderedDict as _OD
                    stack_map = _OD()
                    for idx, p in enumerate(pairs):
                        key = p.out_path.parent
                        if key not in stack_map: stack_map[key] = []
                        stack_map[key].append(idx)
                    self._stacks = list(stack_map.values())
                    self.stack_idx = 0; self.frame_idx = 0
                    if self._stacks: self.pair_idx = self._stacks[0][0]
                    n = len(pairs)
                    rot_count = len(self._stacks)
                    pairs_txt = (
                        f"{n} pair{'s' if n!=1 else ''} found across {rot_count} rotation{'s' if rot_count!=1 else ''}"
                        if n else "No black/white pairs found"
                    )
                    self._js_status("pairs-status", pairs_txt, "status-ok" if n else "status-muted")
                    if hasattr(self, "_update_nav_labels"): self._update_nav_labels()
                    m = len(matte_dirs)
                    self._js_status("matte-status",
                        f"{m} matte/ stack{'s' if m!=1 else ''} ready" if m else "No matte/ folders found",
                        "status-ok" if m else "status-muted")
                    # batch readiness: how many rotations still need work
                    try:
                        todo=0
                        for grp in self._stacks:
                            if not grp: continue
                            rd=pairs[grp[0]].out_path.parent.parent
                            sp=rd/"stacked"/f"{rd.name}_stacked.tif"
                            ip=Path(proj)/"colmap_images"/f"{rd.name}.png"
                            if not (sp.exists() and ip.exists()): todo+=1
                        rc=len(self._stacks)
                        if not rc:
                            txt="No rotations found"
                        elif self.batch_reprocess:
                            txt=(f"{rc} rotation{'s' if rc!=1 else ''} — all will be "
                                 f"reprocessed (Overwrite is on)")
                        else:
                            txt=(f"{rc} rotation{'s' if rc!=1 else ''} — {todo} to "
                                 f"process, {rc-todo} already complete")
                        self._js_status("batch-status", txt,
                                        "status-ok" if rc else "status-muted")
                    except Exception:
                        pass
                    s = len(stacked)
                    self._js_status("stacked-status",
                        f"{s} stacked TIFF{'s' if s!=1 else ''} ready" if s else "No stacked TIFFs — run Stack Mattes first",
                        "status-ok" if s else "status-muted")
                    if getattr(self, "_keep_stack_preview", False):
                        # This rescan was triggered by a finished stack. Restore the
                        # stacked result rather than loading a matte frame over it.
                        self._keep_stack_preview = False
                        if getattr(self, "_stack_prev_imgs", None) is not None:
                            self._prev_imgs = self._stack_prev_imgs
                            self._update_preview_image()
                    elif n > 0:
                        threading.Thread(target=_load_pair_preview_bg,
                                         args=(self, self.pair_idx), daemon=True).start()
                    # Apply cal paths found during walk (already on event loop)
                    if cal_paths:
                        self._q.put(("flats_found", cal_paths))
                elif k == "project_scan_err":
                    self._js_status("pairs-status", f"Scan error: {item[1][:60]}", "status-err")
                elif k == "cal_loaded":
                    self.cal_loading = False
                    self.cal = item[1]
                    parts = []
                    if self.cal.black_full is not None: parts.append("black ✓")
                    if self.cal.white_full is not None: parts.append("white ✓")
                    if self.cal.grey_full  is not None: parts.append("grey ✓")
                    active = self.cal.available
                    cal_txt = "Cal: " + "  ".join(parts) + ("  — per-pixel BG active" if active else "")
                    self._js_status("cal-status", cal_txt, "status-ok" if active else "status-muted")
                    # Re-run preview now that cal is available (first preview ran without it)
                    self.request_preview_refresh()
                elif k == "cal_err":
                    self.cal_loading = False
                    self._js_status("cal-status", f"Cal error: {item[1][:50]}", "status-err")
                elif k == "flats_found":
                    updates = item[1]
                    for attr, found in updates.items():
                        setattr(self, attr, found)
                        fid = f"fi_{attr}"
                        ui.run_javascript(
                            f'var el=document.getElementById("{fid}");if(el)el.value={repr(found)};'
                        )
                    self.load_cal_images()
        except queue.Empty:
            pass

    _BG_SOLID = {
        "black":   (  0,   0,   0), "kelvin":  (200, 169, 110),
        "white":   (255, 255, 255), "grey":    (136, 136, 136),
        "red":     (224,  80,  80), "green":   ( 80, 192,  80),
        "blue":    ( 64, 112, 224), "cyan":    ( 64, 192, 192),
        "magenta": (192,  80, 192), "yellow":  (208, 192,  64),
    }

    def _composite_on_bg(self, fg8, alpha_raw):
        """Composite sharpened uint8 fg (gamma-encoded) over the current _comp_bg."""
        bg_key = getattr(self, "_comp_bg", "checker")
        h, w = fg8.shape[:2]
        if bg_key == "checker":
            bg = _checkerboard(h, w).astype(np.float32)
        else:
            c = self._BG_SOLID.get(bg_key, (80, 80, 80))
            bg = np.full((h, w, 3), [float(x) for x in c], dtype=np.float32)
        a = np.clip(alpha_raw[:,:,None], 0, 1)
        comp = fg8.astype(np.float32) * a + bg * (1 - a)
        return comp.clip(0, 255).astype(np.uint8)

    # Temp file served as a static route — avoids embedding MB-sized base64 in JS
    _PREVIEW_TMP = "/tmp/mattepro_preview.png"
    _ALPHA_TMP   = "/tmp/mattepro_alpha.png"

    def _update_preview_image(self):
        imgs = self._prev_imgs
        if not imgs: return
        mode = self.view_mode
        if mode == "composite" and "fg8" in imgs and "alpha_raw" in imgs:
            img_arr = self._composite_on_bg(imgs["fg8"], imgs["alpha_raw"])
        else:
            img_arr = imgs.get(mode) if mode in imgs else imgs.get("composite")
        if img_arr is None: return
        try:
            # Display image stays OPAQUE RGB. Putting the matte alpha in this PNG's A
            # channel would make the canvas premultiply round-trip zero the RGB wherever
            # alpha==0, destroying the RGB readout over the background. The alpha map is
            # published as a separate greyscale PNG and sampled from its own canvas.
            if img_arr.ndim == 2:
                pil = Image.fromarray(img_arr, "L")
            else:
                pil = Image.fromarray(img_arr[:, :, :3].astype(np.uint8), "RGB")
            pil.save(self._PREVIEW_TMP, format="PNG")

            alpha_src = imgs.get("alpha_raw")
            if alpha_src is None and "alpha" in imgs:
                a_img = imgs["alpha"]
                alpha_src = (a_img[:, :, 0] if a_img.ndim == 3 else a_img).astype(np.float32) / 255.0
            has_alpha = (alpha_src is not None and alpha_src.shape[:2] == img_arr.shape[:2])
            if has_alpha:
                Image.fromarray(
                    (np.clip(alpha_src, 0, 1) * 255).round().astype(np.uint8), "L"
                ).save(self._ALPHA_TMP, format="PNG")
            img_h, img_w = img_arr.shape[:2]
            import time as _time
            ts = int(_time.time() * 1000)
            src = f"/preview/mattepro_preview.png?t={ts}"
            asrc = f"/preview/mattepro_alpha.png?t={ts}" if has_alpha else ""
            # Layout/zoom is owned entirely by the client-side controller
            # (window._mpZoom). Python only publishes the new image and its
            # natural size; gestures never round-trip to the server.
            ui.run_javascript(
                f'(function(){{'
                f'var i=document.getElementById("preview-img-el");'
                f'var pa=document.querySelector(".preview-area");'
                f'if(!i||!pa)return;'
                f'var ph=document.getElementById("preview-placeholder");'
                f'if(ph){{ph.style.display="none";'
                f'        if(ph.parentElement&&ph.parentElement!==pa)'
                f'          ph.parentElement.style.display="none";}}'
                f'i.style.display="block";'
                f'i.onload=function(){{'
                f'  var cv=document.getElementById("px-canvas");'
                f'  if(cv){{cv.width=i.naturalWidth;cv.height=i.naturalHeight;'
                f'         var ctx=cv.getContext("2d");ctx.drawImage(i,0,0);cv._ctx=ctx;}}'
                f'  if(window._mpZoom)window._mpZoom.onNewImage({img_w},{img_h},true);'
                f'}};'
                f'i.src={repr(src)};'
                # Separate greyscale alpha map -> its own canvas, so the hover
                # readout reports true matte alpha without touching the RGB readout.
                f'var acv=document.getElementById("px-canvas-a");'
                f'if(acv){{'
                f'  var asrc={repr(asrc)};'
                f'  if(asrc){{'
                f'    var ai=acv._img||new Image();acv._img=ai;'
                f'    ai.onload=function(){{'
                f'      acv.width=ai.naturalWidth;acv.height=ai.naturalHeight;'
                f'      var actx=acv.getContext("2d");actx.drawImage(ai,0,0);acv._ctx=actx;'
                f'    }};'
                f'    ai.src=asrc;'
                f'  }}else{{acv._ctx=null;}}'
                f'}}'
                f'}})()'
            )
        except Exception:
            import traceback
            self._q.put(("log", "DISPLAY ERR: " + traceback.format_exc()))

    def derive_cal(self):
        """Reconstruct calibration from the captures when no cal shots exist."""
        if not self.pairs:
            self._q.put(("log", "Derive calibration: load a project folder first."))
            return
        if self.cal_loading:
            return
        self.cal_loading = True
        self._q.put(("log", "Deriving calibration from captures…"))
        def worker():
            try:
                cal, msg = derive_calibration(
                    self.pairs, max_frames=30,
                    on_log=lambda m: self._q.put(("log", m)),
                    should_run=lambda: True)
                if cal is None:
                    self._q.put(("log", f"Derive calibration failed: {msg}"))
                    self.cal_loading = False
                    return
                self._q.put(("log", f"Calibration derived — {msg}"))
                self._q.put(("cal_loaded", cal))
            except Exception as e:
                self._q.put(("cal_err", str(e)))
            finally:
                self.cal_loading = False
        threading.Thread(target=worker, daemon=True).start()

    def load_cal_images(self):
        if self.cal_loading: return
        paths = [self.cal_black_path, self.cal_white_path, self.cal_grey_path]
        if not any(p.strip() for p in paths): return
        self.cal_loading = True
        def worker():
            try:
                cal = CalImages()
                if self.cal_black_path:
                    p = Path(self.cal_black_path)
                    if p.exists():
                        cal.black_full = _read_raw_linear(p)
                        cal.black_half = _read_raw_linear(p, half_size=True)
                if self.cal_white_path:
                    p = Path(self.cal_white_path)
                    if p.exists():
                        cal.white_full = _read_raw_linear(p)
                        cal.white_half = _read_raw_linear(p, half_size=True)
                if self.cal_grey_path:
                    p = Path(self.cal_grey_path)
                    if p.exists():
                        cal.grey_full = _read_raw_linear(p)
                        cal.grey_half = _read_raw_linear(p, half_size=True)
                self._q.put(("cal_loaded", cal))
            except Exception as e:
                self._q.put(("cal_err", str(e)))
        threading.Thread(target=worker, daemon=True).start()


state = MatteApp()


def _load_pair_preview_bg(app_state, idx):
    """Background-thread helper — caches half-size linear RAW data so slider changes skip disk reads."""
    if idx >= len(app_state.pairs): return
    pair = app_state.pairs[idx]
    try:
        # Cache raw linear data (no CA) so repeated compute_preview calls (slider changes) are fast
        cached = app_state._raw_cache.get(idx)
        if cached is None:
            cb_raw = _read_raw_linear(pair.black, half_size=True)
            cw_raw = _read_raw_linear(pair.white, half_size=True)
            cg_raw = _read_raw_linear(pair.grey, half_size=True) if pair.grey else None
            app_state._raw_cache[idx] = (cb_raw, cw_raw, cg_raw)
            if len(app_state._raw_cache) > 5:
                app_state._raw_cache.popitem(last=False)
        else:
            cb_raw, cw_raw, cg_raw = cached

        cb, cw, cg = cb_raw, cw_raw, cg_raw

        cal = app_state.cal
        if cal and cal.available and cal.white_half is not None:
            cw_shape, cal_shape = cb.shape[:2], cal.white_half.shape[:2]
            if cw_shape != cal_shape:
                app_state._q.put(("log", f"SIZE MISMATCH: pair {cw_shape} vs cal {cal_shape}"))
        imgs = compute_preview(cb, cw, cg, kelvin_to_linear(app_state.kelvin),
                               app_state.alpha_min, cal, app_state.grey_brightness,
                               app_state.alpha_lut, app_state.sharpen, app_state.solidify, app_state.defringe,
                               ca=app_state.get_ca(), exposure=app_state.exposure)
        app_state._q.put(("preview", imgs))
    except Exception:
        import traceback
        app_state._q.put(("log", "PREVIEW ERR: " + traceback.format_exc()))


# ── File dialog helper (osascript — no tkinter/PyWebView conflict) ────────────

def _pick_dir():
    import subprocess
    r = subprocess.run(
        ["osascript", "-e",
         'tell application "Finder"\nactivate\nset f to choose folder\nreturn POSIX path of f\nend tell'],
        capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else ""

def _pick_file():
    import subprocess
    r = subprocess.run(
        ["osascript", "-e",
         'tell application "Finder"\nactivate\nset f to choose file\nreturn POSIX path of f\nend tell'],
        capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else ""


# ── UI helpers ────────────────────────────────────────────────────────────────

def _section(title):
    ui.html(f'<span class="card-title">{title}</span>')

def _slider_row(label, min_val, max_val, step, init, fmt, attr):
    """Param row with label / Quasar slider / value label — all in the right context."""
    with ui.element("div").classes("param-row"):
        # Caption line: label left, live value right — above a full-width slider.
        with ui.element("div").classes("param-head"):
            ui.html(f'<span class="param-label">{label}</span>')
            val_lbl = ui.html(f'<span class="param-value">{fmt(init)}</span>')

        def _on_change(e, a=attr, f=fmt):
            try:
                val = e.value
                setattr(state, a, val)
                val_lbl.set_content(f'<span class="param-value">{f(val)}</span>')
                state.request_preview_refresh()
            except Exception:
                import traceback
                state._q.put(("log", "SLIDER ERR: " + traceback.format_exc()))

        sl = ui.slider(min=min_val, max=max_val, step=step, value=init,
                       on_change=_on_change).props("dense")
    return sl, val_lbl

def _raw_btn(label, onclick_js=None, classes="mp-btn-sm", disabled=False, style=""):
    """Render a raw <button> that escapes Quasar styling entirely."""
    dis = "disabled" if disabled else ""
    onclick = f'onclick="{onclick_js}"' if onclick_js else ""
    s = f'style="{style}"' if style else ""
    return ui.html(f'<button class="{classes}" {dis} {onclick} {s}>{label}</button>')

def _file_row(label_txt, attr, allow_dir=False):
    """File-path input row with Browse button using raw <input> and <button>."""
    field_id = f"fi_{attr}"
    with ui.element("div").classes("file-row"):
        ui.html(f'<span class="file-label">{label_txt}</span>')
        inp = ui.html(
            f'<input id="{field_id}" class="file-input" '
            f'placeholder="…" value="{getattr(state, attr)}" />'
        )
        async def browse(a=attr, fid=field_id, is_dir=allow_dir):
            if is_dir:
                path = await asyncio.to_thread(_pick_dir)
            else:
                path = await asyncio.to_thread(_pick_file)
            if path:
                setattr(state, a, path)
                await ui.run_javascript(
                    f'document.getElementById("{fid}").value = {repr(path)};'
                )
                if a == "proj_path":
                    state.reload_project()
                else:
                    state.load_cal_images()
        browse_btn = ui.html(f'<button class="mp-btn-sm">Browse</button>')
        browse_btn.on("click", browse)
    # Wire native input change event back to state
    async def on_native_input(e, a=attr):
        val = await ui.run_javascript(f'document.getElementById("{field_id}").value')
        setattr(state, a, val or "")
        if a == "proj_path":
            state.reload_project()
        else:
            state.load_cal_images()
    inp.on("change", on_native_input)
    return inp


def _build_matte_panel():
    with ui.element("div").classes("mp-panel"):

        # 1. PROJECT FOLDER
        with ui.element("div").classes("card"):
            _section("1.  PROJECT FOLDER")
            _file_row("Folder", "proj_path", allow_dir=True)
            ui.html('<span id="pairs-status" class="status-muted">No project loaded</span>')

        # 2. BACKGROUND CALIBRATION
        with ui.element("div").classes("card"):
            _section("2.  BACKGROUND CALIBRATION  ·  optional")
            ui.html('<p class="status-muted" style="margin-bottom:12px">Empty-frame captures — enables per-pixel BG correction.</p>')
            _file_row("Black cal", "cal_black_path")
            _file_row("White cal", "cal_white_path")
            _file_row("Grey cal",  "cal_grey_path")
            ui.html('<span id="cal-status" class="status-muted">No calibration images</span>')
            ui.element("div").style("height:8px")
            derive_btn = ui.html(
                '<button class="mp-btn-secondary" id="derive-cal-btn" '
                'title="Reconstruct the background from the captures themselves: '
                'sample frames across rotations, take a percentile so the specimen '
                'is rejected where it moves, then fit a smooth surface through the '
                'region it always covers.">&#9881;&nbsp; DERIVE FROM CAPTURES</button>')
            derive_btn.on("click", state.derive_cal)
            ui.html('<p class="status-muted" style="margin-top:8px;font-size:11px;line-height:1.5">'
                    'Use when calibration shots were not taken. Real empty-frame '
                    'captures are better — this infers the background instead of '
                    'measuring it.</p>')

        # 3. MATTING SETTINGS
        with ui.element("div").classes("card"):
            _section("3.  MATTING SETTINGS")
            _slider_row("White BG Kelvin",    2700, 9000,  100,  state.kelvin,           lambda v: f"{int(v)}K",              "kelvin")
            _slider_row("Min opacity cut",    0.0,  0.2,   0.005,state.alpha_min,         lambda v: f"{v:.3f}",                "alpha_min")
            _slider_row("Grey BG brightness", 10,   100,   5,    state.grey_brightness,   lambda v: f"{int(v)}%",              "grey_brightness")
            _slider_row("Fill body",          0,    100,   1,    state.solidify,          lambda v: f"{int(v)}%" if v else "off","solidify")

        # 4. ALPHA CURVE
        with ui.element("div").classes("card"):
            _section("4.  ALPHA CURVE")
            ui.html('<canvas id="curve-matte" class="curve-canvas"></canvas>')
            with ui.element("div").style(
                    "display:flex;align-items:center;justify-content:space-between;"
                    "gap:12px;margin-top:10px"):
                ui.html('<span class="curve-hint"><b>Click</b> add · '
                        '<b>Right-click</b> remove · <b>Drag</b> adjust</span>')
                reset_btn = ui.html('<button class="mp-btn-sm">Reset</button>')
                reset_btn.on("click", lambda: ui.run_javascript("if(window.curve_matte_changed_reset)window.curve_matte_changed_reset()"))
            def _on_curve_matte(e):
                try:
                    pts = e.args["pts"]   # [[x,y], ...] control points from JS
                    lut = _monotone_cubic_lut(pts)
                    lut[0] = 0.0   # black point always 0 — curve cannot raise background
                    state._q.put(("log", f"Curve: {len(pts)} pts, range [{lut.min():.3f}, {lut.max():.3f}]"))
                    state.alpha_lut = np.clip(lut, 0.0, 1.0)
                    state.request_preview_refresh()
                except Exception:
                    import traceback
                    state._q.put(("log", "CURVE ERR: " + traceback.format_exc()))
            ui.on("curve_matte_changed", _on_curve_matte)

        # 5. CORRECTION
        with ui.element("div").classes("card"):
            _section("5.  CORRECTION")
            _slider_row("Exposure", -200, 200, 1, state.exposure, lambda v: f"{v/100:+.2f} EV" if v else "0 EV", "exposure")
            # -300..300 in steps of 1 == +/-3.00% in 0.01% increments (was 0.1%).
            # get_ca() divides by 10000 to keep the same physical range.
            _slider_row("CA red",   -300, 300, 1, state.ca_red,  lambda v: f"{v/100:+.2f}%" if v else "0.00%", "ca_red")
            _slider_row("CA blue",  -300, 300, 1, state.ca_blue, lambda v: f"{v/100:+.2f}%" if v else "0.00%", "ca_blue")
            _slider_row("Sharpen (USM)", 0, 200, 5, state.sharpen, lambda v: f"{int(v)}%" if v else "off", "sharpen")
            _slider_row("Defringe",      0, 100, 1, state.defringe,lambda v: f"{int(v)}%" if v else "off", "defringe")

        # 6. MATTE EXPORT
        with ui.element("div").classes("card"):
            _section("6.  EXPORT")
            run_html = ui.html('<button class="mp-btn-primary" id="run-btn">▶  EXPORT MATTES</button>')
            run_html.on("click", state.start_matte)
            state.ui["run_btn"] = run_html
            ui.element("div").style("height:8px")
            with ui.element("div").classes("btn-row"):
                cancel_html = ui.html('<button class="mp-btn-secondary" id="cancel-btn" disabled>■  CANCEL</button>')
                cancel_html.on("click", state.cancel)
                state.ui["cancel_btn"] = cancel_html
                finder_html = ui.html('<button class="mp-btn-secondary">⌘  FINDER</button>')
                finder_html.on("click", state.show_output)
            ui.html('<div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>')
            prog_lbl = ui.label("READY").classes("status-muted")
            state.ui["prog_label"] = prog_lbl
            matte_log = ui.log(max_lines=200).style(
                "background:var(--card2);border:1px solid var(--border);"
                "border-radius:var(--rad-sm);font-family:'Menlo',monospace;"
                "font-size:11px;color:var(--sub);padding:10px;"
                "height:160px;line-height:1.65;margin-top:10px"
            )
            state.ui["matte_log"] = matte_log

    state.ui["progress"] = None


def _build_batch_panel():
    """Batch tab: run every rotation and report what each stage is doing."""
    with ui.element("div").classes("mp-panel"):
        with ui.element("div").classes("card"):
            _section("1.  BATCH ALL ROTATIONS")
            ui.html('<p class="status-muted" style="margin-bottom:12px">'
                    'Runs matte → stack → COLMAP for every rotation using the '
                    'settings on the MATTE and STACK tabs.<br>'
                    'Peak disk stays at about one rotation of mattes instead of all '
                    'of them. Resumable: finished rotations are skipped.</p>')
            ui.html('<span id="batch-status" class="status-muted" '
                    'style="display:block;margin-bottom:4px">Load a project folder</span>')

        # ── live progress dashboard ────────────────────────────────────
        with ui.element("div").classes("card"):
            _section("2.  PROGRESS")
            ui.html("""
              <div class="bstat-row"><span class="bstat-k">Overall</span>
                   <span class="bstat-v" id="b-overall">not started</span></div>
              <div class="bbar"><div class="bbar-fill" id="b-overall-fill"></div></div>

              <div class="bstat-row" style="margin-top:14px">
                   <span class="bstat-k">Matting</span>
                   <span class="bstat-v" id="b-matte">—</span></div>
              <div class="bbar"><div class="bbar-fill alt" id="b-matte-fill"></div></div>

              <div class="bstat-row" style="margin-top:14px">
                   <span class="bstat-k">Stacking</span>
                   <span class="bstat-v" id="b-stack">—</span></div>
              <div class="bbar"><div class="bbar-fill alt" id="b-stack-fill"></div></div>

              <div class="bstat-row" style="margin-top:14px">
                   <span class="bstat-k">COLMAP</span>
                   <span class="bstat-v" id="b-colmap">—</span></div>
              <div class="bbar"><div class="bbar-fill alt" id="b-colmap-fill"></div></div>

              <div class="bgrid">
                <div><div class="bnum ok"   id="b-done">0</div><div class="blab">done</div></div>
                <div><div class="bnum"      id="b-skip">0</div><div class="blab">skipped</div></div>
                <div><div class="bnum err"  id="b-fail">0</div><div class="blab">failed</div></div>
                <div><div class="bnum"      id="b-elapsed">0:00</div><div class="blab">elapsed</div></div>
                <div><div class="bnum"      id="b-eta">—</div><div class="blab">eta</div></div>
                <div><div class="bnum"      id="b-freed">0 GB</div><div class="blab">reclaimed</div></div>
              </div>
            """)

        with ui.element("div").classes("card"):
            _section("3.  OPTIONS")
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Delete mattes after each</span>')
                    _dm_lbl = ui.html('<span class="param-value">on</span>')
                _dm = ui.switch("", value=state.batch_delete_mattes).props("dense")
                def _on_dm(e):
                    state.batch_delete_mattes = bool(e.value)
                    _dm_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _dm.on_value_change(_on_dm)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Removed only after that rotation\'s stack AND COLMAP export both '
                    'succeed, so a failure never destroys them.</p>')
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Overwrite existing output</span>')
                    _rp_lbl = ui.html('<span class="param-value">on</span>')
                _rp = ui.switch("", value=state.batch_reprocess).props("dense")
                def _on_rp(e):
                    state.batch_reprocess = bool(e.value)
                    _rp_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _rp.on_value_change(_on_rp)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'On: every rotation is redone and any earlier stacked TIFF, COLMAP '
                    'pair and leftover mattes for it are replaced — use this after '
                    'tuning, so test-run files are not kept.<br>'
                    'Off: rotations already finished are skipped, to resume an '
                    'interrupted batch.</p>')
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Empty COLMAP dirs first</span>')
                    _cc_lbl = ui.html('<span class="param-value">on</span>')
                _cc = ui.switch("", value=state.batch_clean_colmap).props("dense")
                def _on_cc(e):
                    state.batch_clean_colmap = bool(e.value)
                    _cc_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _cc.on_value_change(_on_cc)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Clears colmap_images/ and colmap_masks/ once at the start, so the '
                    'set contains only this batch. Ignored when Overwrite is off, since '
                    'a resume depends on those files.<br>'
                    '<b>Turn off if you build one COLMAP set from several orbits.</b></p>')
            _slider_row("Mask cut", 1, 254, 1, state.mask_threshold,
                        lambda v: f"{int(v)}", "mask_threshold")
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Alpha at or above this becomes 255 in the COLMAP mask, below it '
                    'becomes 0. COLMAP\'s masks are strictly binary — a value of 128 '
                    'gives byte-identical results to 255 — so this cut, not a grey '
                    'ramp, is what decides which pixels it sees.</p>')
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">RGBA images (for Brush)</span>')
                    _rg_lbl = ui.html('<span class="param-value">on</span>')
                _rg = ui.switch("", value=state.rgba_images).props("dense")
                def _on_rg(e):
                    state.rgba_images = bool(e.value)
                    _rg_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _rg.on_value_change(_on_rg)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Writes colmap_images/ as RGBA with the matte in the alpha channel. '
                    'Brush takes transparency from the image itself — its '
                    '<i>alpha-loss-weight</i> applies "if input view has transparency" — '
                    'and ignores separate mask folders. COLMAP reads RGBA fine.<br>'
                    'Turn off only if a downstream tool needs plain RGB.</p>')
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Also write soft alpha</span>')
                    _sa_lbl = ui.html('<span class="param-value">off</span>')
                _sa = ui.switch("", value=state.batch_soft_alpha).props("dense")
                def _on_sa(e):
                    state.batch_soft_alpha = bool(e.value)
                    _sa_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _sa.on_value_change(_on_sa)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Writes the full greyscale matte to <b>alpha_soft/</b> alongside the '
                    'binary COLMAP masks — for 3DGS/NeRF training, which unlike COLMAP '
                    'does use the soft edge.</p>')
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Overlap matte &amp; stack</span>')
                    _bp_lbl = ui.html('<span class="param-value">on</span>')
                _bp = ui.switch("", value=state.batch_pipelined).props("dense")
                def _on_bp(e):
                    state.batch_pipelined = bool(e.value)
                    _bp_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _bp.on_value_change(_on_bp)
            ui.html('<p class="status-muted" style="font-size:11px;margin:-4px 0 12px">'
                    'Mattes the next rotation while the current one stacks. Holds two '
                    'rotations of mattes instead of one.</p>')

        with ui.element("div").classes("card"):
            _section("4.  RUN")
            batch_html = ui.html('<button class="mp-btn-primary" id="batch-btn">'
                                 '\u229e  BATCH ALL</button>')
            batch_html.on("click", state.start_batch)
            state.ui["batch_btn"] = batch_html
            cancel_b = ui.html('<button class="mp-btn-secondary" id="batch-cancel-btn" '
                               'style="margin-top:8px">\u25a0  CANCEL</button>')
            cancel_b.on("click", state.cancel)

        with ui.element("div").classes("card"):
            _section("5.  BATCH LOG")
            blog = ui.log(max_lines=500).style(
                "background:var(--card2);border:1px solid var(--border);"
                "border-radius:var(--rad-sm);padding:10px;height:260px;"
                "font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--sub)")
            state.ui["batch_log"] = blog


def _build_stack_panel():
    with ui.element("div").classes("mp-panel"):

        # 1. FOCUS STACKING
        with ui.element("div").classes("card"):
            _section("1.  FOCUS STACKING")
            ss_avail, ss_ver, ss_problem = _shinestacker_status()
            _ss_label = f"shinestacker {ss_ver}" if ss_ver else "shinestacker"
            ui.html(f'<p class="status-muted" style="margin-bottom:12px">{_ss_label} PyramidAutoStack (alpha-aware).<br>Output → stacked/ alongside matte/.</p>')
            ui.html('<span id="matte-status" class="status-muted" style="display:block;margin-bottom:12px">No matte/ folders found</span>')
            cls = "mp-btn-primary" if ss_avail else "mp-btn-secondary"
            dis = "" if ss_avail else "disabled"
            stack_html = ui.html(f'<button class="{cls}" {dis} id="stack-run-btn">⊞  STACK MATTES</button>')
            stack_html.on("click", state.start_stacking)
            state.ui["stack_btn"] = stack_html
            if not ss_avail:
                ui.html(f'<p class="status-err" style="margin-top:8px">{ss_problem}</p>')
            ui.element("div").style("height:8px")
            stack_cancel = ui.html('<button class="mp-btn-secondary" id="stack-cancel-btn" disabled>■  CANCEL</button>')
            stack_cancel.on("click", state.cancel)
            state.ui["stack_cancel_btn"] = stack_cancel
            ui.html('<div class="prog-track"><div class="prog-fill" id="stack-prog-fill"></div></div>')
            stack_prog_lbl = ui.label("READY").classes("status-muted")
            state.ui["stack_prog_label"] = stack_prog_lbl

        # 2. STACK QUALITY
        with ui.element("div").classes("card"):
            _section("2.  STACK QUALITY")
            _slider_row("Denoise",       0,    100,  1,   state.stack_denoise,  lambda v: f"{int(v)}",     "stack_denoise")
            _slider_row("Sharpen",       0,    200,  1,   state.stack_sharpen,  lambda v: f"{int(v)}%",    "stack_sharpen")
            _slider_row("Sharpen radius",0.5,  5.0,  0.5, state.stack_sharpen_r,lambda v: f"{v:.1f}",      "stack_sharpen_r")
            _slider_row("Pyr kernel",    3,    7,    2,   state.stack_kernel,   lambda v: f"{int(v)}",     "stack_kernel")
            _slider_row("Gen kernel",    0.25, 0.55, 0.01,state.stack_genk,     lambda v: f"{v:.2f}",      "stack_genk")
            _slider_row("Pyr depth px",  8,    128,  8,   state.stack_minsize,  lambda v: f"{int(v)}px",   "stack_minsize")
            # Un-premultiplied fusion is unconditional — it is simply correct, so
            # there is no toggle. This slider only controls the alpha ceiling.
            _slider_row("Halo suppression", 25, 100, 5, state.stack_halo_pct,
                        lambda v: "off" if v >= 100 else f"p{int(v)}", "stack_halo_pct")
            ui.html('<p class="status-muted" style="font-size:11px;margin-top:-4px">'
                    'Caps fused alpha at a per-frame percentile so defocused frames '
                    'cannot widen the silhouette. Lower = tighter edges, but below '
                    'about p50 it also thins antennae and legs.</p>')
            # Real tradeoff (~9% less halo for ~55% more time), so it gets a switch.
            with ui.element("div").classes("param-row"):
                with ui.element("div").classes("param-head"):
                    ui.html('<span class="param-label">Align frames</span>')
                    _al_lbl = ui.html('<span class="param-value">on</span>')
                _al = ui.switch("", value=state.stack_align).props("dense")
                def _on_align(e):
                    state.stack_align = bool(e.value)
                    _al_lbl.set_content(
                        f'<span class="param-value">{"on" if e.value else "off"}</span>')
                _al.on_value_change(_on_align)
            ui.html('<p class="status-muted" style="font-size:11px;margin-top:-4px">'
                    'Focus bracketing changes magnification, so frames drift and '
                    'scale (measured here: 8.8% scale, ~170px shift). Registering '
                    'them narrows the silhouette and sharpens detail. Slower.</p>')

        # 3. ALPHA CURVE
        with ui.element("div").classes("card"):
            _section("3.  STACKED ALPHA CURVE")
            ui.html('<canvas id="curve-stack" class="curve-canvas"></canvas>')
            with ui.element("div").style(
                    "display:flex;align-items:center;justify-content:space-between;"
                    "gap:12px;margin-top:10px"):
                ui.html('<span class="curve-hint"><b>Click</b> add · '
                        '<b>Right-click</b> remove · <b>Drag</b> adjust</span>')
                reset_s = ui.html('<button class="mp-btn-sm">Reset</button>')
                reset_s.on("click", lambda: ui.run_javascript("if(window.curve_stack_changed_reset)window.curve_stack_changed_reset()"))
            def _on_curve_stack(e):
                lut = _monotone_cubic_lut(e.args["pts"])
                lut[0] = 0.0
                state.stack_alpha_lut = np.clip(lut, 0.0, 1.0)
            ui.on("curve_stack_changed", _on_curve_stack)

        # 4. COLMAP MASKS
        with ui.element("div").classes("card"):
            _section("4.  COLMAP MASKS")
            ui.html('<p class="status-muted" style="margin-bottom:12px">Extracts RGB image + binary alpha mask from each stacked TIFF.<br>Output → colmap_images/  colmap_masks/<br>Pass mask dir to COLMAP via --ImageReader.mask_path.</p>')
            ui.html('<span id="stacked-status" class="status-muted" style="display:block;margin-bottom:12px">No stacked TIFFs — run Stack Mattes first</span>')
            colmap_html = ui.html('<button class="mp-btn-secondary" id="colmap-btn">⊡  GEN COLMAP MASKS</button>')
            colmap_html.on("click", state.start_colmap)
            state.ui["colmap_btn"] = colmap_html

        # 5. LOG
        with ui.element("div").classes("card"):
            _section("5.  LOG")
            log = ui.log(max_lines=300).style(
                "background:var(--card2);border:1px solid var(--border);"
                "border-radius:var(--rad-sm);font-family:'Menlo',monospace;"
                "font-size:11px;color:var(--sub);padding:10px;"
                "height:200px;line-height:1.65"
            )
            state.ui["log"] = log


def _build_preview_panel():
    ZOOM_LEVELS = {"fit": "FIT", "1:1": "1:1", "2:1": "2:1", "4:1": "4:1"}
    BG_COLORS = {
        "checker": ("CK",  "#555"),
        "black":   ("K",   "#111"),
        "kelvin":  ("Kv",  "#c8a96e"),
        "white":   ("W",   "#fff"),
        "grey":    ("Gr",  "#888"),
        "red":     ("R",   "#e05050"),
        "green":   ("G",   "#50c050"),
        "blue":    ("B",   "#4070e0"),
        "cyan":    ("C",   "#40c0c0"),
        "magenta": ("M",   "#c050c0"),
        "yellow":  ("Y",   "#d0c040"),
    }

    VIEW_LABELS = {"black":"BLK","white":"WHT","grey":"GRY","alpha":"α","composite":"COMP"}

    with ui.element("div").classes("mp-right"):

        # ── Row 1: view mode + pair nav ──────────────────────────────
        with ui.element("div").classes("preview-bar"):
            ui.html('<span style="font-size:10px;font-weight:700;color:var(--sub);letter-spacing:0.1em;text-transform:uppercase">View</span>')
            view_btns = {}
            with ui.element("div").classes("seg-ctrl"):
                for key in ["black","white","grey","alpha","composite"]:
                    is_active = key == "composite"
                    btn = ui.html(f'<button class="seg-btn{"  active" if is_active else ""}">{VIEW_LABELS[key]}</button>')
                    view_btns[key] = btn

            ui.element("div").style("flex:1")

            # Stack navigator
            ui.html('<span style="font-size:10px;color:var(--sub);letter-spacing:0.06em">STACK</span>')
            stack_prev_btn = ui.html('<button class="mp-btn-sm" style="padding:4px 8px">‹</button>')
            ui.html('<span id="stack-nav-label" style="font-size:11px;color:var(--text);min-width:40px;text-align:center">—</span>')
            stack_next_btn = ui.html('<button class="mp-btn-sm" style="padding:4px 8px">›</button>')

            ui.element("div").style("width:12px")

            # Frame navigator
            ui.html('<span style="font-size:10px;color:var(--sub);letter-spacing:0.06em">FRAME</span>')
            prev_btn = ui.html('<button class="mp-btn-sm" style="padding:4px 8px">‹</button>')
            ui.html('<span id="frame-nav-label" style="font-size:11px;color:var(--text);min-width:40px;text-align:center">—</span>')
            next_btn = ui.html('<button class="mp-btn-sm" style="padding:4px 8px">›</button>')


        # ── Row 2: BG chips + zoom ────────────────────────────────────
        with ui.element("div").style(
            "display:flex;align-items:center;gap:8px;padding:6px 18px;"
            "background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0"
        ):
            ui.html('<span style="font-size:10px;font-weight:700;color:var(--sub);letter-spacing:0.1em;text-transform:uppercase">BG</span>')
            bg_btns = {}
            for key, (lbl, color) in BG_COLORS.items():
                is_active = key == "checker"
                dot = f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{color};margin-right:3px"></span>'
                active_style = "background:var(--accent);color:#fff;border-color:var(--accent)" if is_active else ""
                btn = ui.html(f'<button class="mp-btn-sm" style="padding:3px 8px;font-size:10px;{active_style}">{dot}{lbl}</button>')
                bg_btns[key] = btn

            ui.element("div").style("flex:1")

            ui.html('<span style="font-size:10px;font-weight:700;color:var(--sub);letter-spacing:0.1em;text-transform:uppercase">ZOOM</span>')
            # Live scale readout — pinch zoom is continuous, so the preset
            # buttons alone can't tell you where you are.
            ui.html('<span id="zoom-live" style="font-size:11px;color:var(--sub);'
                    'font-variant-numeric:tabular-nums;min-width:42px;text-align:right">100%</span>')
            zoom_btns = {}
            with ui.element("div").classes("seg-ctrl").props("id=zoom-btns"):
                for key, lbl in ZOOM_LEVELS.items():
                    is_active = key == "fit"
                    btn = ui.html(f'<button class="seg-btn{"  active" if is_active else ""}" '
                                  f'data-zoom="{key}">{lbl}</button>')
                    zoom_btns[key] = btn

        def set_view(key):
            state.view_mode = key
            for k, b in view_btns.items():
                b.set_content(f'<button class="seg-btn{" active" if k == key else ""}">{VIEW_LABELS[k]}</button>')
            state._update_preview_image()

        def set_bg(key):
            state._comp_bg = key
            for k, b in bg_btns.items():
                lbl, color = BG_COLORS[k]
                dot = f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{color};margin-right:3px"></span>'
                active_style = "background:var(--accent);color:#fff;border-color:var(--accent)" if k == key else ""
                b.set_content(f'<button class="mp-btn-sm" style="padding:3px 8px;font-size:10px;{active_style}">{dot}{lbl}</button>')
            state._update_preview_image()

        def set_zoom(key):
            # Zoom is a pure client-side transform — no image regeneration, no
            # server round-trip. Just tell the controller which level to snap to.
            state._zoom = key
            for k, b in zoom_btns.items():
                b.set_content(f'<button class="seg-btn{" active" if k == key else ""}">{ZOOM_LEVELS[k]}</button>')
            ui.run_javascript(
                f'if(window._mpZoom)window._mpZoom.setLevel({key!r});')

        def _update_nav_labels():
            n_stacks = len(state._stacks)
            if not state._stacks:
                stack_txt = "—"; frame_txt = "—"
            else:
                stack_txt = f"{state.stack_idx+1}/{n_stacks}"
                n_frames = len(state._stacks[state.stack_idx])
                frame_txt = f"{state.frame_idx+1}/{n_frames}"
            ui.run_javascript(
                f'var s=document.getElementById("stack-nav-label");if(s)s.textContent={repr(stack_txt)};'
                f'var f=document.getElementById("frame-nav-label");if(f)f.textContent={repr(frame_txt)};'
            )
        state._update_nav_labels = _update_nav_labels

        def nav_stack(delta):
            if not state._stacks: return
            state.stack_idx = (state.stack_idx + delta) % len(state._stacks)
            n_frames = len(state._stacks[state.stack_idx])
            state.frame_idx = min(state.frame_idx, n_frames - 1)
            state.pair_idx = state._stacks[state.stack_idx][state.frame_idx]
            _update_nav_labels()
            threading.Thread(target=_load_pair_preview, args=(state.pair_idx,), daemon=True).start()

        def nav_frame(delta):
            if not state._stacks: return
            n_frames = len(state._stacks[state.stack_idx])
            if n_frames == 0: return
            state.frame_idx = (state.frame_idx + delta) % n_frames
            state.pair_idx = state._stacks[state.stack_idx][state.frame_idx]
            _update_nav_labels()
            threading.Thread(target=_load_pair_preview, args=(state.pair_idx,), daemon=True).start()

        def _load_pair_preview(idx):
            _load_pair_preview_bg(state, idx)  # uses shared cache

        for key, btn in view_btns.items():
            btn.on("click", lambda _, k=key: set_view(k))
        for key, btn in bg_btns.items():
            btn.on("click", lambda _, k=key: set_bg(k))
        for key, btn in zoom_btns.items():
            btn.on("click", lambda _, k=key: set_zoom(k))
        stack_prev_btn.on("click", lambda: nav_stack(-1))
        stack_next_btn.on("click", lambda: nav_stack(+1))
        prev_btn.on("click", lambda: nav_frame(-1))
        next_btn.on("click", lambda: nav_frame(+1))

        # ── Image area ────────────────────────────────────────────────
        # Use ui.element() (not ui.html()) for img and canvas so NiceGUI renders
        # them as direct DOM children of pa — no intervening wrapper divs.
        # Styles are driven from Python so Vue's VDOM stays correct across re-renders.
        _pa_el = ui.element("div").classes("preview-area")
        with _pa_el:
            _img_el = ui.element("img").props("id=preview-img-el").style("display:none")
            ui.html('<span id="preview-placeholder" class="status-muted" style="font-size:13px">Load a project to preview</span>')
            ui.element("canvas").props("id=px-canvas").style("display:none")
            ui.element("canvas").props("id=px-canvas-a").style("display:none")
        state.ui["preview_pa"]  = _pa_el
        state.ui["preview_img"] = _img_el
        # ── Zoom / pan controller ──────────────────────────────────────
        # All gesture handling is client-side against a CSS transform, so pinch
        # and pan stay at display refresh rate and never round-trip to Python.
        ui.add_body_html('''<script>
(function(){
  var Z={s:1,tx:0,ty:0,iw:0,ih:0,fit:1,level:"fit",ready:false};
  window._mpZoom=Z;
  function pa(){return document.querySelector(".preview-area");}
  function img(){return document.getElementById("preview-img-el");}

  function apply(snap){
    var i=img(); if(!i)return;
    if(snap){i.classList.add("mp-snap");setTimeout(function(){i.classList.remove("mp-snap");},200);}
    // Pixelate only when magnified beyond native resolution.
    if(Z.s>1.05){i.classList.add("mp-crisp");}else{i.classList.remove("mp-crisp");}
    i.style.transform="translate("+Z.tx+"px,"+Z.ty+"px) scale("+Z.s+")";
    var lbl=document.getElementById("zoom-live");
    if(lbl)lbl.textContent=(Z.s*100).toFixed(0)+"%";
  }
  // Keep the image from being flung off-screen: always leave it overlapping
  // the viewport, and centre it on whichever axis is smaller than the viewport.
  function clamp(){
    var p=pa(); if(!p)return;
    var cw=p.clientWidth,ch=p.clientHeight,w=Z.iw*Z.s,h=Z.ih*Z.s;
    if(w<=cw){Z.tx=(cw-w)/2;}else{Z.tx=Math.min(0,Math.max(cw-w,Z.tx));}
    if(h<=ch){Z.ty=(ch-h)/2;}else{Z.ty=Math.min(0,Math.max(ch-h,Z.ty));}
  }
  function fitScale(){
    var p=pa(); if(!p||!Z.iw)return 1;
    return Math.min(p.clientWidth/Z.iw,p.clientHeight/Z.ih);
  }
  // Zoom about a fixed point in container space so content under the cursor
  // (or the viewport centre) stays put.
  function zoomAt(ns,px,py,snap){
    ns=Math.max(0.02,Math.min(64,ns));
    var ix=(px-Z.tx)/Z.s, iy=(py-Z.ty)/Z.s;
    Z.s=ns; Z.tx=px-ix*ns; Z.ty=py-iy*ns;
    clamp(); apply(snap);
  }
  Z.setLevel=function(k){
    Z.level=k; var p=pa(); if(!p||!Z.iw)return;
    var cx=p.clientWidth/2, cy=p.clientHeight/2;
    if(k==="fit"){Z.s=fitScale();clamp();apply(true);}
    else{zoomAt({"1:1":1,"2:1":2,"4:1":4}[k]||1,cx,cy,true);}
    markButtons();
  };
  function markButtons(){
    var m=document.getElementById("zoom-btns"); if(!m)return;
    var btns=m.querySelectorAll("button");
    for(var i=0;i<btns.length;i++){
      var k=btns[i].getAttribute("data-zoom");
      var on=(k===Z.level);
      btns[i].className="seg-btn"+(on?" active":"");
    }
  }
  Z.markButtons=markButtons;
  // Called by Python whenever a new preview image is published.
  Z.onNewImage=function(w,h,keep){
    var first=(!Z.ready||Z.iw!==w||Z.ih!==h);
    Z.iw=w;Z.ih=h;Z.ready=true;
    var i=img(); if(i){i.style.width=w+"px";i.style.height=h+"px";}
    Z.fit=fitScale();
    if(first||!keep||Z.level==="fit"){Z.setLevel(Z.level==="fit"?"fit":Z.level);}
    else{clamp();apply(false);}
  };

  function install(){
    var p=pa(); if(!p){setTimeout(install,150);return;}
    if(p._mpZoomWired)return; p._mpZoomWired=true;

    p.addEventListener("wheel",function(e){
      if(!Z.ready)return;
      e.preventDefault();
      var r=p.getBoundingClientRect();
      var px=e.clientX-r.left, py=e.clientY-r.top;
      // macOS trackpad pinch arrives as a wheel event with ctrlKey set.
      if(e.ctrlKey||e.metaKey){
        zoomAt(Z.s*Math.exp(-e.deltaY*0.01),px,py,false);
        Z.level="free";markButtons();
      }else{
        // Two-finger scroll pans; shift swaps the axis.
        var dx=e.shiftKey?-e.deltaY:-e.deltaX, dy=e.shiftKey?0:-e.deltaY;
        Z.tx+=dx; Z.ty+=dy; clamp(); apply(false);
      }
    },{passive:false});

    var drag=null;
    p.addEventListener("pointerdown",function(e){
      if(!Z.ready||e.button!==0)return;
      drag={x:e.clientX,y:e.clientY,tx:Z.tx,ty:Z.ty};
      p.setPointerCapture(e.pointerId); p.classList.add("mp-panning");
    });
    p.addEventListener("pointermove",function(e){
      if(!drag)return;
      Z.tx=drag.tx+(e.clientX-drag.x); Z.ty=drag.ty+(e.clientY-drag.y);
      clamp(); apply(false);
    });
    function endDrag(e){
      if(!drag)return; drag=null; p.classList.remove("mp-panning");
      try{p.releasePointerCapture(e.pointerId);}catch(_){}
    }
    p.addEventListener("pointerup",endDrag);
    p.addEventListener("pointercancel",endDrag);
    // Double-click toggles fit <-> 1:1 at the cursor.
    p.addEventListener("dblclick",function(e){
      if(!Z.ready)return;
      var r=p.getBoundingClientRect();
      if(Math.abs(Z.s-fitScale())<0.01){
        zoomAt(1,e.clientX-r.left,e.clientY-r.top,true);Z.level="1:1";
      }else{Z.setLevel("fit");}
      markButtons();
    });
    window.addEventListener("resize",function(){
      if(!Z.ready)return;
      if(Z.level==="fit"){Z.s=fitScale();}
      clamp();apply(false);
    });
  }
  install();
})();
</script>''')
        # Wire up pixel-readout mouse tracking once the DOM is ready
        ui.add_body_html('''<script>
document.addEventListener("DOMContentLoaded",function(){
  function _mpInitPx(){
    var pa=document.querySelector(".preview-area");
    var rd=document.getElementById("px-readout");
    if(!pa||!rd){setTimeout(_mpInitPx,200);return;}
    pa.addEventListener("mousemove",function(e){
      var i=document.getElementById("preview-img-el");
      var cv=document.getElementById("px-canvas");
      if(!i||!cv||!cv._ctx){rd.style.display="none";return;}
      var rect=i.getBoundingClientRect();
      var cssX=e.clientX-rect.left,cssY=e.clientY-rect.top;
      if(cssX<0||cssY<0||cssX>=rect.width||cssY>=rect.height){rd.style.display="none";return;}
      var nx=Math.round(cssX*cv.width/rect.width);
      var ny=Math.round(cssY*cv.height/rect.height);
      nx=Math.max(0,Math.min(cv.width-1,nx));
      ny=Math.max(0,Math.min(cv.height-1,ny));
      var px=cv._ctx.getImageData(nx,ny,1,1).data;
      var txt="x "+nx+" y "+ny+"  R "+px[0]+" G "+px[1]+" B "+px[2];
      // True matte alpha comes from its own greyscale canvas - lossless, and not
      // subject to the canvas premultiply round-trip that pins A to 255 here.
      var acv=document.getElementById("px-canvas-a");
      if(acv&&acv._ctx&&acv.width>0){
        var ax=Math.max(0,Math.min(acv.width-1,Math.round(cssX*acv.width/rect.width)));
        var ay=Math.max(0,Math.min(acv.height-1,Math.round(cssY*acv.height/rect.height)));
        var av=acv._ctx.getImageData(ax,ay,1,1).data[0];
        txt+="  A "+av;
        if(av===0){txt+=" CLEAR";}else if(av===255){txt+=" SOLID";}
      }else{txt+="  A -";}
      rd.textContent=txt;
      rd.style.display="inline";
    });
    pa.addEventListener("mouseleave",function(){rd.style.display="none";});
  }
  _mpInitPx();
});
</script>''')


@ui.page("/")
async def main():
    ui.add_head_html(STYLE)
    ui.add_head_html(CURVE_JS)

    with ui.element("div").classes("mp-shell"):
        # Header
        with ui.element("div").classes("mp-header"):
            ui.html('<span style="font-size:18px;font-weight:700;color:var(--text)">MattePro</span>')
            ui.html('<span style="color:var(--border);padding:0 4px">·</span>')
            ui.html('<span style="font-size:12px;color:var(--sub)">Triangulation Matting  ·  Focus Stacking</span>')
            ui.element("div").style("flex:1")
            ui.html(
                '<span id="px-readout" style="font-size:11px;font-family:monospace;'
                'color:var(--sub);letter-spacing:0.04em;white-space:nowrap;display:none"></span>'
            )
            ui.html('<span style="font-size:11px;color:var(--sub)">PiSlider Macro</span>')

        # Left column
        with ui.element("div").classes("mp-left"):
            # Tabs
            with ui.element("div").classes("mp-tabs"):
                matte_tab = ui.html('<button class="mp-tab active" id="tab-matte">MATTE</button>')
                stack_tab = ui.html('<button class="mp-tab" id="tab-stack">STACK</button>')
                batch_tab = ui.html('<button class="mp-tab" id="tab-batch">BATCH</button>')

            with ui.element("div").classes("mp-scroll"):
                with ui.element("div").style("display:block") as matte_wrap:
                    _build_matte_panel()
                with ui.element("div").style("display:none") as stack_wrap:
                    _build_stack_panel()
                with ui.element("div").style("display:none") as batch_wrap:
                    _build_batch_panel()

            def switch_tab(key):
                wraps={"matte":matte_wrap,"stack":stack_wrap,"batch":batch_wrap}
                tabs={"matte":(matte_tab,"tab-matte","MATTE"),
                      "stack":(stack_tab,"tab-stack","STACK"),
                      "batch":(batch_tab,"tab-batch","BATCH")}
                for k,w in wraps.items():
                    w.style("display:block" if k==key else "display:none")
                for k,(el,eid,label) in tabs.items():
                    cls="mp-tab active" if k==key else "mp-tab"
                    el.set_content(f'<button class="{cls}" id="{eid}">{label}</button>')

            matte_tab.on("click", lambda: switch_tab("matte"))
            stack_tab.on("click", lambda: switch_tab("stack"))
            batch_tab.on("click", lambda: switch_tab("batch"))

        # Right: preview
        _build_preview_panel()

    ui.timer(0.08, state.poll)

    # ── Sync browser-cached field values back to server state on page load ──
    # Chrome caches form values between app restarts; the server state starts
    # empty, but the browser may show old values — read them and kick reload.
    async def _sync_fields_on_load():
        import asyncio as _asyncio
        await _asyncio.sleep(0.6)  # let DOM fully paint first
        fields = [
            ("proj_path",      "fi_proj_path"),
            ("cal_black_path", "fi_cal_black_path"),
            ("cal_white_path", "fi_cal_white_path"),
            ("cal_grey_path",  "fi_cal_grey_path"),
        ]
        changed_proj = False
        changed_cal  = False
        for attr, fid in fields:
            val = await ui.run_javascript(
                f'(document.getElementById("{fid}")||{{}}).value||""', timeout=3
            )
            if val and not getattr(state, attr, ""):
                setattr(state, attr, val)
                if attr == "proj_path":
                    changed_proj = True
                else:
                    changed_cal = True
        if changed_proj:
            state.reload_project()
        if changed_cal:
            state.load_cal_images()

    ui.timer(0.0, _sync_fields_on_load, once=True)

    # ── Attach native 'change' listeners + drag-to-pan ───────────────────────
    ui.run_javascript("""
      setTimeout(function(){
        var proj = document.getElementById('fi_proj_path');
        if(proj){
          proj.addEventListener('change', function(){
            var e = new CustomEvent('change', {bubbles:true});
            proj.dispatchEvent(e);
          });
        }
        // NOTE: initCurve is NOT called here. CURVE_JS's own DOMContentLoaded
        // hook owns initialisation and is idempotent; calling it again created a
        // SECOND instance per canvas, each with its own pts array and its own
        // listeners. Both fired on every edit (last one won in Python) and the
        // later instance clobbered the shared _reset/_getlut globals, so Reset
        // only reset one of them — leaving the drawn curve and the applied LUT
        // out of sync.
        // Drag-to-pan is also gone: the zoom controller (window._mpZoom) owns
        // pan via pointer events on a CSS transform. This old scrollLeft-based
        // handler targeted the pre-transform layout and fought the new one.
      }, 400);
    """)


# ── Launch ────────────────────────────────────────────────────────────────────

def _chrome_app_open(url):
    """Open url in Chrome/Edge/Brave app-mode (no browser chrome). Returns Popen on success, None on failure."""
    profile_dir = "/tmp/mattepro-chrome-profile"
    for browser in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]:
        if os.path.exists(browser):
            proc = subprocess.Popen([
                browser,
                f"--user-data-dir={profile_dir}",
                f"--app={url}",
                "--window-size=1280,900",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-session-crashed-bubble",
                "--disable-infobars",
                "--disable-features=TranslateUI",
                "--noerrdialogs",
            ])
            return proc
    return None


if __name__ == "__main__":
    import fcntl, socket, subprocess, threading, os, sys, time as _time
    import multiprocessing; multiprocessing.freeze_support()

    # ── Single-instance lock ──────────────────────────────────────────
    LOCK_PATH = "/tmp/mattepro.lock"
    PORT_PATH = "/tmp/mattepro.port"  # separate file so port survives lock truncation

    # Read existing port from dedicated port file
    _existing_port = None
    try:
        with open(PORT_PATH, "r") as _pf:
            _s = _pf.read().strip()
            _existing_port = int(_s) if _s.isdigit() else None
    except Exception:
        pass

    # Acquire exclusive lock (non-blocking)
    _lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        # Server already running — open a new app-mode window pointed at the existing server
        _url = f"http://127.0.0.1:{_existing_port}" if _existing_port else "http://127.0.0.1:8765"
        try:
            _chrome_app_open(_url)
        except Exception:
            subprocess.Popen(["open", _url])
        sys.exit(0)

    def _find_port(start=8765):
        # Must bind 0.0.0.0 — NiceGUI/uvicorn binds on all interfaces
        for p in range(start, start + 50):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                    s.bind(("0.0.0.0", p)); return p
            except OSError:
                pass
        return start

    PORT = _find_port()
    # Write port to dedicated file so secondary instances always find it
    try:
        with open(PORT_PATH, "w") as _pf:
            _pf.write(str(PORT))
    except Exception:
        pass

    import atexit, shutil

    def _cleanup_lock():
        """Remove lock/port files so the next launch starts fresh."""
        try: os.unlink(LOCK_PATH)
        except Exception: pass
        try: os.unlink(PORT_PATH)
        except Exception: pass

    atexit.register(_cleanup_lock)

    def _open_window():
        _time.sleep(2.0)
        url = f"http://127.0.0.1:{PORT}"
        # Wipe the profile dir so Chrome never sees a crashed previous session
        try:
            shutil.rmtree("/tmp/mattepro-chrome-profile", ignore_errors=True)
        except Exception:
            pass
        proc = _chrome_app_open(url)
        if proc is None:
            subprocess.Popen(["open", url])
            return  # can't track browser process — server runs until killed
        # Block until the Chrome window closes, then exit cleanly
        proc.wait()
        _cleanup_lock()
        os._exit(0)

    threading.Thread(target=_open_window, daemon=False).start()

    # Serve /tmp so the preview PNG can be fetched as a URL (avoids huge base64 in JS)
    app.add_static_files("/preview", "/tmp")

    ui.run(title="MattePro", port=PORT, reload=False, show=False, favicon="🎯")
