#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════
 GENETIC ART SHOW  —  PySide6 · final build
════════════════════════════════════════════════════════════════════════
 PLAYLIST      images fetched from the web + optional captions (persisted)
 GENOME        31 shape modes (triangles … spirals, crescents, polygons)
               + "Mixed" genome (27 kinds per row) — pick in the GUI grid
 COLOR         100% HSV — genes are hue/sat/val/alpha per shape,
               fitness matches in HSV space (circular hue, sat-weighted),
               population seeded from real target pixels
 GA            tournament selection, elitism, row crossover, role-based
               mutation (position/size/angle), live PSNR graph
 GUI           toolbar + shortcuts · playlist filter · tabbed settings ·
               shape-picker grid with live previews · 7 presets
 MP4           threaded recorder: title cards, crossfades, keyframe-
               interpolated time-lapse, burn-in overlay, fade in/out,
               outro, 480p→1080p · 24/30/60 fps · quality presets ·
               background music muxed into the file (AAC)
 MUSIC         procedurally generated ambient loop (numpy → wav), looped
 OUTPUTS       everything in the current working directory:
               ga_show_*.mp4 · ambient.wav · playlist.json
               ga_best_*.png · ga_art_show.log (rewritten each launch)

 Install:  pip install PySide6 numpy pillow requests imageio imageio-ffmpeg
════════════════════════════════════════════════════════════════════════
"""

import io, json, logging, math, os, queue, random, shutil, subprocess, sys, time, wave
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import Qt, QTimer, QUrl, QThread, Signal, QSize
from PySide6.QtGui import (QAction, QBrush, QColor, QCloseEvent, QIcon, QImage,
                           QKeySequence, QPainter, QPen, QPointF, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QSlider, QSpinBox, QSplitter, QStyle, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    HAS_MEDIA = True
except Exception:
    HAS_MEDIA = False

APP_TITLE = "Genetic Art Show"
OUTPUT_DIR = Path.cwd()                       # all outputs land in the cwd
LOG_FILE = OUTPUT_DIR / "ga_art_show.log"     # reopened with mode="w" every launch

DEMO_IMAGES = [
    ("https://picsum.photos/id/237/512/384",
     "Man's best friend — 80 translucent triangles mutating toward pixel perfection."),
    ("https://picsum.photos/id/1015/512/384",
     "A river runs through it. Watch pure randomness become a landscape."),
    ("https://picsum.photos/id/1025/512/384",
     "Pug life. Survival of the flattest."),
    ("https://picsum.photos/id/1074/512/384",
     "King of the jungle, bred by mutation and selection."),
    ("https://picsum.photos/seed/ga-art/512/384",
     "A surprise picture, pulled straight from the web."),
]


def setup_logging():
    """Console + logfile (mode='w' → truncated fresh on every run)."""
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"))
    except OSError as e:
        print(f"! logfile disabled ({e}) — console logging only")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(threadName)-11s | %(message)s",
        datefmt="%H:%M:%S", handlers=handlers)
    return logging.getLogger("ga_art_show")


log = setup_logging()


# ═══════════════════════════ small helpers ═══════════════════════════════

_FONTS = {}
def get_font(size):
    size = int(size)
    if size in _FONTS:
        return _FONTS[size]
    f = None
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "segoeui.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            f = ImageFont.truetype(name, size); break
        except Exception:
            pass
    if f is None:
        f = ImageFont.load_default()
    _FONTS[size] = f
    return f


def fetch_image(url, timeout=15):
    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0 (GeneticArtShow)"})
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)); im.load()
    return im.convert("RGB")


def fit_even(im, max_side):
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    w2 = max(2, int(w * s) & ~1); h2 = max(2, int(h * s) & ~1)
    return im.resize((w2, h2), Image.LANCZOS)


def letterbox(im, size=(640, 480), bg=(10, 11, 18)):
    """Center a picture on a fixed-size dark canvas (uniform video frames)."""
    cw, ch = size
    s = min(cw / im.width, ch / im.height)
    w2 = max(2, int(im.width * s) & ~1); h2 = max(2, int(im.height * s) & ~1)
    im2 = im.resize((w2, h2), Image.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(im2, ((cw - w2) // 2, (ch - h2) // 2))
    return canvas


def pil_to_qimage(im):
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    return QImage(data, im.width, im.height, im.width * 4,
                  QImage.Format_RGBA8888).copy()


def wrap_text(draw, text, font, maxw):
    words = text.split(); lines = []; cur = ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if draw.textlength(t, font=font) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines


def make_title_card(text, sub, size=(640, 480)):
    """Title card with soft GA-triangle decoration; typography scales to canvas."""
    w, h = size
    grad = np.zeros((h, w, 3), np.float32)
    top, bot = (16, 18, 30), (34, 28, 52)
    for c in range(3):
        grad[..., c] = np.linspace(top[c], bot[c], h, dtype=np.float32)[:, None]
    img = Image.fromarray(grad.astype(np.uint8), "RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    rng = random.Random(hash(text) & 0xffff)
    for _ in range(16):
        pts = [(rng.uniform(-40, w + 40), rng.uniform(-40, h + 40)) for _ in range(3)]
        dr.polygon(pts, fill=(rng.randint(50, 120), rng.randint(60, 130),
                              rng.randint(120, 210), rng.randint(12, 30)))
    d = ImageDraw.Draw(img)
    f1, f2 = get_font(int(h * 0.085)), get_font(int(h * 0.047))
    lines = wrap_text(d, text.strip() or "Untitled", f1, int(w * 0.78))[:4]
    lh = int(h * 0.10)
    y = (h - lh * len(lines)) // 2 - int(h * 0.05)
    for ln in lines:
        tw = d.textlength(ln, font=f1)
        d.text(((w - tw) / 2, y), ln, font=f1, fill=(238, 241, 248)); y += lh
    if sub:
        s = sub.strip(); tw = d.textlength(s, font=f2)
        d.text(((w - tw) / 2, h - int(h * 0.12)), s, font=f2, fill=(148, 155, 176))
    return img


def make_music(path, seconds=48, sr=22050, seed=11):
    """Soft ambient chord-pad loop with sparse pentatonic plucks → 16-bit stereo WAV."""
    rng = random.Random(seed)
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n)
    chords = [(110.00, (220.00, 261.63, 329.63, 440.00)),   # Am
              ( 87.31, (174.61, 220.00, 261.63, 349.23)),   # F
              ( 65.41, (196.00, 261.63, 329.63, 392.00)),   # C
              ( 98.00, (196.00, 246.94, 293.66, 392.00))]   # G
    seg = seconds / len(chords)
    for ci, (bass, notes) in enumerate(chords):
        s0, s1 = int(ci * seg * sr), int(min(n, (ci + 1) * seg * sr))
        L = s1 - s0; env = np.ones(L); f = int(1.2 * sr)
        env[:f] = np.linspace(0, 1, f) ** 1.5
        env[-f:] = np.linspace(1, 0, f) ** 1.5
        st = t[s0:s1]; pad = np.zeros(L)
        for fr in notes:
            pad += np.sin(2 * np.pi * fr * st + rng.uniform(0, 6.28))
            pad += 0.4 * np.sin(2 * np.pi * fr * 1.005 * st + rng.uniform(0, 6.28))
        pad += 1.6 * np.sin(2 * np.pi * bass * st)
        out[s0:s1] += pad * env * 0.05
    out *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.1 * t)
    scale = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33]
    for _ in range(int(seconds * 1.1)):
        start = rng.uniform(0, seconds - 2.0); fr = rng.choice(scale)
        L = int(1.6 * sr); s0 = int(start * sr)
        envl = np.exp(-np.linspace(0, 7, L))
        out[s0:s0 + L] += 0.10 * envl * np.sin(2 * np.pi * fr * np.arange(L) / sr)
        out[s0:s0 + L] += 0.05 * envl * np.sin(2 * np.pi * fr * 2 * np.arange(L) / sr)
    out = np.tanh(out * 1.2)
    out /= np.max(np.abs(out)) + 1e-9
    d = int(0.012 * sr)
    right = np.concatenate([np.zeros(d), out[:-d]])
    pcm = (np.stack([out, right], 1) * 32767 * 0.85).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


# ═══════════════════════════ HSV color core ═══════════════════════════════

def hsv_to_rgb_batch(h, s, v):
    """Vectorized HSV→RGB. h in [0,1) (fraction of the wheel), s,v in [0,1] → uint8 RGB."""
    h6 = np.asarray(h, np.float64) * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    sel = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(sel, [v, q, p, p, t, v])
    g = np.select(sel, [t, v, v, q, p, p])
    b = np.select(sel, [p, p, t, v, v, q])
    return ((r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8))


def hsv_view(pil):
    """Display variant: H,S,V reinterpreted as R,G,B — the classic HSV false-color look."""
    return Image.frombytes("RGB", pil.size, pil.convert("HSV").tobytes())


def make_hsv_strip(w=320, h=45):
    """Legend for the Genome tab: hue wheel (sat 1.0 / 0.45) + value ramp."""
    arr = np.zeros((h, w, 3), np.uint8)
    xs = np.arange(w) / w
    third = h // 3
    arr[:third] = np.stack(hsv_to_rgb_batch(xs, np.ones(w), np.ones(w)), -1)
    arr[third:2 * third] = np.stack(hsv_to_rgb_batch(xs, np.full(w, 0.45), np.ones(w)), -1)
    g = (xs * 255).astype(np.uint8)
    arr[2 * third:] = np.stack([g, g, g], -1)
    return Image.fromarray(arr, "RGB")


# ═══════════════════════ genome architecture ══════════════════════════════
# one individual = list of shapes; each row = [geometry genes] + [h, s, v, alpha]
# roles:  p = position (0..1)   s = size (0.003..1.5)   a = angle (turns, wraps)

SHAPE_MENU = [
    ("🔺 Triangles", "triangle"), ("🟦 Squares", "square"), ("▭ Rectangles", "rectangle"),
    ("● Circles", "circle"), ("🥚 Ellipses", "ellipse"),
    ("✏ Strokes", "stroke"), ("➰ Beziers", "bezier"), ("◎ Rings", "ring"),
    ("◔ Wedges", "wedge"), ("🧀 Chords", "chord"), ("🥐 Arcs", "arc"),
    ("⭐ Stars ×5", "star:5"), ("⭐ Stars ×6", "star:6"), ("⭐ Stars ×8", "star:8"),
    ("🔶 Diamonds", "diamond"), ("💊 Capsules", "capsule"), ("➕ Crosses", "cross"),
    ("❤ Hearts", "heart"), ("⬛ Squircles", "superellipse"), ("💧 Teardrops", "teardrop"),
    ("☁ Blobs", "blob"), ("🌊 Waves", "wave"), ("🌀 Spirals", "spiral"),
    ("🌙 Crescents", "crescent"),
    ("🔷 Polygons ×4", "polygon:4"), ("🔷 Polygons ×5", "polygon:5"),
    ("🔷 Polygons ×6", "polygon:6"), ("🔷 Polygons ×8", "polygon:8"),
    ("🔷 Polygons ×10", "polygon:10"), ("🔷 Polygons ×12", "polygon:12"),
    ("🧬 Mixed — everything", "mixed"),
]


def gene_roles(kind):
    if kind.startswith("polygon:"):
        return "p" * (2 * int(kind.split(":")[1]))
    if kind.startswith("star:"):
        return "ppsaa"                                   # cx, cy, r_out, r_in, rot
    return {
        "triangle": "pppppp", "square": "pps", "rectangle": "ppss",
        "circle": "pps", "ellipse": "ppss",
        "stroke": "pppps",                               # x1,y1,x2,y2,width
        "bezier": "pppppps",                             # x0,y0,cx,cy,x1,y1,width
        "ring": "ppss",                                  # cx,cy,r,stroke width
        "wedge": "ppsaa", "chord": "ppsaa",              # cx,cy,r,a0,a1
        "arc": "ppsaas",                                 # cx,cy,r,a0,a1,width
        "diamond": "ppss", "capsule": "pppps", "cross": "ppss",
        "heart": "ppsa", "teardrop": "ppsa",             # cx,cy,r,rot
        "superellipse": "ppss",                          # cx,cy,r,exponent
        "blob": "ppss",                                  # cx,cy,r,wobble
        "wave": "ppppsss",                               # x0,y0,x1,y1,amp,periods,width
        "spiral": "ppss",                                # cx,cy,r,turns
        "crescent": "ppssa",                             # cx,cy,r,offset,rot
    }[kind]


def shape_coords(kind):
    return len(gene_roles(kind))


_ALL_KINDS = [k for _, k in SHAPE_MENU if k != "mixed"]
GEO_MAX = 12                                           # widest mixed row (Polygon ×6)
MIXED_KINDS = [k for k in _ALL_KINDS if shape_coords(k) <= GEO_MAX]   # 27 kinds


def color_offset(shape):
    """Index of the first color gene (h) in a row."""
    return 1 + GEO_MAX if shape == "mixed" else shape_coords(shape)


def genes_per_shape(shape):
    return color_offset(shape) + 4                     # h, s, v + alpha


# ═══════════════════════════════ rendering ════════════════════════════════

def _draw_shape(d, kind, geo, fill, w, h):
    """Draw one shape; geo = its geometry genes (sliced from the genome row)."""
    m = min(w, h)
    if kind == "triangle" or kind.startswith("polygon:"):
        d.polygon([(geo[i] * w, geo[i + 1] * h) for i in range(0, len(geo), 2)], fill=fill)
    elif kind == "rectangle":
        x0, y0 = geo[0] * w, geo[1] * h
        d.rectangle([x0, y0, x0 + geo[2] * w, y0 + geo[3] * h], fill=fill)
    elif kind == "square":
        x0, y0, s = geo[0] * w, geo[1] * h, geo[2] * m
        d.rectangle([x0, y0, x0 + s, y0 + s], fill=fill)
    elif kind == "ellipse":
        cx, cy = geo[0] * w, geo[1] * h
        d.ellipse([cx - geo[2] * w, cy - geo[3] * h,
                   cx + geo[2] * w, cy + geo[3] * h], fill=fill)
    elif kind == "circle":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif kind == "stroke":
        d.line([(geo[0] * w, geo[1] * h), (geo[2] * w, geo[3] * h)],
               fill=fill, width=max(1, int(geo[4] * m / 6)))
    elif kind == "bezier":
        x0, y0, cx, cy, x1, y1 = (geo[i] * (w if i % 2 == 0 else h) for i in range(6))
        pts = []
        for s_ in range(13):
            t = s_ / 12; u = 1.0 - t
            pts.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                        u * u * y0 + 2 * u * t * cy + t * t * y1))
        d.line(pts, fill=fill, width=max(1, int(geo[6] * m / 6)), joint="curve")
    elif kind == "ring":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        wpix = max(1, min(int(geo[3] * m / 6), max(1, int(r))))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=wpix)
    elif kind == "wedge":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        d.pieslice([cx - r, cy - r, cx + r, cy + r],
                   geo[3] * 360, geo[4] * 360, fill=fill)
    elif kind == "chord":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        d.chord([cx - r, cy - r, cx + r, cy + r], geo[3] * 360, geo[4] * 360, fill=fill)
    elif kind == "arc":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        d.arc([cx - r, cy - r, cx + r, cy + r], geo[3] * 360, geo[4] * 360,
              fill=fill, width=max(1, int(geo[5] * m / 6)))
    elif kind.startswith("star:"):
        npts = int(kind.split(":")[1])
        cx, cy = geo[0] * w, geo[1] * h
        ro, ri, rot = geo[2] * m, geo[3] * m, geo[4] * 2 * math.pi
        d.polygon([(cx + (ro if i % 2 == 0 else ri) * math.cos(rot + math.pi * i / npts),
                    cy + (ro if i % 2 == 0 else ri) * math.sin(rot + math.pi * i / npts))
                   for i in range(npts * 2)], fill=fill)
    elif kind == "diamond":
        cx, cy, rx, ry = geo[0] * w, geo[1] * h, geo[2] * w, geo[3] * h
        d.polygon([(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)], fill=fill)
    elif kind == "capsule":
        x0, x1 = sorted((geo[0] * w, geo[2] * w))
        y0, y1 = sorted((geo[1] * h, geo[3] * h))
        rad = max(0, min(geo[4] * m / 2, (x1 - x0) / 2, (y1 - y0) / 2))
        try:
            d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=fill)
        except AttributeError:                         # Pillow < 8.2 → plain rectangle
            d.rectangle([x0, y0, x1, y1], fill=fill)
    elif kind == "cross":                              # single 12-pt polygon → no double-blend
        cx, cy = geo[0] * w, geo[1] * h
        a = geo[2] * m; t = min(geo[3] * m, a * 0.9)
        d.polygon([(cx - t, cy - a), (cx + t, cy - a), (cx + t, cy - t), (cx + a, cy - t),
                   (cx + a, cy + t), (cx + t, cy + t), (cx + t, cy + a), (cx - t, cy + a),
                   (cx - t, cy + t), (cx - a, cy + t), (cx - a, cy - t), (cx - t, cy - t)],
                  fill=fill)
    elif kind == "heart":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m / 17.0
        rot = geo[3] * 2 * math.pi; cr, sr = math.cos(rot), math.sin(rot)
        pts = []
        for i in range(48):
            t = 2 * math.pi * i / 48
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + r * (x * cr - y * sr), cy - r * (x * sr + y * cr)))
        d.polygon(pts, fill=fill)
    elif kind == "superellipse":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        ne = 2.0 / (0.3 + max(0.02, geo[3]) * 4.0)
        pts = []
        for i in range(72):
            t = 2 * math.pi * i / 72; c, s = math.cos(t), math.sin(t)
            pts.append((cx + r * math.copysign(abs(c) ** ne, c),
                        cy + r * math.copysign(abs(s) ** ne, s)))
        d.polygon(pts, fill=fill)
    elif kind == "teardrop":                           # cardioid: cusp + round lobe
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        rot = geo[3] * 2 * math.pi
        tx, ty = cx - r * math.cos(rot), cy - r * math.sin(rot)
        pts = [(tx + r * (1 + math.cos(t)) * math.cos(rot + t),
                ty + r * (1 + math.cos(t)) * math.sin(rot + t))
               for t in (2 * math.pi * i / 44 for i in range(44))]
        d.polygon(pts, fill=fill)
    elif kind == "blob":                               # deterministic wobble (same genome → same blob)
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        wob = min(0.9, geo[3] * 2.0)
        pts = []
        for i in range(14):
            ang = 2 * math.pi * i / 14
            h01 = abs(math.sin(i * 127.1) * 43758.5453) % 1.0
            rr = r * (1.0 - wob / 2 + wob * h01)
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=fill)
    elif kind == "wave":
        x0, y0, x1, y1 = geo[0] * w, geo[1] * h, geo[2] * w, geo[3] * h
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
        if L >= 3:
            amp, per = geo[4] * m * 0.5, max(0.5, geo[5] * 10.0)
            pts = [(x0 + dx * t - dy / L * amp * math.sin(t * per * 2 * math.pi),
                    y0 + dy * t + dx / L * amp * math.sin(t * per * 2 * math.pi))
                   for t in (i / 60.0 for i in range(61))]
            d.line(pts, fill=fill, width=max(1, int(geo[6] * m / 6)), joint="curve")
    elif kind == "spiral":
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        turns = max(0.25, geo[3] * 4.0); steps = int(70 * turns) + 14
        pts = [(cx + r * t * math.cos(t * turns * 2 * math.pi),
                cy + r * t * math.sin(t * turns * 2 * math.pi))
               for t in (i / steps for i in range(steps + 1))]
        d.line(pts, fill=fill, width=max(1, int(r * 0.12)), joint="curve")
    elif kind == "crescent":                           # true moon: circle1 − circle2
        cx, cy, r = geo[0] * w, geo[1] * h, geo[2] * m
        rot = geo[4] * 2 * math.pi
        dist = r * (0.10 + 1.70 * min(1.0, max(0.0, geo[3])))
        c2x, c2y = cx + dist * math.cos(rot), cy + dist * math.sin(rot)
        phi0 = math.acos(min(1.0, dist / (2 * r)))
        psi0 = math.acos(max(-1.0, -dist / (2 * r)))
        pts = [(cx + r * math.cos(rot + phi0 + (2 * math.pi - 2 * phi0) * i / 39),
                cy + r * math.sin(rot + phi0 + (2 * math.pi - 2 * phi0) * i / 39))
               for i in range(40)]
        pts += [(c2x + r * math.cos(rot + psi0 + (2 * math.pi - 2 * psi0) * i / 39),
                 c2y + r * math.sin(rot + psi0 + (2 * math.pi - 2 * psi0) * i / 39))
                for i in range(40)]
        d.polygon(pts, fill=fill)


def render_genome(g, w, h, bg, shape="triangle"):
    """g: (n, genes) — colors are HSV genes (h,s,v ∈ [0,1]) + alpha, batch-converted."""
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img, "RGBA")
    nc = color_offset(shape)
    R, G, B = hsv_to_rgb_batch(g[:, nc], g[:, nc + 1], g[:, nc + 2])
    for idx, row in enumerate(g):
        fill = (int(R[idx]), int(G[idx]), int(B[idx]), int(row[nc + 3]))
        if shape == "mixed":
            kind = MIXED_KINDS[int(row[0]) % len(MIXED_KINDS)]
            _draw_shape(d, kind, row[1:1 + shape_coords(kind)], fill, w, h)
        else:
            _draw_shape(d, shape, row[:nc], fill, w, h)
    return img


# ═══════════════════════ random genome rows ═══════════════════════════════

def rand_geometry(kind, rng):
    return [rng.uniform(0.02, 0.45) if r == "s" else rng.random()
            for r in gene_roles(kind)]


def rand_shape_row(shape, rng, seeds=None):
    if shape == "mixed":
        kind = rng.choice(MIXED_KINDS)
        row = [MIXED_KINDS.index(kind)] + rand_geometry(kind, rng)
        row += [0.0] * (1 + GEO_MAX - len(row))
    else:
        row = rand_geometry(shape, rng)
    if seeds and rng.random() < 0.55:                  # born from a real target pixel
        h, s, v = seeds[rng.randrange(len(seeds))]
        row += [float(h) % 1.0,
                min(1.0, max(0.0, float(s) + rng.gauss(0, 0.12))),
                min(1.0, max(0.0, float(v) + rng.gauss(0, 0.12)))]
    else:                                              # full HSV cylinder — every color reachable
        row += [rng.random(), rng.random(), rng.random()]
    row.append(rng.uniform(25, 90))                    # alpha
    return row


def make_shape_preview(shape, w=300, h=180):
    """16 random individuals of a shape — shown live in the Genome tab."""
    rng = random.Random(20240101)
    g = np.array([rand_shape_row(shape, rng, None) for _ in range(16)], np.float32)
    return render_genome(g, w, h, (24, 26, 38), shape)


# ═══════════════════════════════ MP4 recorder ═════════════════════════════

class Mp4Recorder(QThread):
    """Assembles the show MP4 off the GUI thread: title cards, crossfades,
    keyframe-interpolated time-lapse, burn-in overlays, outro, audio mux."""
    recStatus = Signal(str)
    recProgress = Signal(int, int)          # frames written, estimated total
    recDone = Signal(str, int, bool)        # path, frames, audio muxed

    def __init__(self, path, est_total, fps, size, quality, burn_in=True, audio_path=None):
        super().__init__()
        self.path = Path(path); self.fps = fps; self.size = size
        self.quality = quality; self.burn_in = burn_in
        self.audio_path = audio_path
        self._est = max(1, est_total)
        self._frames = 0; self._muxed = False
        self._first_image = True
        self._last = None                   # (t, arr, gen, gens, psnr)
        self._seg = 0.0
        self.q = queue.Queue()
        self._writer = None; self._mode = None

    # ── public API (thread-safe; just enqueues commands) ──
    def start_image(self, title, sub, target_arr, seg_secs):
        self.q.put(("image", title, sub, target_arr, float(seg_secs)))

    def keyframe(self, arr, gen, gens, psnr):
        self.q.put(("kf", arr, int(gen), int(gens), float(psnr)))

    def end_image(self, hold=1.0):
        self.q.put(("end", float(hold)))

    def outro(self, sub):
        self.q.put(("outro", sub))

    def finish(self):
        self.q.put(None)

    # ── encoder ──
    def _open_writer(self):
        errs = []
        try:
            import imageio
            self._writer = imageio.get_writer(
                str(self.path), fps=self.fps, codec="libx264",
                quality={"high": 9, "medium": 7, "low": 5}[self.quality],
                macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
            self._mode = "imageio"; return
        except Exception as e:
            errs.append(f"imageio: {e}")
        try:
            import cv2
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.fps, self.size)
            if not self._writer.isOpened():
                raise RuntimeError("writer failed to open")
            self._mode = "cv2"; return
        except Exception as e:
            errs.append(f"opencv: {e}")
        raise RuntimeError("No MP4 encoder available. " + " | ".join(errs))

    def _write(self, arr):
        if self._mode == "imageio":
            self._writer.append_data(arr)
        else:
            self._writer.write(np.ascontiguousarray(arr[:, :, ::-1]))
        self._frames += 1
        if self._frames % 30 == 0:
            self.recProgress.emit(self._frames, self._est)

    @staticmethod
    def _blend(a, b, alpha):
        return (a.astype(np.float32) * (1.0 - alpha)
                + b.astype(np.float32) * alpha).astype(np.uint8)

    def _n(self, secs):
        return max(1, int(round(secs * self.fps)))

    def _overlay(self, arr, gen, gens, psnr, t):
        if not self.burn_in:
            return arr
        img = Image.fromarray(arr); d = ImageDraw.Draw(img)
        W, H = self.size
        f = get_font(max(13, H // 40))
        if gen is not None:
            d.text((10, H - int(H * 0.055)), f"gen {gen}/{gens}", font=f, fill=(235, 238, 245))
        if psnr is not None:
            s = f"HSV-fit {psnr:.1f} dB"
            d.text((W - d.textlength(s, font=f) - 10, H - int(H * 0.055)),
                   s, font=f, fill=(235, 238, 245))
        if self._seg:
            d.rectangle([0, H - 3, int(W * min(1.0, t / self._seg)), H], fill=(58, 112, 234))
        return np.asarray(img)

    # ── assembly loop ──
    def run(self):
        try:
            self._open_writer()
        except Exception as e:
            self.recStatus.emit(f"Recorder failed: {e}")
            self.recDone.emit(str(self.path), 0, False)
            return
        self.recStatus.emit(f"Recording {self.size[0]}×{self.size[1]}@{self.fps} → {self.path.name}")

        while True:
            item = self.q.get()
            if item is None:
                break
            cmd = item[0]

            if cmd == "image":
                _, title, sub, tgt, seg = item
                self._seg = seg
                card = np.asarray(make_title_card(title, sub, self.size), np.uint8)
                n = self._n(2.0)                                   # title card
                for i in range(n):
                    arr = card
                    if self._first_image and i < self._n(0.5):     # fade-in from black
                        arr = self._blend(np.zeros_like(card), card, (i + 1) / self._n(0.5))
                    self._write(arr)
                self._first_image = False
                self._cross(card, tgt, 0.4)                        # crossfade to target
                for _ in range(self._n(0.6)):                      # target reveal hold
                    self._write(tgt)
                self._last = (0.0, tgt, 0, 1, 0.0)

            elif cmd == "kf":
                _, arr, gen, gens, psnr = item
                if self._last is None:
                    continue
                t_prev, arr_prev, g_prev, _, _ = self._last
                t_new = (gen / max(1, gens)) * self._seg
                steps = max(1, int(round((t_new - t_prev) * self.fps)))
                for i in range(1, steps + 1):
                    a = i / steps
                    fr = self._blend(arr_prev, arr, a)
                    gi = int(round(g_prev + (gen - g_prev) * a))
                    self._write(self._overlay(fr, gi, gens, psnr,
                                              t_prev + (t_new - t_prev) * a))
                self._last = (t_new, arr, gen, gens, psnr)

            elif cmd == "end":
                if self._last is None:
                    continue
                hold = item[1]
                t_prev, arr, gen, gens, psnr = self._last
                while t_prev < self._seg:                          # pad early-finished images
                    t_prev = min(self._seg, t_prev + 1.0 / self.fps)
                    self._write(self._overlay(arr, gen, gens, psnr, t_prev))
                for _ in range(self._n(hold)):
                    self._write(self._overlay(arr, gen, gens, psnr, self._seg))
                self._last = None

            elif cmd == "outro":
                card = np.asarray(make_title_card("Genetic Art Show", item[1], self.size), np.uint8)
                n, nf = self._n(2.6), self._n(0.6)
                for i in range(n):                                 # hold, then fade to black
                    arr = card
                    if i >= n - nf:
                        arr = self._blend(card, np.zeros_like(card), (i - (n - nf)) / nf)
                    self._write(arr)

        try:
            self._writer.close() if self._mode == "imageio" else self._writer.release()
        except Exception:
            pass
        if self._frames and self.audio_path and Path(self.audio_path).exists():
            self._muxed = self._mux_audio(self.path, Path(self.audio_path))
        self.recDone.emit(str(self.path), self._frames, self._muxed)

    def _cross(self, a, b, secs):
        n = self._n(secs)
        for i in range(1, n + 1):
            self._write(self._blend(a, b, i / n))

    def _mux_audio(self, video, wav):
        exe = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = shutil.which("ffmpeg")
        if not exe:
            self.recStatus.emit("ffmpeg not found — MP4 saved without audio")
            return False
        tmp = video.with_suffix(".mux.mp4")
        try:
            r = subprocess.run(
                [exe, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "160k", "-shortest", str(tmp)],
                capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and tmp.exists():
                os.replace(tmp, video)
                return True
            self.recStatus.emit(f"audio mux failed: {r.stderr.strip()[:200]}")
        except Exception as e:
            self.recStatus.emit(f"audio mux failed: {e}")
        return False


# ═══════════════════════════ background fetcher ═══════════════════════════

class Fetcher(QThread):
    loaded = Signal(str, bool, object, str)        # url, ok, PIL image | None, error

    def __init__(self):
        super().__init__()
        self.q = []; self.alive = True

    def add(self, url):
        self.q.append(url)

    def run(self):
        while self.alive:
            try:
                url = self.q.pop(0)
            except IndexError:
                time.sleep(0.1); continue
            try:
                img = fit_even(fetch_image(url), 512)
                log.info("fetched %s  (%dx%d)", url, img.width, img.height)
                self.loaded.emit(url, True, img, "")
            except Exception as e:
                log.warning("fetch FAILED %s : %s", url, e)
                self.loaded.emit(url, False, None, str(e))

    def stop(self):
        self.alive = False


# ═══════════════════════════ GA evolution worker ══════════════════════════

class EvolutionWorker(QThread):
    uiFrame = Signal(QImage, object)                        # QImage + PIL (for HSV view)
    videoFrame = Signal(np.ndarray, int, int, float)        # arr, gen, gens, fit
    progressed = Signal(int, int, float, float)             # gen, total, fit, gen/sec
    message = Signal(str)
    finalImage = Signal(QImage)
    finishedImage = Signal(bool)

    def __init__(self, target_pil, params):
        super().__init__()
        self.target = target_pil
        self.p = params
        self.paused = False
        self._stop = False
        self._skip = False

    def stop(self): self._stop = True; self.paused = False; log.info("worker: stop requested")
    def skip(self): self._skip = True; self.paused = False; log.info("worker: skip requested")

    def _evolve(self):
        rng = random.Random()
        npr = np.random.default_rng()
        tw, th = self.target.size
        bg = tuple(int(round(c)) for c in
                   np.asarray(self.target, np.float64).reshape(-1, 3).mean(0))

        # ── HSV fitness target planes + pixel seeds ──
        hsv_t = np.asarray(self.target.convert("HSV"), np.float32) / 255.0
        tgt_h, tgt_s, tgt_v = hsv_t[..., 0], hsv_t[..., 1], hsv_t[..., 2]
        px_count = tgt_h.size
        pick = npr.choice(px_count, size=min(96, px_count), replace=False)
        seeds = list(zip(tgt_h.ravel()[pick].tolist(),
                         tgt_s.ravel()[pick].tolist(),
                         tgt_v.ravel()[pick].tolist()))

        n = int(self.p["triangles"])
        pop_n = max(4, int(self.p["population"]))
        gens = int(self.p["generations"])
        rate = float(self.p["mutation"])
        goal = float(self.p["psnr_target"])
        shape = self.p["shape"]
        nc = color_offset(shape)
        K = genes_per_shape(shape)
        ui_every = max(1, gens // 500)
        kf_every = max(1, gens // 240)
        elite_k = 2 if pop_n >= 6 else 1

        log.info("GA start: %s | HSV fitness — circular hue, sat-weighted | "
                 "target %dx%d | count=%d pop=%d gens=%d mut=%.2f goal=%s | %d pixel seeds",
                 shape, tw, th, n, pop_n, gens, rate,
                 f"{goal:.0f}" if goal > 0 else "off", len(seeds))

        def fitness(g):
            arr = np.asarray(render_genome(g, tw, th, bg, shape), np.uint8)
            hsv = np.asarray(Image.fromarray(arr).convert("HSV"), np.float32) / 255.0
            rh, rs, rv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            dh = np.abs(rh - tgt_h)
            np.minimum(dh, 1.0 - dh, out=dh)           # circular hue distance ∈ [0, 0.5]
            hw = np.minimum(rs, tgt_s)                 # hue matters only where BOTH are saturated
            return float((((dh * hw * 2.0) ** 2
                           + (rs - tgt_s) ** 2
                           + (rv - tgt_v) ** 2) / 3.0).mean())

        def psnr_of(mse):
            return 99.0 if mse <= 1e-9 else -10.0 * math.log10(mse)

        rand_shape = lambda: rand_shape_row(shape, rng, seeds)

        def mutate(g):
            g = g.copy()
            ncol = K - nc                              # always 4: h, s, v, alpha
            for i in np.nonzero(npr.random(n) < rate)[0]:
                kind, geo0 = shape, 0
                if shape == "mixed":
                    kind = MIXED_KINDS[int(g[i, 0]) % len(MIXED_KINDS)]
                    geo0 = 1
                roles = gene_roles(kind)
                nk = len(roles)

                if rng.random() < ncol / (nk + ncol):            # ── color / alpha gene
                    k = rng.randrange(nc, K)
                    if k == K - 1:                               # alpha
                        g[i, k] = min(110.0, max(10.0, g[i, k] + rng.gauss(0, 10)))
                    elif k == nc:                                # hue — circular, wraps 1.0→0.0
                        step = 0.25 if rng.random() < 0.10 else 0.06
                        g[i, k] = (g[i, k] + rng.gauss(0, step)) % 1.0
                    else:                                        # saturation / value
                        step = 0.30 if rng.random() < 0.10 else 0.08
                        g[i, k] = min(1.0, max(0.0, g[i, k] + rng.gauss(0, step)))
                else:                                            # ── geometry gene
                    if shape == "mixed" and rng.random() < 1.0 / (nk + 1):
                        g[i, :nc] = rand_shape_row(shape, rng, seeds)[:nc]   # kind change
                        continue
                    j = geo0 + rng.randrange(nk)
                    role = roles[j - geo0]
                    if role == "a":                              # angle — wraps
                        g[i, j] = (g[i, j] + rng.gauss(0, 0.08)) % 1.0
                    elif role == "s":                            # size
                        g[i, j] = min(1.5, max(0.003, g[i, j] + rng.gauss(0, 0.05)))
                    else:                                        # position
                        step = 0.15 if rng.random() < 0.10 else 0.04
                        g[i, j] = min(1.0, max(0.0, g[i, j] + rng.gauss(0, step)))

            r = rng.random()
            if r < 0.10:
                g[rng.randrange(n)] = rand_shape()
            elif r < 0.14:
                i, j = rng.randrange(n), rng.randrange(n)
                g[[i, j]] = g[[j, i]]
            return g

        def tourney():
            best = None
            for _ in range(3):
                c = pop[rng.randrange(len(pop))]
                if best is None or c[0] < best[0]:
                    best = c
            return best

        pop = sorted(((fitness(g), g) for g in (rand_shape() for _ in range(pop_n))),
                     key=lambda x: x[0])

        def emit_frames(g, force=False):
            img = render_genome(pop[0][1], tw, th, bg, shape)
            self.uiFrame.emit(pil_to_qimage(img), img)
            if force or g % kf_every == 0:
                self.videoFrame.emit(np.asarray(img, np.uint8), g, gens, psnr_of(pop[0][0]))

        emit_frames(0, force=True)
        self.progressed.emit(0, gens, psnr_of(pop[0][0]), 0.0)
        t0 = time.perf_counter()

        gen = 0
        for gen in range(1, gens + 1):
            while self.paused and not self._stop and not self._skip:
                time.sleep(0.08)
            if self._stop:
                return False
            if self._skip:
                break

            new = pop[:elite_k]                              # elitism
            while len(new) < pop_n:
                a, b = tourney(), tourney()
                mask = (npr.random(n) < 0.5)[:, None]        # row-wise crossover
                child = np.where(mask, a[1], b[1])
                child = mutate(child)
                new.append((fitness(child), child))
            new.sort(key=lambda x: x[0])
            pop = new

            fit = psnr_of(pop[0][0])
            if gen % ui_every == 0:
                emit_frames(gen)
            self.progressed.emit(gen, gens, fit,
                                 gen / max(1e-9, time.perf_counter() - t0))
            if gen % 50 == 0 or gen == gens:
                log.info("gen %5d/%d | hsv-fit %6.2f | mse %.5f", gen, gens, fit, pop[0][0])
            if goal > 0 and fit >= goal:
                self.message.emit(f"HSV-fit goal {goal:.0f} reached at generation {gen}.")
                break

        emit_frames(gen, force=True)                         # final keyframe at actual gen
        self.progressed.emit(gen, gens, psnr_of(pop[0][0]), 0.0)
        log.info("GA finished: hsv-fit %.2f after %d gens", psnr_of(pop[0][0]), gen)

        s = 1024.0 / max(tw, th)                             # hi-res render for PNG export
        hi = render_genome(pop[0][1], max(2, int(tw * s) & ~1),
                           max(2, int(th * s) & ~1), bg, shape)
        self.finalImage.emit(pil_to_qimage(hi))
        return True

    def run(self):
        try:
            ok = self._evolve()
        except Exception as e:
            log.exception("evolution crashed")               # full traceback → logfile
            self.message.emit(f"Evolution error: {e}")
            ok = False
        self.finishedImage.emit(bool(ok and not self._stop))


# ═══════════════════════ GUI helpers: graph + presets ═════════════════════

class PsnrGraph(QWidget):
    """Live sparkline of HSV-fit vs. generation."""
    def __init__(self):
        super().__init__()
        self.hist = []
        self.setMinimumHeight(88)

    def clear(self):
        self.hist.clear(); self.update()

    def add(self, fit):
        self.hist.append(fit)
        if len(self.hist) > 2400:
            self.hist = self.hist[::2]
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0e1016"))
        p.setPen(QPen(QColor("#1c2030"), 1))
        for f in (1 / 3, 2 / 3):
            p.drawLine(0, int(h * f), w, int(h * f))
        if len(self.hist) >= 2:
            lo, hi = min(self.hist) - 0.5, max(self.hist) + 0.5
            span = max(1.0, hi - lo)
            p.setPen(QPen(QColor("#7aa2f7"), 1.5))
            p.drawPolyline([QPointF(w * i / (len(self.hist) - 1),
                                    h - (v - lo) / span * (h - 10) - 5)
                            for i, v in enumerate(self.hist)])
            p.setPen(QColor("#9aa1b3"))
            p.drawText(self.rect().adjusted(6, 4, -6, -4), Qt.AlignRight | Qt.AlignTop,
                       f"{self.hist[-1]:.1f} dB")
        else:
            p.setPen(QColor("#5a6178"))
            p.drawText(self.rect(), Qt.AlignCenter, "HSV-fit history")
        p.end()


PRESETS = [
    ("Fast draft", dict(generations=600, population=12, shape_count=60,
                        mutation=10, working=96, seconds=5, psnr=0)),
    ("Balanced (default)", dict(generations=1500, population=16, shape_count=80,
                                mutation=8, working=128, seconds=10, psnr=20)),
    ("Fine detail", dict(generations=4000, population=24, shape_count=120,
                         mutation=6, working=192, seconds=18, psnr=22)),
    ("Painterly — strokes", dict(shape="stroke", shape_count=120, generations=1800,
                                 population=16, mutation=9, working=128, seconds=10, psnr=0)),
    ("Mosaic — squares", dict(shape="square", shape_count=150, generations=2000,
                              population=16, mutation=9, working=128, seconds=10, psnr=0)),
    ("Wild — mixed genome", dict(shape="mixed", shape_count=110, generations=2500,
                                 population=18, mutation=8, working=128, seconds=14, psnr=0)),
    ("Moonlight — crescents", dict(shape="crescent", shape_count=90, generations=1800,
                                   population=16, mutation=9, working=128, seconds=10, psnr=0)),
]


# ═══════════════════════════════ main window ══════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 800)
        self._running = False; self._abort = False
        self._queue = []; self._qpos = 0
        self._worker = None
        self._recorder = None; self._rec_frames = 0; self._rec_muxed = False
        self._player = None; self._audio = None
        self._last_final = None
        self._target_pil = None

        self.fetcher = Fetcher()
        self.fetcher.loaded.connect(self._on_loaded)
        self.fetcher.start()

        self._build_ui()
        self._next_timer = QTimer(self); self._next_timer.setSingleShot(True)
        self._next_timer.timeout.connect(self._play_next)

        self._load_playlist()
        if self.listw.count() == 0:
            self._add_demo()
        self._update_count()

    # ─────────────────────────── UI construction ───────────────────────────
    def _image_label(self):
        lab = QLabel("—")
        lab.setObjectName("imgLabel")
        lab.setMinimumSize(320, 260)
        lab.setAlignment(Qt.AlignCenter)
        return lab

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)
        self._build_toolbar()

        split = QSplitter(Qt.Horizontal); root.addWidget(split, 1)

        # ── left · playlist ──
        left = QGroupBox("Playlist — images from the web")
        lv = QVBoxLayout(left)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔎 filter by caption or url…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._filter_playlist)
        lv.addWidget(self.filter_edit)
        self.listw = QListWidget(); self.listw.setIconSize(QSize(72, 72))
        self.listw.itemDoubleClicked.connect(self._edit_caption)
        lv.addWidget(self.listw, 1)
        self.count_label = QLabel("0 images"); lv.addWidget(self.count_label)
        lrow = QHBoxLayout()
        self.add_btn = QPushButton("Add URLs…"); self.add_btn.clicked.connect(self._add_urls)
        self.demo_btn = QPushButton("Demo set"); self.demo_btn.clicked.connect(self._add_demo)
        self.rem_btn = QPushButton("Remove"); self.rem_btn.clicked.connect(self._remove_sel)
        self.clear_btn = QPushButton("Clear"); self.clear_btn.clicked.connect(self._clear)
        for b in (self.add_btn, self.demo_btn, self.rem_btn, self.clear_btn):
            lrow.addWidget(b)
        lv.addLayout(lrow)
        lv.addWidget(QLabel("Double-click an entry to edit its caption."))
        split.addWidget(left)

        # ── center · stage ──
        center = QWidget(); cv = QVBoxLayout(center); cv.setSpacing(8)
        capg = QGroupBox("Caption"); capv = QVBoxLayout(capg)
        self.caption_label = QLabel("(start the show…)")
        self.caption_label.setObjectName("caption"); self.caption_label.setWordWrap(True)
        self.caption_label.setMinimumHeight(46); capv.addWidget(self.caption_label)
        cv.addWidget(capg)

        imgrow = QHBoxLayout(); imgrow.setSpacing(8)
        self.target_label = self._image_label(); self.evolve_label = self._image_label()
        for title, lab in (("TARGET", self.target_label), ("EVOLVED", self.evolve_label)):
            vv = QVBoxLayout()
            hd = QLabel(title); hd.setObjectName("hdr"); hd.setAlignment(Qt.AlignCenter)
            vv.addWidget(hd); vv.addWidget(lab, 1); imgrow.addLayout(vv)
        cv.addLayout(imgrow, 1)

        graphg = QGroupBox("Fitness — HSV fit history")
        gh = QHBoxLayout(graphg); self.psnr_graph = PsnrGraph(); gh.addWidget(self.psnr_graph)
        cv.addWidget(graphg)

        strip = QHBoxLayout()
        self.pos_label = QLabel(" ")
        self.stats_label = QLabel("Ready."); self.stats_label.setObjectName("stats")
        strip.addWidget(self.pos_label); strip.addStretch(1); strip.addWidget(self.stats_label)
        cv.addLayout(strip)
        split.addWidget(center)

        # ── right · tabbed settings ──
        self.tabs = QTabWidget(); self.tabs.setMinimumWidth(340)
        self.tabs.addTab(self._tab_genome(), "🧬 Genome")
        self.tabs.addTab(self._tab_evolution(), "⚙ Evolution")
        self.tabs.addTab(self._tab_output(), "🎬 Output")
        split.addWidget(self.tabs)
        split.setSizes([300, 660, 350]); split.setStretchFactor(1, 1)

        self.progress = QProgressBar(); root.addWidget(self.progress)
        self.statusBar().showMessage(f"Outputs: {Path.cwd()}  ·  Log: ga_art_show.log")

    def _build_toolbar(self):
        tb = QToolBar("Show controls"); tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb); st = self.style()

        def mk(text, icon, slot, shortcut=None, tip=None):
            a = QAction(icon, text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip); a.setStatusTip(tip)
            a.triggered.connect(slot); tb.addAction(a)
            return a

        self.act_start = mk(" Start Show", st.standardIcon(QStyle.SP_MediaPlay),
                            self.start_show)
        self.act_pause = mk(" Pause", st.standardIcon(QStyle.SP_MediaPause),
                            self.toggle_pause, "Ctrl+P", "Pause / resume evolution")
        self.act_skip = mk(" Skip Image", st.standardIcon(QStyle.SP_MediaSkipForward),
                           self.skip_image, "Ctrl+Right", "Finish this image early")
        self.act_stop = mk(" Stop", st.standardIcon(QStyle.SP_MediaStop),
                           self.stop_show, "Ctrl+.", "Stop the show")
        tb.addSeparator()
        self.act_save = mk(" Save PNG", st.standardIcon(QStyle.SP_DialogSaveButton),
                           self.save_png, "Ctrl+S", "Export the last finished image (1024 px)")
        self.act_save.setEnabled(False)
        for a in (self.act_pause, self.act_skip, self.act_stop):
            a.setEnabled(False)

    def _tab_genome(self):
        w = QWidget(); v = QVBoxLayout(w); v.setSpacing(8)

        prow = QHBoxLayout(); prow.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox(); self.preset_combo.addItem("— choose —", None)
        for name, cfg in PRESETS:
            self.preset_combo.addItem(name, cfg)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        prow.addWidget(self.preset_combo, 1); v.addLayout(prow)

        self.shape_preview = QLabel(); self.shape_preview.setObjectName("preview")
        self.shape_preview.setAlignment(Qt.AlignCenter); self.shape_preview.setMinimumHeight(96)
        v.addWidget(self.shape_preview)

        self.shape_list = QListWidget(); self.shape_list.setObjectName("shapeList")
        self.shape_list.setViewMode(QListWidget.IconMode)
        self.shape_list.setResizeMode(QListWidget.Adjust)
        self.shape_list.setMovement(QListWidget.Static)
        self.shape_list.setGridSize(QSize(150, 34))
        self.shape_list.setUniformItemSizes(True)
        for label, key in SHAPE_MENU:
            it = QListWidgetItem(label); it.setData(Qt.UserRole, key)
            it.setToolTip("every row is a different shape" if key == "mixed"
                          else f"{shape_coords(key)} geometry genes + hue/sat/val/alpha")
            self.shape_list.addItem(it)
        v.addWidget(self.shape_list, 1)

        legend = QLabel(); legend.setObjectName("preview")
        legend.setAlignment(Qt.AlignCenter); legend.setMinimumHeight(30)
        legend.setPixmap(QPixmap.fromImage(pil_to_qimage(make_hsv_strip())))
        legend.setToolTip("the color space: hue wheel (full / 45% saturation) + value ramp")
        v.addWidget(legend)

        self.hsv_view_cb = QCheckBox("Show target & evolved in HSV planes")
        self.hsv_view_cb.setChecked(True)
        self.hsv_view_cb.setToolTip(
            "fitness always matches in HSV — this toggles the display only")
        self.hsv_view_cb.toggled.connect(self._refresh_target_view)
        v.addWidget(self.hsv_view_cb)

        form = QFormLayout()
        self.tri_spin = QSpinBox(); self.tri_spin.setRange(10, 300); self.tri_spin.setValue(80)
        form.addRow("Shape count", self.tri_spin)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["96 px — fast", "128 px — balanced", "192 px — fine"])
        self.quality_combo.setCurrentIndex(1)
        form.addRow("Working resolution", self.quality_combo)
        v.addLayout(form)

        self.shape_list.itemSelectionChanged.connect(self._update_preview)
        self.shape_list.setCurrentRow(0)
        return w

    def _tab_evolution(self):
        w = QWidget(); form = QFormLayout(w)
        self.gen_spin = QSpinBox(); self.gen_spin.setRange(100, 50000)
        self.gen_spin.setValue(1000); self.gen_spin.setSingleStep(100)
        self.pop_spin = QSpinBox(); self.pop_spin.setRange(4, 64); self.pop_spin.setValue(16)
        self.mut_spin = QSpinBox(); self.mut_spin.setRange(1, 60)
        self.mut_spin.setValue(8); self.mut_spin.setSuffix(" %")
        self.psnr_spin = QDoubleSpinBox(); self.psnr_spin.setRange(0, 40)
        self.psnr_spin.setValue(20.0); self.psnr_spin.setSingleStep(0.5)
        self.psnr_spin.setSuffix(" dB"); self.psnr_spin.setSpecialValueText("off")
        self.psnr_spin.setToolTip("HSV-space fit threshold = −10·log10(mean error)")
        self.sec_spin = QDoubleSpinBox(); self.sec_spin.setRange(2, 60)
        self.sec_spin.setValue(10.0); self.sec_spin.setSuffix(" s")
        form.addRow("Generations", self.gen_spin)
        form.addRow("Population", self.pop_spin)
        form.addRow("Mutation rate", self.mut_spin)
        form.addRow("Stop at HSV-fit", self.psnr_spin)
        form.addRow("Video secs / image", self.sec_spin)
        tip = QLabel("Gene-rich shapes (polygons, mixed) need more generations.\n"
                     "Settings apply from the next image on.")
        tip.setWordWrap(True); form.addRow(tip)
        return w

    def _tab_output(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.record_cb = QCheckBox("Record MP4 of the show"); self.record_cb.setChecked(True)
        self.music_cb = QCheckBox("Background music")
        self.loop_cb = QCheckBox("Loop playlist")
        for cb in (self.record_cb, self.music_cb, self.loop_cb):
            v.addWidget(cb)

        form = QFormLayout()
        self.res_combo = QComboBox()
        for label, wh in (("640×480 — fast", (640, 480)), ("960×540", (960, 540)),
                          ("1280×720 — HD", (1280, 720)), ("1920×1080 — Full HD", (1920, 1080))):
            self.res_combo.addItem(label, wh)
        self.res_combo.setCurrentIndex(2)
        self.fps_combo = QComboBox()
        for f_ in (24, 30, 60):
            self.fps_combo.addItem(f"{f_} fps", f_)
        self.fps_combo.setCurrentIndex(1)
        self.qual_combo = QComboBox()
        for q in ("high", "medium", "low"):
            self.qual_combo.addItem(q, q)
        form.addRow("Resolution", self.res_combo)
        form.addRow("Frame rate", self.fps_combo)
        form.addRow("Quality", self.qual_combo)
        v.addLayout(form)

        self.burnin_cb = QCheckBox("Burn in gen / fit + progress bar")
        self.burnin_cb.setChecked(True)
        self.mux_cb = QCheckBox("Mux music into the MP4"); self.mux_cb.setChecked(True)
        v.addWidget(self.burnin_cb); v.addWidget(self.mux_cb)

        self.est_label = QLabel(""); self.est_label.setStyleSheet("color:#8890a4;")
        self.est_label.setWordWrap(True)
        v.addWidget(self.est_label)

        vrow = QHBoxLayout(); vrow.addWidget(QLabel("Vol"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100); self.vol_slider.setValue(40)
        vrow.addWidget(self.vol_slider); v.addLayout(vrow)
        v.addStretch(1)

        self.output_hint = QLabel(
            f"Outputs → {Path.cwd()}\nLog → ga_art_show.log (rewritten each launch)")
        self.output_hint.setWordWrap(True); self.output_hint.setStyleSheet("color:#8890a4;")
        v.addWidget(self.output_hint)

        for wdgt in (self.res_combo, self.fps_combo, self.qual_combo):
            wdgt.currentIndexChanged.connect(self._update_est)
        self.sec_spin.valueChanged.connect(self._update_est)
        self.vol_slider.valueChanged.connect(self._vol_changed)
        self.music_cb.toggled.connect(self._music_toggled)
        if not HAS_MEDIA:
            self.music_cb.setEnabled(False)
            self.music_cb.setToolTip("QtMultimedia not available in this PySide6 install")
        self._update_est()
        return w

    # ─────────────────────────── playlist management ───────────────────────
    def status(self, msg):
        log.info("%s", msg)
        self.statusBar().showMessage(msg)

    def _add_image(self, url, caption):
        d = {"url": url, "caption": caption, "image": None, "error": None}
        it = QListWidgetItem()
        it.setData(Qt.UserRole, d)
        self.listw.addItem(it)
        self._refresh_item(it)
        self.fetcher.add(url)
        self._update_count()

    def _refresh_item(self, it):
        d = it.data(Qt.UserRole)
        if d.get("error"):
            it.setText(f"⚠  {d['url']}  —  {d['error']}")
        elif d.get("image") is None:
            it.setText(f"⏳  {d['url']}")
        else:
            it.setText(d.get("caption") or d["url"])
        it.setToolTip(d["url"])

    def _add_urls(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Add image URLs", "Paste image URLs, one per line:", "")
        if not ok:
            return
        added = 0
        for line in text.splitlines():
            u = line.strip()
            if u.lower().startswith(("http://", "https://")):
                self._add_image(u, ""); added += 1
        if added == 0:
            self.status("No valid http(s) URLs found.")

    def _add_demo(self):
        for u, c in DEMO_IMAGES:
            self._add_image(u, c)

    def _remove_sel(self):
        for it in self.listw.selectedItems():
            self.listw.takeItem(self.listw.row(it))
        self._update_count()

    def _clear(self):
        self.listw.clear()
        self._update_count()

    def _edit_caption(self, it):
        d = it.data(Qt.UserRole)
        text, ok = QInputDialog.getMultiLineText(
            self, "Caption",
            f"Optional text shown while this image evolves:\n{d['url']}",
            d.get("caption") or "")
        if ok:
            d["caption"] = text.strip()
            self._refresh_item(it)
            log.info("caption updated: %s", d["url"])

    def _on_loaded(self, url, ok, img, err):
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            d = it.data(Qt.UserRole)
            if d["url"] == url and d.get("image") is None and not d.get("error"):
                if ok:
                    d["image"] = img
                    it.setIcon(QIcon(QPixmap.fromImage(pil_to_qimage(fit_even(img, 128)))))
                else:
                    d["error"] = err
                    it.setForeground(QBrush(QColor("#e06c75")))
                self._refresh_item(it)
                break
        self._update_count()

    def _filter_playlist(self, text):
        t = text.lower()
        for i in range(self.listw.count()):
            it = self.listw.item(i); d = it.data(Qt.UserRole)
            it.setHidden(t not in (((d.get("caption") or "") + " " + d["url"]).lower()))

    def _update_count(self):
        n = self.listw.count()
        ok = sum(1 for i in range(n) if self.listw.item(i).data(Qt.UserRole).get("image"))
        self.count_label.setText(f"{n} image(s) · {ok} loaded")

    # ─────────────────────────── show control ───────────────────────────
    def _params(self):
        it = self.shape_list.currentItem()
        return dict(
            shape=it.data(Qt.UserRole) if it else "triangle",
            working=int(self.quality_combo.currentText().split()[0]),
            triangles=self.tri_spin.value(),
            population=self.pop_spin.value(),
            generations=self.gen_spin.value(),
            mutation=self.mut_spin.value() / 100.0,
            psnr_target=self.psnr_spin.value(),
            seconds_per_image=self.sec_spin.value(),
        )

    def _scaled(self, pix, label):
        return pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def start_show(self):
        if self._running:
            return
        queue_ = [i for i in range(self.listw.count())
                  if self.listw.item(i).data(Qt.UserRole).get("image") is not None]
        if not queue_:
            QMessageBox.information(self, APP_TITLE,
                "No downloaded images yet — wait for the playlist to load or add URLs.")
            return
        self._queue, self._qpos = queue_, 0
        self._running, self._abort = True, False
        self._apply_running_ui(True)
        log.info("show start: %d image(s), params=%s", len(queue_), self._params())
        self._prepare_music()          # wav must exist before the recorder muxes it
        self._prepare_record()
        self._play_next()

    def _apply_running_ui(self, run):
        self.act_start.setEnabled(not run)
        self.act_pause.setEnabled(run); self.act_skip.setEnabled(run)
        self.act_stop.setEnabled(run)
        self.act_pause.setText(" Pause")
        for b in (self.add_btn, self.demo_btn, self.rem_btn, self.clear_btn):
            b.setEnabled(not run)
        for wdg in (self.record_cb, self.loop_cb):
            wdg.setEnabled(not run)

    def _prepare_record(self):
        self._recorder = None
        if not self.record_cb.isChecked():
            return
        path = Path.cwd() / f"ga_show_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        wav = Path.cwd() / "ambient.wav"
        self._recorder = Mp4Recorder(
            path, self._estimate_frames(),
            fps=self.fps_combo.currentData(), size=self.res_combo.currentData(),
            quality=self.qual_combo.currentData(), burn_in=self.burnin_cb.isChecked(),
            audio_path=str(wav) if (self.music_cb.isChecked() and self.mux_cb.isChecked())
            else None)
        self._recorder.recStatus.connect(self.status)
        self._recorder.recProgress.connect(self._on_rec_progress)
        self._recorder.recDone.connect(self._on_rec_done)
        self._recorder.start()
        log.info("recording %dx%d@%d fps → %s (burn_in=%s, audio=%s)",
                 *self._recorder.size, self._recorder.fps, path.name,
                 self.burnin_cb.isChecked(), bool(self._recorder.audio_path))

    def _estimate_frames(self):
        fps = self.fps_combo.currentData()
        per_img = (2.0 + 0.4 + 0.6 + self.sec_spin.value() + 1.0) * fps
        return int(per_img * max(1, len(self._queue)) + fps * 2.8)

    def _prepare_music(self):
        if not self.music_cb.isChecked():
            return
        if not HAS_MEDIA:
            self.status("QtMultimedia unavailable — no music."); return
        if self._player is None:
            path = Path.cwd() / "ambient.wav"
            if not path.exists():
                self.status("Composing background music…")
                QApplication.processEvents()
                try:
                    make_music(path)
                except Exception as e:
                    self.status(f"Music generation failed: {e}"); return
            self._player = QMediaPlayer(); self._audio = QAudioOutput()
            self._player.setAudioOutput(self._audio)
            self._player.setSource(QUrl.fromLocalFile(str(path)))
            try:
                self._player.setLoopCount(QMediaPlayer.Loops.Infinite)
            except AttributeError:
                self._player.setLoopCount(QMediaPlayer.Infinite)
            self._audio.setVolume(self.vol_slider.value() / 100.0)
            self._player.errorOccurred.connect(
                lambda e, s: self.status(f"Audio error: {s}"))
        self._player.play()

    def _play_next(self):
        if self._abort:
            return self._finalize()
        if self._qpos >= len(self._queue):
            if self.loop_cb.isChecked():
                self._qpos = 0
            else:
                return self._finalize(done=True)
        idx = self._queue[self._qpos]
        it = self.listw.item(idx); d = it.data(Qt.UserRole)
        self.listw.setCurrentRow(idx)
        p = self._params()

        self.caption_label.setText(
            d.get("caption") or "(no caption — double-click the playlist to add one)")
        self.pos_label.setText(f"Image {self._qpos + 1} of {len(self._queue)}")
        self.stats_label.setText("Evolving…")
        self.progress.setValue(0)
        self.psnr_graph.clear()
        self._target_pil = d["image"]
        self._refresh_target_view()
        self.evolve_label.clear(); self.evolve_label.setText("evolving…")

        log.info("image %d/%d — %s", self._qpos + 1, len(self._queue), d["url"])

        if self._recorder:
            self._recorder.start_image(
                d.get("caption") or "Untitled",
                f"image {self._qpos + 1} / {len(self._queue)}",
                np.asarray(letterbox(d["image"], self._recorder.size), np.uint8),
                p["seconds_per_image"])

        self._worker = EvolutionWorker(fit_even(d["image"], p["working"]), p)
        self._worker.uiFrame.connect(self._on_ui_frame)
        self._worker.videoFrame.connect(self._on_video_frame)
        self._worker.progressed.connect(self._on_progress)
        self._worker.message.connect(self.status)
        self._worker.finalImage.connect(self._on_final)
        self._worker.finishedImage.connect(self._on_image_done)
        self._worker.start()

    # ─────────────────────────── worker / recorder slots ───────────────────
    def _on_ui_frame(self, qimg, pil):
        if self.hsv_view_cb.isChecked():
            qimg = pil_to_qimage(hsv_view(pil))
        self.evolve_label.setPixmap(
            self._scaled(QPixmap.fromImage(qimg), self.evolve_label))

    def _on_video_frame(self, arr, gen, gens, fit):
        if self.hsv_view_cb.isChecked():
            arr = np.asarray(hsv_view(Image.fromarray(arr)), np.uint8)
        if self._recorder:
            arr = np.asarray(letterbox(Image.fromarray(arr), self._recorder.size), np.uint8)
            self._recorder.keyframe(arr, gen, gens, fit)

    def _on_progress(self, gen, total, fit, gps):
        self.progress.setValue(int(gen * 100 / max(1, total)))
        self.psnr_graph.add(fit)
        self.stats_label.setText(
            f"gen {gen}/{total} · HSV-fit {fit:.1f} dB · {gps:.1f} gen/s")

    def _on_final(self, qimg):
        self._last_final = qimg
        self.act_save.setEnabled(True)

    def _on_image_done(self, ok):
        log.info("image finished (ok=%s)", ok)
        w, self._worker = self._worker, None
        if w is not None:
            w.deleteLater()
        if self._abort:
            return self._finalize()
        if self._recorder:
            self._recorder.end_image(1.0)
        self._qpos += 1
        self._next_timer.start(700)

    def _on_rec_progress(self, done, est):
        self._rec_frames = done
        self.statusBar().showMessage(f"recording… {done}/{est} frames")

    def _on_rec_done(self, path, frames, muxed):
        self._rec_frames, self._rec_muxed = frames, muxed
        self.status(f"video saved: {path} ({frames} frames"
                    + (", audio muxed)" if muxed else ")"))

    def toggle_pause(self):
        if self._worker:
            self._worker.paused = not self._worker.paused
            self.act_pause.setText(" Resume" if self._worker.paused else " Pause")

    def skip_image(self):
        if self._worker:
            self._worker.skip()

    def stop_show(self):
        self._abort = True
        if self._worker:
            self._worker.stop()

    def _finalize(self, done=False):
        log.info("finalize (done=%s)", done)
        self._worker = None
        rec = self._recorder; self._recorder = None
        if rec:
            if done:
                rec.outro(f"{len(self._queue)} images  ·  {time.strftime('%Y-%m-%d %H:%M')}")
            rec.finish()
            if not rec.wait(30000):
                rec.terminate()
            if done and rec._frames:
                QMessageBox.information(self, APP_TITLE,
                    f"Show finished.\n\nVideo: {rec.path}\n{rec._frames} frames"
                    + ("\nMusic muxed into the MP4 ✔" if rec._muxed else ""))
            rec.deleteLater()
        msg = "Show complete." if done else "Show stopped."
        if rec:
            msg += (f"  Video: {rec.path.name} ({rec._frames} frames"
                    + (", music muxed)" if rec._muxed else ")"))
        self.status(msg)
        if self._player:
            self._player.pause()
        self._running = False
        self._apply_running_ui(False)
        self.stats_label.setText("Ready." if not done else "Done.")

    # ─────────────────────────── genome tab handlers ───────────────────────
    def _update_preview(self):
        it = self.shape_list.currentItem()
        if not it:
            return
        pm = QPixmap.fromImage(pil_to_qimage(make_shape_preview(it.data(Qt.UserRole))))
        self.shape_preview.setPixmap(pm.scaled(self.shape_preview.size(),
                                               Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))

    def _apply_preset(self, idx):
        cfg = self.preset_combo.itemData(idx)
        if not cfg:
            return
        if cfg.get("shape"):
            for i in range(self.shape_list.count()):
                if self.shape_list.item(i).data(Qt.UserRole) == cfg["shape"]:
                    self.shape_list.setCurrentRow(i); break
        for ck, attr in dict(generations="gen_spin", population="pop_spin",
                             shape_count="tri_spin", mutation="mut_spin",
                             psnr="psnr_spin", seconds="sec_spin").items():
            if ck in cfg:
                getattr(self, attr).setValue(cfg[ck])
        if "working" in cfg:
            self.quality_combo.setCurrentIndex(
                {"96": 0, "128": 1, "192": 2}[str(cfg["working"])])
        self.status(f"Preset applied: {self.preset_combo.currentText()}")
        log.info("preset applied: %s", cfg)

    def _refresh_target_view(self):
        if self._target_pil is None:
            return
        img = hsv_view(self._target_pil) if self.hsv_view_cb.isChecked() else self._target_pil
        self.target_label.setPixmap(
            self._scaled(QPixmap.fromImage(pil_to_qimage(img)), self.target_label))

    def save_png(self):
        if self._last_final is None:
            return
        path = Path.cwd() / f"ga_best_{time.strftime('%Y%m%d_%H%M%S')}.png"
        try:
            self._last_final.save(str(path))
            log.info("saved best genome render -> %s", path)
            self.status(f"Saved PNG → {path.name}")
        except Exception as e:
            self.status(f"PNG save failed: {e}")

    def _update_est(self):
        w, h = self.res_combo.currentData()
        fps = self.fps_combo.currentData()
        bpp = {"high": 0.12, "medium": 0.08, "low": 0.05}[self.qual_combo.currentData()]
        mb = w * h * fps * bpp * 60 / 8e6
        self.est_label.setText(
            f"≈ {mb:.0f} MB/min · ~{self._estimate_frames()} frames this show")

    # ─────────────────────────── music / persistence ───────────────────────
    def _vol_changed(self, v):
        if self._audio:
            self._audio.setVolume(v / 100.0)

    def _music_toggled(self, on):
        if self._player:
            self._player.play() if on else self._player.pause()

    def _save_playlist(self):
        try:
            data = [{"url": self.listw.item(i).data(Qt.UserRole)["url"],
                     "caption": self.listw.item(i).data(Qt.UserRole).get("caption") or ""}
                    for i in range(self.listw.count())]
            (OUTPUT_DIR / "playlist.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
            log.info("saved playlist (%d entries) -> %s",
                     len(data), OUTPUT_DIR / "playlist.json")
        except Exception:
            log.exception("failed to save playlist")

    def _load_playlist(self):
        f = OUTPUT_DIR / "playlist.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for e in data:
                    if e.get("url"):
                        self._add_image(e["url"], e.get("caption") or "")
                log.info("loaded %d playlist entries from %s", len(data), f)
            except Exception as e:
                log.warning("could not load playlist.json: %s", e)

    def closeEvent(self, event: QCloseEvent):
        log.info("closing — stopping threads, saving playlist…")
        self._save_playlist()
        self._abort = True
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        self.fetcher.stop()
        if not self.fetcher.wait(1500):
            self.fetcher.terminate()
        if self._recorder:
            self._recorder.finish()
            if not self._recorder.wait(10000):
                self._recorder.terminate()
        if self._player:
            self._player.stop()
        event.accept()


# ═══════════════════════════ stylesheet & main ════════════════════════════

QSS = """
QMainWindow, QDialog { background:#14161d; }
QWidget { color:#d7dbe4; font-size:13px; background:transparent; }
QGroupBox { background:#191c26; border:1px solid #272c3a; border-radius:10px;
            margin-top:14px; padding:12px 10px 10px 10px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:14px; color:#7aa2f7; }
QListWidget, QComboBox QAbstractItemView { background:#12141b;
    border:1px solid #272c3a; border-radius:8px; }
QListWidget::item { padding:6px; border-radius:6px; }
QListWidget::item:selected { background:#2b3b63; }
QListWidget#shapeList { background:#12141b; border:1px solid #272c3a; border-radius:8px; }
QListWidget#shapeList::item:selected { background:#2b3b63; border-radius:6px; }
QToolBar { background:#171a24; border:none; border-bottom:1px solid #272c3a;
           padding:4px 6px; spacing:4px; }
QToolButton { background:transparent; border:1px solid transparent; border-radius:7px;
              padding:5px 12px; }
QToolButton:hover { background:#2c3348; }
QToolButton:disabled { color:#5a6072; }
QTabWidget::pane { border:1px solid #272c3a; border-radius:8px; background:#151822; }
QTabBar::tab { background:#1a1e2a; color:#9aa1b3; padding:7px 16px; margin-right:3px;
               border-top-left-radius:8px; border-top-right-radius:8px; }
QTabBar::tab:selected { background:#151822; color:#7aa2f7; font-weight:600; }
QPushButton { background:#232838; border:1px solid #31374a; border-radius:8px;
              padding:7px 14px; }
QPushButton:hover { background:#2c3348; }
QPushButton:disabled { color:#666d80; background:#1c202b; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background:#12141b; border:1px solid #31374a; border-radius:7px; padding:5px 7px; }
QProgressBar { background:#12141b; border:1px solid #31374a; border-radius:7px;
               text-align:center; height:18px; }
QProgressBar::chunk { background:#3a70ea; border-radius:6px; }
QSlider::groove:horizontal { height:5px; background:#31374a; border-radius:2px; }
QSlider::handle:horizontal { width:14px; margin:-5px 0; background:#7aa2f7;
                             border-radius:7px; }
QLabel#caption { font-size:16px; color:#eef1f8; }
QLabel#hdr { color:#7aa2f7; font-weight:600; }
QLabel#imgLabel { background:#0e1016; border:1px solid #272c3a; border-radius:8px; }
QLabel#preview { background:#0e1016; border:1px solid #272c3a; border-radius:8px; }
QLabel#stats { font-family:'Consolas','DejaVu Sans Mono',monospace; color:#9fe08a; }
QSplitter::handle { background:#222633; width:5px; }
QStatusBar { background:#101218; color:#9aa1b3; }
"""


def main():
    log.info("═══ %s ═══ cwd=%s | outputs=%s | logfile=%s (mode=w)",
             APP_TITLE, Path.cwd(), OUTPUT_DIR, LOG_FILE)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()