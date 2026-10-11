"""
YouTube Video & Shorts Creator with Procedural Audio Layering
Select folders of sound categories and layer them together.
"""

import sys
import os
import re
import tempfile
import shutil
import random
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QColorDialog, QFileDialog, QProgressBar,
    QFormLayout, QTabWidget, QMessageBox, QCheckBox, QListWidget,
    QListWidgetItem, QAbstractItemView, QGroupBox
)
from PySide6.QtGui import (
    QFont, QFontDatabase, QColor, QPainter, QPixmap, QImage,
    QPalette, QLinearGradient, QBrush
)
from PySide6.QtCore import Qt, QThread, Signal, QRect

try:
    from moviepy.editor import (
        ImageSequenceClip, AudioFileClip, concatenate_audioclips,
        CompositeAudioClip
    )
    from moviepy.audio.AudioClip import AudioClip
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Natural sort helper
# ---------------------------------------------------------------------------

def natural_key(text):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', text)]


# ---------------------------------------------------------------------------
# Animation renderers (unchanged)
# ---------------------------------------------------------------------------

def draw_static_text(painter, text, font, rect, color, align):
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(rect, align | Qt.AlignVCenter, text)


def draw_fade_text(painter, text, font, rect, color, align, progress):
    c = QColor(color)
    c.setAlpha(int(255 * progress))
    painter.setFont(font)
    painter.setPen(c)
    painter.drawText(rect, align | Qt.AlignVCenter, text)


def draw_typewriter_text(painter, text, font, rect, color, align, progress):
    n = max(1, int(len(text) * progress))
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(rect, align | Qt.AlignVCenter, text[:n])


def draw_slide_text(painter, text, font, rect, color, align, progress):
    offset = int(rect.width() * (1.0 - progress))
    r = QRect(rect.x() - offset, rect.y(), rect.width(), rect.height())
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(r, align | Qt.AlignVCenter, text)


def draw_word_by_word(painter, text, font, rect, color, align, progress):
    words = text.split()
    if not words:
        return
    n = max(1, int(len(words) * progress))
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(rect, align | Qt.AlignVCenter, " ".join(words[:n]))


ANIMATIONS = {
    "Static": draw_static_text,
    "Fade In": draw_fade_text,
    "Typewriter": draw_typewriter_text,
    "Slide In": draw_slide_text,
    "Word by Word": draw_word_by_word,
}


# ---------------------------------------------------------------------------
# Frame renderer (unchanged)
# ---------------------------------------------------------------------------

def render_frame(cfg, t, total_t):
    img = QImage(cfg["width"], cfg["height"], QImage.Format_RGBA8888)

    if cfg.get("use_gradient"):
        grad = QLinearGradient(0, 0, 0, cfg["height"])
        grad.setColorAt(0, QColor(cfg["bg_color"]))
        grad.setColorAt(1, QColor(cfg["grad_color2"]))
        fill = QBrush(grad)
    else:
        fill = QBrush(QColor(cfg["bg_color"]))

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.fillRect(img.rect(), fill)

    pad = cfg["padding"]
    text_rect = QRect(pad, pad, cfg["width"] - 2 * pad, cfg["height"] - 2 * pad)

    font = QFont(cfg["font_name"], cfg["font_size"])
    font.setBold(cfg.get("bold", False))
    font.setItalic(cfg.get("italic", False))

    color = QColor(cfg["font_color"])
    align = cfg["align"]
    anim_name = cfg["animation"]

    anim_dur = cfg.get("anim_duration", 1.0)
    progress = min(1.0, t / anim_dur) if total_t > 0 and anim_dur > 0 else 1.0

    draw_fn = ANIMATIONS.get(anim_name, draw_static_text)
    if anim_name in ("Fade In", "Typewriter", "Slide In", "Word by Word"):
        draw_fn(painter, cfg["text"], font, text_rect, color, align, progress)
    else:
        draw_fn(painter, cfg["text"], font, text_rect, color, align)

    painter.end()
    return img


# ---------------------------------------------------------------------------
# Audio layering engine
# ---------------------------------------------------------------------------

class AudioLayerEngine:
    """
    Procedurally layers audio clips from multiple folders.
    """
    
    @staticmethod
    def load_clips_from_folder(folder_path):
        """Load all audio clips from a folder."""
        exts = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")
        clips = []
        for name in os.listdir(folder_path):
            if name.lower().endswith(exts):
                path = os.path.join(folder_path, name)
                try:
                    clip = AudioFileClip(path)
                    clips.append({
                        "path": path,
                        "name": name,
                        "clip": clip,
                        "duration": clip.duration
                    })
                except Exception as e:
                    print(f"Could not load {path}: {e}")
        # Sort naturally
        clips.sort(key=lambda c: natural_key(c["name"]))
        return clips
    
    @staticmethod
    def mix_clips_sequential(folder_clips, target_duration):
        """
        Play clips from one folder sequentially, looping to fill duration.
        """
        if not folder_clips:
            return None
        
        # Concatenate all clips in folder
        clips = [c["clip"] for c in folder_clips]
        base_track = concatenate_audioclips(clips)
        
        # Loop to fill target duration
        loops = int(target_duration / base_track.duration) + 1
        looped = concatenate_audioclips([base_track] * loops)
        return looped.subclip(0, target_duration)
    
    @staticmethod
    def mix_clips_randomized(folder_clips, target_duration):
        """
        Randomly select and concatenate clips to fill duration.
        """
        if not folder_clips:
            return None
        
        result_clips = []
        current_time = 0.0
        
        while current_time < target_duration:
            # Pick random clip
            clip_data = random.choice(folder_clips)
            clip = clip_data["clip"]
            
            # Add it
            result_clips.append(clip)
            current_time += clip.duration
        
        # Concatenate and trim
        track = concatenate_audioclips(result_clips)
        return track.subclip(0, target_duration)
    
    @staticmethod
    def layer_multiple_tracks(tracks, target_duration):
        """
        Mix multiple audio tracks together (overlay them).
        Each track is normalized to prevent clipping.
        """
        if not tracks:
            return None
        
        if len(tracks) == 1:
            return tracks[0]
        
        # Normalize volume for each track (simple scaling)
        scale_factor = 1.0 / len(tracks)
        
        def make_audio_func(track, scale):
            """Create audio function for a track with scaling."""
            def audio_func(t):
                # Get audio samples at time t
                try:
                    samples = track.get_frame(t)
                    if isinstance(samples, np.ndarray):
                        return samples * scale
                    return [0, 0]
                except:
                    return [0, 0]
            return audio_func
        
        # Create scaled audio clips
        scaled_clips = []
        for i, track in enumerate(tracks):
            # Create new clip with scaled audio
            scaled = AudioClip(
                make_audio_func(track, scale_factor),
                duration=target_duration,
                fps=track.fps if hasattr(track, 'fps') else 44100
            )
            scaled_clips.append(scaled)
        
        # Composite them together
        mixed = CompositeAudioClip(scaled_clips)
        return mixed


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class RenderWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(str)
    finished_err = Signal(str)

    def __init__(self, slides, fps, output_path, selected_folders, audio_mode, parent=None):
        super().__init__(parent)
        self.slides = slides
        self.fps = fps
        self.output_path = output_path
        self.selected_folders = selected_folders  # list of folder paths
        self.audio_mode = audio_mode  # "sequential", "randomized", "none"
        self._abort = False
        self._tmpdir = None

    def abort(self):
        self._abort = True

    def run(self):
        if not MOVIEPY_AVAILABLE:
            self.finished_err.emit("moviepy is not installed.")
            return

        self._tmpdir = tempfile.mkdtemp(prefix="ytvideo_")
        video_clip = None
        audio_clip = None
        
        try:
            total_frames = sum(int(s["duration"] * self.fps) for s in self.slides)
            total_duration = sum(s["duration"] for s in self.slides)

            # Render frames
            done = 0
            frame_paths = []
            for s_idx, slide in enumerate(self.slides):
                self.status.emit(f"Rendering slide {s_idx + 1}/{len(self.slides)}...")
                n_frames = max(1, int(slide["duration"] * self.fps))
                total_t = slide["duration"]
                for f in range(n_frames):
                    if self._abort:
                        self.finished_err.emit("Cancelled.")
                        return
                    t = f / self.fps
                    img = render_frame(slide, t, total_t)
                    path = os.path.join(self._tmpdir, f"f_{s_idx:04d}_{f:06d}.png")
                    img.save(path, "PNG")
                    frame_paths.append(path)
                    done += 1
                    self.progress.emit(int(done * 100 / max(1, total_frames)))

            self.status.emit("Building video clip...")
            video_clip = ImageSequenceClip(frame_paths, fps=self.fps)

            # Build layered audio
            if self.selected_folders and self.audio_mode != "none":
                self.status.emit("Loading audio folders...")
                
                # Load clips from each selected folder
                folder_tracks = []
                for folder_path in self.selected_folders:
                    folder_name = os.path.basename(folder_path)
                    self.status.emit(f"Loading {folder_name}...")
                    
                    clips = AudioLayerEngine.load_clips_from_folder(folder_path)
                    if not clips:
                        continue
                    
                    # Create track from this folder
                    if self.audio_mode == "sequential":
                        track = AudioLayerEngine.mix_clips_sequential(clips, total_duration)
                    else:  # randomized
                        track = AudioLayerEngine.mix_clips_randomized(clips, total_duration)
                    
                    if track:
                        folder_tracks.append(track)
                
                # Layer all folder tracks together
                if folder_tracks:
                    self.status.emit("Layering audio tracks...")
                    audio_clip = AudioLayerEngine.layer_multiple_tracks(
                        folder_tracks, total_duration
                    )
                    video_clip = video_clip.set_audio(audio_clip)

            self.status.emit("Encoding video...")
            video_clip.write_videofile(
                self.output_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
                preset="medium",
                fps=self.fps,
            )
            self.finished_ok.emit(self.output_path)

        except Exception as e:
            import traceback
            self.finished_err.emit(f"Error: {e}\n\n{traceback.format_exc()}")
        finally:
            if video_clip:
                try: video_clip.close()
                except: pass
            if audio_clip and video_clip and audio_clip is not video_clip.audio:
                try: audio_clip.close()
                except: pass
            if self._tmpdir and os.path.isdir(self._tmpdir):
                shutil.rmtree(self._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Preview widget (unchanged)
# ---------------------------------------------------------------------------

class PreviewWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background:#222; border:1px solid #444;")
        self._cfg = None

    def set_config(self, cfg):
        self._cfg = cfg
        self.update_preview()

    def update_preview(self, t=0.5):
        if not self._cfg:
            return
        total_t = self._cfg.get("duration", 2.0)
        img = render_frame(self._cfg, t, total_t)
        pm = QPixmap.fromImage(img)
        scaled = pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_preview()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    PRESETS = {
        "YouTube Video (1920x1080)": (1920, 1080),
        "YouTube Short (1080x1920)": (1080, 1920),
        "Square (1080x1080)": (1080, 1080),
        "720p Video (1280x720)": (1280, 720),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Video & Shorts Creator")
        self.resize(1150, 760)

        self.worker = None
        self.current_cfg = self._default_cfg()
        self.audio_root = ""
        self.available_folders = []  # list of folder paths
        self.selected_folders = []   # list of selected folder paths

        self._build_ui()
        self._connect_signals()
        self._refresh_fonts()
        self._rebuild_cfg()
        self.preview.update_preview()

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setMaximumWidth(480)

        tabs = QTabWidget()
        tabs.addTab(self._build_text_tab(), "Text")
        tabs.addTab(self._build_style_tab(), "Style")
        tabs.addTab(self._build_animation_tab(), "Animation")
        tabs.addTab(self._build_audio_tab(), "Audio Layers")
        tabs.addTab(self._build_export_tab(), "Export")
        left_layout.addWidget(tabs)

        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("Refresh Preview")
        self.btn_export = QPushButton("Export Video...")
        self.btn_export.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;padding:8px;font-weight:bold;}"
        )
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_export)
        left_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("Ready.")
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status_label)

        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.addWidget(QLabel("Preview:"))
        self.preview = PreviewWidget()
        r_layout.addWidget(self.preview, 1)
        self.fmt_label = QLabel("Format: 1920 x 1080")
        r_layout.addWidget(self.fmt_label)

        root.addWidget(left)
        root.addWidget(right, 1)

    def _build_text_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(QLabel("Text (one slide per line):"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Each line becomes a separate slide.")
        self.text_edit.setPlainText("Hello YouTube!\nWelcome to my video.")
        lay.addWidget(self.text_edit)

        form = QFormLayout()
        self.slide_duration = QDoubleSpinBox()
        self.slide_duration.setRange(0.5, 120.0)
        self.slide_duration.setValue(3.0)
        self.slide_duration.setSuffix(" s")
        form.addRow("Duration per slide:", self.slide_duration)

        self.align_combo = QComboBox()
        self.align_combo.addItems(["Center", "Left", "Right"])
        form.addRow("Text alignment:", self.align_combo)

        lay.addLayout(form)
        return w

    def _build_style_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()

        self.font_combo = QComboBox()
        form.addRow("Font:", self.font_combo)

        self.font_size = QSpinBox()
        self.font_size.setRange(12, 400)
        self.font_size.setValue(96)
        form.addRow("Font size:", self.font_size)

        self.bold_chk = QCheckBox("Bold")
        self.italic_chk = QCheckBox("Italic")
        chk_row = QHBoxLayout()
        chk_row.addWidget(self.bold_chk)
        chk_row.addWidget(self.italic_chk)
        form.addRow("Style:", chk_row)

        self.font_color_btn = QPushButton("Choose...")
        self.font_color = "#FFFFFF"
        self._update_color_btn(self.font_color_btn, self.font_color)
        form.addRow("Text color:", self.font_color_btn)

        self.bg_color_btn = QPushButton("Choose...")
        self.bg_color = "#1a1a2e"
        self._update_color_btn(self.bg_color_btn, self.bg_color)
        form.addRow("Background:", self.bg_color_btn)

        self.gradient_chk = QCheckBox("Use vertical gradient")
        form.addRow("", self.gradient_chk)

        self.grad_color2_btn = QPushButton("Choose...")
        self.grad_color2 = "#0f3460"
        self._update_color_btn(self.grad_color2_btn, self.grad_color2)
        form.addRow("Gradient end:", self.grad_color2_btn)

        self.padding = QSpinBox()
        self.padding.setRange(0, 500)
        self.padding.setValue(80)
        form.addRow("Padding:", self.padding)

        lay.addLayout(form)
        return w

    def _build_animation_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()

        self.anim_combo = QComboBox()
        self.anim_combo.addItems(list(ANIMATIONS.keys()))
        form.addRow("Animation:", self.anim_combo)

        self.anim_duration = QDoubleSpinBox()
        self.anim_duration.setRange(0.1, 10.0)
        self.anim_duration.setValue(1.0)
        self.anim_duration.setSingleStep(0.1)
        self.anim_duration.setSuffix(" s")
        form.addRow("Animation duration:", self.anim_duration)

        lay.addLayout(form)
        lay.addStretch()
        return w

    def _build_audio_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(QLabel("<b>Procedural Audio Layering</b>"))
        lay.addWidget(QLabel("Select a root folder containing subfolders of sounds."))

        # Root folder picker
        folder_row = QHBoxLayout()
        self.audio_root_edit = QLineEdit()
        self.audio_root_edit.setReadOnly(True)
        self.audio_root_edit.setPlaceholderText("(no root folder selected)")
        self.btn_pick_root = QPushButton("Pick Root Folder...")
        folder_row.addWidget(self.audio_root_edit, 1)
        folder_row.addWidget(self.btn_pick_root)
        lay.addLayout(folder_row)

        # Mode
        form = QFormLayout()
        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItems([
            "Sequential (play clips in order, loop)",
            "Randomized (randomly pick clips)",
            "No audio",
        ])
        form.addRow("Mixing mode:", self.audio_mode_combo)
        lay.addLayout(form)

        # Available folders
        lay.addWidget(QLabel("<b>Available sound folders:</b>"))
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.MultiSelection)
        lay.addWidget(self.folder_list, 1)

        # Info
        self.audio_info_label = QLabel("Select a root folder to see available sound categories.")
        lay.addWidget(self.audio_info_label)

        return w

    def _build_export_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        form.addRow("Format preset:", self.preset_combo)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 4096)
        self.width_spin.setValue(1920)
        form.addRow("Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 4096)
        self.height_spin.setValue(1080)
        form.addRow("Height:", self.height_spin)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.setValue(30)
        form.addRow("FPS:", self.fps_spin)

        lay.addLayout(form)
        lay.addStretch()
        return w

    # ---- Signals ----------------------------------------------------------

    def _connect_signals(self):
        self.font_combo.currentTextChanged.connect(self._on_change)
        self.font_size.valueChanged.connect(self._on_change)
        self.bold_chk.toggled.connect(self._on_change)
        self.italic_chk.toggled.connect(self._on_change)
        self.font_color_btn.clicked.connect(lambda: self._pick_color("font"))
        self.bg_color_btn.clicked.connect(lambda: self._pick_color("bg"))
        self.grad_color2_btn.clicked.connect(lambda: self._pick_color("grad2"))
        self.gradient_chk.toggled.connect(self._on_change)
        self.padding.valueChanged.connect(self._on_change)
        self.align_combo.currentIndexChanged.connect(self._on_change)
        self.anim_combo.currentTextChanged.connect(self._on_change)
        self.anim_duration.valueChanged.connect(self._on_change)
        self.slide_duration.valueChanged.connect(self._on_change)
        self.preset_combo.currentTextChanged.connect(self._on_preset)
        self.width_spin.valueChanged.connect(self._on_change)
        self.height_spin.valueChanged.connect(self._on_change)
        self.fps_spin.valueChanged.connect(self._on_change)

        self.btn_preview.clicked.connect(self.preview.update_preview)
        self.btn_export.clicked.connect(self._export)

        # Audio signals
        self.btn_pick_root.clicked.connect(self._pick_audio_root)
        self.folder_list.itemSelectionChanged.connect(self._update_selected_folders)

    def _on_change(self, *args):
        self._rebuild_cfg()
        self.preview.update_preview()

    def _on_preset(self, name):
        if name in self.PRESETS:
            w, h = self.PRESETS[name]
            self.width_spin.setValue(w)
            self.height_spin.setValue(h)

    # ---- Helpers ----------------------------------------------------------

    def _refresh_fonts(self):
        self.font_combo.blockSignals(True)
        self.font_combo.clear()
        families = sorted(set(QFontDatabase.families()))
        self.font_combo.addItems(families)
        for preferred in ["Arial", "Helvetica", "DejaVu Sans", "Segoe UI",
                          "Roboto", "Ubuntu", "Sans Serif"]:
            if preferred in families:
                self.font_combo.setCurrentText(preferred)
                break
        self.font_combo.blockSignals(False)

    def _pick_color(self, which):
        current = {"font": self.font_color, "bg": self.bg_color, "grad2": self.grad_color2}[which]
        c = QColorDialog.getColor(QColor(current), self)
        if c.isValid():
            hexc = c.name()
            if which == "font":
                self.font_color = hexc
                self._update_color_btn(self.font_color_btn, hexc)
            elif which == "bg":
                self.bg_color = hexc
                self._update_color_btn(self.bg_color_btn, hexc)
            else:
                self.grad_color2 = hexc
                self._update_color_btn(self.grad_color2_btn, hexc)
            self._on_change()

    def _update_color_btn(self, btn, color):
        btn.setStyleSheet(
            f"background:{color}; color:{'white' if QColor(color).lightness() < 128 else 'black'};"
        )
        btn.setText(color)

    def _default_cfg(self):
        return {
            "width": 1920, "height": 1080,
            "bg_color": "#1a1a2e",
            "text": "Hello YouTube!",
            "font_name": "Arial",
            "font_size": 96,
            "font_color": "#FFFFFF",
            "bold": False, "italic": False,
            "animation": "Fade In",
            "anim_duration": 1.0,
            "align": Qt.AlignCenter,
            "padding": 80,
            "duration": 3.0,
            "use_gradient": False,
            "grad_color2": "#0f3460",
        }

    def _rebuild_cfg(self):
        align_map = {
            "Center": Qt.AlignCenter,
            "Left": Qt.AlignLeft,
            "Right": Qt.AlignRight,
        }
        self.current_cfg = {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "bg_color": self.bg_color,
            "text": self.text_edit.toPlainText().splitlines()[0] or " ",
            "font_name": self.font_combo.currentText() or "Arial",
            "font_size": self.font_size.value(),
            "font_color": self.font_color,
            "bold": self.bold_chk.isChecked(),
            "italic": self.italic_chk.isChecked(),
            "animation": self.anim_combo.currentText(),
            "anim_duration": self.anim_duration.value(),
            "align": align_map.get(self.align_combo.currentText(), Qt.AlignCenter),
            "padding": self.padding.value(),
            "duration": self.slide_duration.value(),
            "use_gradient": self.gradient_chk.isChecked(),
            "grad_color2": self.grad_color2,
        }
        self.fmt_label.setText(
            f"Format: {self.current_cfg['width']} x {self.current_cfg['height']}"
        )

    # ---- Audio folder handling -------------------------------------------

    def _pick_audio_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Audio Folder")
        if not folder:
            return
        self.audio_root = folder
        self.audio_root_edit.setText(folder)
        self._scan_audio_folders()

    def _scan_audio_folders(self):
        """Scan root folder for subfolders containing audio."""
        self.available_folders = []
        self.folder_list.clear()
        
        if not os.path.isdir(self.audio_root):
            return
        
        # Find all subfolders
        for item in os.listdir(self.audio_root):
            item_path = os.path.join(self.audio_root, item)
            if os.path.isdir(item_path):
                # Check if it contains audio files
                exts = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")
                has_audio = any(
                    f.lower().endswith(exts)
                    for f in os.listdir(item_path)
                    if os.path.isfile(os.path.join(item_path, f))
                )
                if has_audio:
                    self.available_folders.append(item_path)
        
        # Sort naturally
        self.available_folders.sort(key=lambda p: natural_key(os.path.basename(p)))
        
        # Populate list
        for folder_path in self.available_folders:
            folder_name = os.path.basename(folder_path)
            # Count audio files
            exts = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")
            count = sum(1 for f in os.listdir(folder_path) if f.lower().endswith(exts))
            item = QListWidgetItem(f"{folder_name} ({count} files)")
            item.setData(Qt.UserRole, folder_path)
            self.folder_list.addItem(item)
        
        self._update_info()

    def _update_selected_folders(self):
        """Update selected folders based on list selection."""
        self.selected_folders = []
        for item in self.folder_list.selectedItems():
            folder_path = item.data(Qt.UserRole)
            self.selected_folders.append(folder_path)
        self._update_info()

    def _update_info(self):
        n = len(self.selected_folders)
        if n == 0:
            self.audio_info_label.setText("No folders selected.")
        else:
            names = [os.path.basename(p) for p in self.selected_folders]
            self.audio_info_label.setText(
                f"Selected {n} folder(s): {', '.join(names)}"
            )

    # ---- Export -----------------------------------------------------------

    def _export(self):
        if not MOVIEPY_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "moviepy is not installed.\nRun: pip install moviepy numpy"
            )
            return

        lines = [ln for ln in self.text_edit.toPlainText().splitlines() if ln.strip()]
        if not lines:
            QMessageBox.warning(self, "No text", "Please enter some text.")
            return

        mode_idx = self.audio_mode_combo.currentIndex()
        audio_mode = ["sequential", "randomized", "none"][mode_idx]

        default_name = f"youtube_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save video", default_name, "MP4 Video (*.mp4)"
        )
        if not path:
            return

        base = dict(self.current_cfg)
        slides = []
        for line in lines:
            s = dict(base)
            s["text"] = line
            slides.append(s)

        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Starting...")

        self.worker = RenderWorker(
            slides=slides,
            fps=self.fps_spin.value(),
            output_path=path,
            selected_folders=list(self.selected_folders),
            audio_mode=audio_mode,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self._on_export_ok)
        self.worker.finished_err.connect(self._on_export_err)
        self.worker.start()

    def _on_export_ok(self, path):
        self.btn_export.setEnabled(True)
        self.progress.setValue(100)
        self.status_label.setText(f"Saved: {path}")
        QMessageBox.information(self, "Done", f"Video saved to:\n{path}")

    def _on_export_err(self, msg):
        self.btn_export.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText("Error.")
        QMessageBox.critical(self, "Export failed", msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)

    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 40))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(40, 40, 55))
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(55, 55, 70))
    p.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(p)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()