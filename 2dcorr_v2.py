# -*- coding: utf-8 -*-
"""
==================================================================================================
||                                                                                              ||
||      🔬 2D Class Average Analysis Tool v4.2 (Restored & Enhanced, GPU Accelerated) 🔬        ||
||                                                                                              ||
==================================================================================================
||                                                                                              ||
||  Author: Stavros Azinas (Restored and Enhanced by Gemini)                                    ||
||  Date: 2025-07-31                                                                            ||
||                                                                                              ||
||  Description:                                                                                ||
||  This is a restored and significantly enhanced version of the original analysis tool.        ||
||  It provides a robust platform for analyzing 2D image datasets, such as those from           ||
||  cryo-electron microscopy. The tool detects particles (subsquares), preprocesses them,        ||
||  and performs intensive pairwise cross-correlation to identify structural similarities.      ||
||                                                                                              ||
||  This version uses a stable method for displaying plots by generating an HTML file           ||
||  and opening it in a dedicated application window, avoiding system browser dependencies.     ||
||                                                                                              ||
||  Core Features:                                                                              ||
||  - GPU acceleration via CuPy or PyTorch for massive speed-up in correlation.                 ||
||  - Fully functional Automatic and Reference-Based particle detection modes.                  ||
||  - Selectable correlation modes (All vs All, One Image vs Others).                           ||
||  - Pre-analysis quality filter to remove low-variance/noise particles.                       ||
||  - FIX: Rotational alignment is now correctly visualized in the Top Pairs gallery.           ||
||  - FIX: Quality filter is now correctly applied to all analysis steps.                       ||
||  - FIX: Correlation processing logic has been restored to a stable state.                    ||
||  - FIX: App icon loading is now more robust.                                                 ||
||  - NEW: User-configurable rotation step for correlation search.                              ||
||  - NEW: Shortened particle naming for 'J####' files.                                         ||
||  - NEW: Heatmap and Network Graph now open in a dedicated, stable app window.                ||
||  - Advanced image preprocessing pipeline (CLAHE, advanced edge detection).                   ||
||  - Dynamic quality weighting for more accurate and nuanced correlation scores.               ||
||  - A rich top-pairs gallery for detailed results visualization.                              ||
||  - Particle Analysis window with per-particle average scores and best matches.               ||
||  - Detailed system monitoring (CPU, RAM, GPU) and granular progress feedback.                ||
||  - Theming support (Modern Dark/Light) for improved readability and user experience.         ||
||  - Session management for saving and loading parameters.                                     ||
||  - A detailed, real-time application logging panel for debugging and transparency.           ||
||                                                                                              ||
==================================================================================================
"""
# --- Core Imports ---
import sys
import os
import re
import cv2
import numpy as np
import psutil
import logging
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import webbrowser
import tempfile

# --- Third-Party Imports ---
# Attempt to import GPUtil for detailed GPU monitoring.
try:
    import GPUtil
except ImportError:
    print("⚠️ GPUtil library not found. GPU monitoring will be limited. Install with: pip install gputil")
    GPUtil = None

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QRadioButton, QButtonGroup,
    QComboBox, QDialog, QDialogButtonBox, QFrame,
    QSplitter, QTabWidget, QTextEdit, QCheckBox, QSlider, QStackedWidget,
    QMenuBar, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QColor, QTransform, QFont,
    QPalette, QBrush, QAction, QActionGroup
)
from PyQt6.QtCore import (
    Qt, QSize, QThread, pyqtSignal, QPoint, QRect, QObject, pyqtSlot, QTimer,
    QSettings, QByteArray, QUrl
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
# Import for plotting library.
import plotly.graph_objects as go


# --- Application Setup ---
# No longer need QWebEngineView flags as the component has been removed for stability.
print("✅ Application starting up.")


# --- GPU Acceleration Setup ---
# This block intelligently detects the presence and availability of CUDA-enabled
# GPU acceleration libraries (CuPy or PyTorch). It sets global flags that the
# rest of the application will use to dynamically switch between CPU and GPU processing.
GPU_AVAILABLE = False
GPU_BACKEND = None

try:
    # First, try to import and initialize CuPy.
    import cupy as cp
    if cp.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_BACKEND = 'cupy'
        print("🚀 CuPy GPU acceleration enabled! Correlation will be significantly faster.")
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    # If CuPy fails, try PyTorch as a fallback.
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_BACKEND = 'pytorch'
            print("🚀 PyTorch GPU acceleration enabled! Correlation will be significantly faster.")
    except (ImportError, RuntimeError):
        # If both fail, inform the user that processing will be CPU-bound.
        print("⚠️ No GPU acceleration backend (CuPy/PyTorch) found. Processing will use CPU.")
        print("   For massive performance gains, install CuPy or PyTorch with CUDA support.")


# --- Theming and Styling ---
class AppTheme:
    """
    A centralized class to manage application color schemes and stylesheets.
    This makes it trivial to switch between themes (e.g., Light and Dark)
    and ensures a consistent, modern, and readable look across all widgets.
    """
    # Color palette for the Dark theme.
    DARK = {
        "BACKGROUND": "#2c3e50", "CONTENT_BG": "#34495e", "BORDER": "#7f8c8d",
        "TEXT": "#ecf0f1", "TEXT_MUTED": "#bdc3c7", "PRIMARY": "#3498db",
        "PRIMARY_HOVER": "#5dade2", "PRIMARY_PRESSED": "#21618c", "SUCCESS": "#2ecc71",
        "WARNING": "#f1c40f", "ERROR": "#e74c3c", "WIDGET_BG": "#2c3e50",
        "WIDGET_BORDER": "#7f8c8d", "PROGRESS_CHUNK": "#3498db", "LOG_BG": "#212f3d",
        "LOG_TEXT": "#d5dbdb",
    }
    # Color palette for the Light theme.
    LIGHT = {
        "BACKGROUND": "#ecf0f1", "CONTENT_BG": "#ffffff", "BORDER": "#bdc3c7",
        "TEXT": "#2c3e50", "TEXT_MUTED": "#7f8c8d", "PRIMARY": "#3498db",
        "PRIMARY_HOVER": "#2980b9", "PRIMARY_PRESSED": "#21618c", "SUCCESS": "#27ae60",
        "WARNING": "#f39c12", "ERROR": "#c0392b", "WIDGET_BG": "#ffffff",
        "WIDGET_BORDER": "#bdc3c7", "PROGRESS_CHUNK": "#3498db", "LOG_BG": "#f8f9f9",
        "LOG_TEXT": "#34495e",
    }

    # The currently active theme is stored here. Defaults to Light.
    CURRENT = LIGHT

    @staticmethod
    def set_theme(theme_name):
        """Sets the global application theme."""
        if theme_name.lower() == 'dark':
            AppTheme.CURRENT = AppTheme.DARK
            print("🎨 Switched to Dark Theme.")
        else:
            AppTheme.CURRENT = AppTheme.LIGHT
            print("🎨 Switched to Light Theme.")

    @staticmethod
    def get_stylesheet():
        """Generates a full Qt Stylesheet (QSS) string from the current theme colors."""
        T = AppTheme.CURRENT
        return f"""
            /* General Window Styling */
            QMainWindow, QDialog {{ background-color: {T['BACKGROUND']}; }}
            /* GroupBox Styling */
            QGroupBox {{
                font-weight: bold; border: 1px solid {T['BORDER']}; border-radius: 8px;
                margin-top: 10px; background-color: {T['CONTENT_BG']}; color: {T['TEXT']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top center;
                padding: 0 10px; color: {T['PRIMARY']};
            }}
            /* Button Styling */
            QPushButton {{
                background-color: {T['PRIMARY']}; color: white; border: none; border-radius: 6px;
                padding: 8px 16px; font-weight: bold; min-height: 20px;
            }}
            QPushButton:hover {{ background-color: {T['PRIMARY_HOVER']}; }}
            QPushButton:pressed {{ background-color: {T['PRIMARY_PRESSED']}; }}
            QPushButton:disabled {{ background-color: {T['BORDER']}; color: {T['TEXT_MUTED']}; }}
            /* Input Widget Styling */
            QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid {T['WIDGET_BORDER']}; border-radius: 4px; padding: 4px;
                background-color: {T['WIDGET_BG']}; color: {T['TEXT']};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {T['PRIMARY']}; }}
            /* ScrollArea Styling */
            QScrollArea {{
                border: 1px solid {T['BORDER']}; border-radius: 8px;
                background-color: {T['CONTENT_BG']};
            }}
            /* Label and Text Styling */
            QLabel {{ color: {T['TEXT']}; }}
            QRadioButton, QCheckBox {{ color: {T['TEXT']}; spacing: 5px; }}
            /* Tab Widget Styling */
            QTabWidget::pane {{
                border: 1px solid {T['BORDER']}; border-radius: 8px;
                background-color: {T['CONTENT_BG']};
            }}
            QTabBar::tab {{
                background-color: {T['BACKGROUND']}; border: 1px solid {T['BORDER']};
                border-bottom: none; border-radius: 6px 6px 0 0; padding: 8px 16px;
                margin-right: 2px; color: {T['TEXT_MUTED']};
            }}
            QTabBar::tab:selected {{
                background-color: {T['CONTENT_BG']}; color: {T['PRIMARY']};
                border-bottom: 1px solid {T['CONTENT_BG']};
            }}
            QTabBar::tab:hover {{ background-color: {T['BORDER']}; color: {T['TEXT']}; }}
            /* Menu and Status Bar Styling */
            QStatusBar {{ color: {T['TEXT']}; }}
            QMenuBar {{ background-color: {T['CONTENT_BG']}; color: {T['TEXT']}; }}
            QMenuBar::item:selected {{ background-color: {T['PRIMARY']}; color: white; }}
            QMenu {{
                background-color: {T['CONTENT_BG']}; color: {T['TEXT']};
                border: 1px solid {T['BORDER']};
            }}
            QMenu::item:selected {{ background-color: {T['PRIMARY']}; color: white; }}
        """


# --- Logging Setup ---
class QtLogHandler(logging.Handler):
    """
    A custom logging handler that redirects Python's standard logging output
    to a PyQt signal. This allows log messages to be displayed in a QTextEdit
    widget within the GUI in real-time.
    """
    def __init__(self, parent_emitter):
        super().__init__()
        self.parent_emitter = parent_emitter

    def emit(self, record):
        """Formats the log record and emits it via the signal."""
        msg = self.format(record)
        self.parent_emitter.log_message.emit(msg)

class LogEmitter(QObject):
    """A simple QObject that contains the signal for log messages."""
    log_message = pyqtSignal(str)


# --- GPU Processing Wrapper ---
class GPUProcessor:
    """
    A static wrapper class that provides a unified, backend-agnostic API for
    GPU-accelerated operations. It abstracts away the differences between CuPy
    and PyTorch, allowing the main application logic to remain clean. If no GPU
    is available, it gracefully falls back to a CPU-based implementation.
    """
    @staticmethod
    def is_available():
        """Returns True if a supported GPU backend is active."""
        return GPU_AVAILABLE

    @staticmethod
    def get_backend():
        """Returns the name of the active GPU backend ('cupy' or 'pytorch')."""
        return GPU_BACKEND

    @staticmethod
    def to_gpu(array):
        """Moves a NumPy array from the CPU to the active GPU device."""
        if not GPU_AVAILABLE: return array
        try:
            if GPU_BACKEND == 'cupy':
                return cp.asarray(array)
            elif GPU_BACKEND == 'pytorch':
                return torch.tensor(array, device='cuda', dtype=torch.float32)
        except Exception as e:
            logging.warning(f"Failed to move array to GPU: {e}")
        return array

    @staticmethod
    def to_cpu(array):
        """Moves a GPU array back to a NumPy array on the CPU."""
        if not GPU_AVAILABLE or isinstance(array, np.ndarray): return array
        try:
            if GPU_BACKEND == 'cupy':
                return cp.asnumpy(array)
            elif GPU_BACKEND == 'pytorch':
                return array.cpu().numpy()
        except Exception as e:
            logging.warning(f"Failed to move array to CPU: {e}")
        return array

    @staticmethod
    def correlate_2d(img1, img2):
        """
        Performs normalized 2D cross-correlation. This is the core accelerated
        function. It dispatches the task to the appropriate GPU backend or the
        CPU fallback if necessary.
        """
        if not GPU_AVAILABLE:
            return GPUProcessor._cpu_correlate_2d(img1, img2)
        try:
            if GPU_BACKEND == 'cupy':
                return GPUProcessor._cupy_correlate_2d(img1, img2)
            elif GPU_BACKEND == 'pytorch':
                return GPUProcessor._pytorch_correlate_2d(img1, img2)
        except Exception as e:
            logging.error(f"GPU correlation failed unexpectedly: {e}. Falling back to CPU for this pair.")
            return GPUProcessor._cpu_correlate_2d(img1, img2)

    @staticmethod
    def _cupy_correlate_2d(img1, img2):
        """CuPy-based 2D cross-correlation using the highly efficient FFT method."""
        # Transfer images to GPU memory
        gpu_img1 = cp.asarray(img1, dtype=cp.float32)
        gpu_img2 = cp.asarray(img2, dtype=cp.float32)

        # Normalize images on the GPU to have zero mean and unit variance
        gpu_img1 = (gpu_img1 - cp.mean(gpu_img1)) / (cp.std(gpu_img1) + 1e-7)
        gpu_img2 = (gpu_img2 - cp.mean(gpu_img2)) / (cp.std(gpu_img2) + 1e-7)

        # The Cross-Correlation Theorem states that correlation in the spatial domain
        # is equivalent to element-wise multiplication in the frequency domain.
        # This is significantly faster for large images than direct convolution.
        f1 = cp.fft.fft2(gpu_img1)
        # The kernel must be flipped for correlation (as opposed to convolution).
        f2 = cp.fft.fft2(cp.flipud(cp.fliplr(gpu_img2)))

        # Perform the multiplication and transform back to the spatial domain.
        correlation = cp.fft.ifft2(f1 * f2)
        correlation = cp.abs(correlation)

        # Find the maximum correlation peak and return it as a float.
        max_corr = float(cp.asnumpy(cp.max(correlation)))
        return max_corr

    @staticmethod
    def _pytorch_correlate_2d(img1, img2):
        """PyTorch-based 2D cross-correlation using the `conv2d` function."""
        device = torch.device('cuda')
        # Transfer images to GPU tensor
        tensor1 = torch.tensor(img1, device=device, dtype=torch.float32)
        tensor2 = torch.tensor(img2, device=device, dtype=torch.float32)

        # Normalize tensors on the GPU
        tensor1 = (tensor1 - tensor1.mean()) / (tensor1.std() + 1e-7)
        tensor2 = (tensor2 - tensor2.mean()) / (tensor2.std() + 1e-7)

        # `conv2d` requires tensors in (Batch, Channel, Height, Width) format.
        tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
        tensor2 = tensor2.unsqueeze(0).unsqueeze(0)

        # PyTorch's `conv2d` performs convolution. To achieve correlation,
        # we must manually flip the kernel (tensor2) in its spatial dimensions.
        tensor2_flipped = torch.flip(tensor2, [2, 3])

        # Apply padding to ensure the output correlation map has the same
        # dimensions as the input, finding the peak at the center.
        pad_h = (tensor1.shape[2] - 1) // 2
        pad_w = (tensor1.shape[3] - 1) // 2

        correlation = torch.nn.functional.conv2d(tensor1, tensor2_flipped, padding=(pad_h, pad_w))
        max_corr = float(torch.max(correlation).cpu().item())
        return max_corr

    @staticmethod
    def _cpu_correlate_2d(img1, img2):
        """CPU fallback correlation using OpenCV's highly optimized `matchTemplate`."""
        # Ensure images are float32, as required by the correlation method.
        img1_32 = img1.astype(np.float32)
        img2_32 = img2.astype(np.float32)
        # TM_CCOEFF_NORMED provides a correlation score between -1 and 1.
        result = cv2.matchTemplate(img1_32, img2_32, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)


# --- GUI Widgets ---

class LoadingOverlay(QWidget):
    """
    A full-panel, semi-transparent loading animation widget. This provides a much
    better user experience by clearly indicating a background task is running
    while preventing interaction with the underlying UI.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make the widget background transparent so we can draw our own.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)

    def _update_animation(self):
        """Updates the rotation angle for the spinning arc."""
        self.angle = (self.angle + 5) % 360
        self.update() # Triggers a repaint

    def paintEvent(self, event):
        """Paints a modern, spinning arc animation in the center of the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a semi-transparent background to dim the underlying content.
        bg_color = QColor(AppTheme.CURRENT['CONTENT_BG'])
        bg_color.setAlpha(200) # 0 (transparent) to 255 (opaque)
        painter.fillRect(self.rect(), bg_color)

        # Define the geometry for the spinning arc.
        side = min(self.width(), self.height())
        rect = QRect(0, 0, int(side * 0.2), int(side * 0.2))
        rect.moveCenter(self.rect().center())

        # Configure the pen for drawing the arc.
        pen = QPen(QColor(AppTheme.CURRENT['PRIMARY']), 12)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # Draw the arc. Angles are specified in 1/16th of a degree.
        painter.drawArc(rect, self.angle * 16, 120 * 16)

    def start_animation(self):
        """Starts the animation timer and makes the overlay visible."""
        self.angle = 0
        self.timer.start(20) # Target ~50 FPS
        self.setVisible(True)
        self.raise_() # Ensure it's on top of other widgets

    def stop_animation(self):
        """Stops the animation and hides the overlay."""
        self.timer.stop()
        self.setVisible(False)


class SystemMonitor(QWidget):
    """A status bar widget to monitor system resources (CPU, RAM, GPU) in real-time."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        # Create labels for each metric.
        self.cpu_label = QLabel("CPU: 0%")
        self.ram_label = QLabel("RAM: 0%")
        self.gpu_label = QLabel("GPU: N/A")
        self.gpu_mem_label = QLabel("VRAM: N/A")

        # Create a label to show which processing backend is active.
        backend_text = f"🚀 {GPU_BACKEND.upper()}" if GPU_AVAILABLE else "⚙️ CPU"
        self.backend_label = QLabel(backend_text)
        
        T = AppTheme.CURRENT
        gpu_color = T['SUCCESS'] if GPU_AVAILABLE else T['ERROR']
        self.backend_label.setStyleSheet(f"color: {gpu_color}; font-weight: bold;")

        # Add widgets to the layout.
        layout.addWidget(self.backend_label)
        layout.addWidget(self._create_separator())
        layout.addWidget(self.cpu_label)
        layout.addWidget(self._create_separator())
        layout.addWidget(self.ram_label)
        if GPUtil and GPU_AVAILABLE:
            layout.addWidget(self._create_separator())
            layout.addWidget(self.gpu_label)
            layout.addWidget(self._create_separator())
            layout.addWidget(self.gpu_mem_label)
        layout.addStretch()

        self.setStyleSheet(f"QLabel {{ color: {T['TEXT_MUTED']}; font-size: 11px; }}")

    def _create_separator(self):
        """Helper to create a styled vertical bar separator."""
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {AppTheme.CURRENT['BORDER']}; margin: 0 5px;")
        return sep

    def start_monitoring(self):
        """Starts the periodic updates."""
        self.update_stats()
        self.timer.start(2000)  # Update every 2 seconds

    def stop_monitoring(self):
        """Stops the periodic updates."""
        self.timer.stop()

    def update_stats(self):
        """Fetches and displays the latest system resource usage."""
        # Get CPU and RAM usage from psutil.
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
        self.ram_label.setText(f"RAM: {ram_percent:.1f}%")

        # Get GPU stats from GPUtil if available.
        if GPUtil and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_load = gpu.load * 100
                    vram_load = gpu.memoryUtil * 100
                    self.gpu_label.setText(f"GPU: {gpu_load:.1f}%")
                    self.gpu_mem_label.setText(f"VRAM: {vram_load:.1f}%")
                    
                    # Color-code the GPU load for at-a-glance status.
                    T = AppTheme.CURRENT
                    color = T['ERROR'] if gpu_load > 80 else T['WARNING'] if gpu_load > 50 else T['SUCCESS']
                    self.gpu_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
                else:
                    self.gpu_label.setText("GPU: N/A")
                    self.gpu_mem_label.setText("VRAM: N/A")
            except Exception as e:
                logging.warning(f"Could not poll GPU stats: {e}")
                self.gpu_label.setText("GPU: Error")
                self.gpu_mem_label.setText("VRAM: Error")


class QualityMetrics:
    """A collection of static methods to calculate image quality metrics."""
    @staticmethod
    def calculate_texture_quality(img):
        """Calculates texture quality based on the standard deviation of pixel intensities."""
        if img is None: return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        std_dev = np.std(gray.astype(np.float32))
        return np.clip(std_dev / 128.0, 0, 1.0) # Normalize to a 0-1 range

    @staticmethod
    def calculate_edge_quality(edge_img):
        """Calculates edge quality based on the density of detected edges."""
        if edge_img is None: return 0.0
        gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY) if len(edge_img.shape) == 3 else edge_img
        # Calculate the percentage of pixels that are considered edges.
        edge_density = np.sum(gray > 50) / gray.size
        return np.clip(edge_density * 10, 0, 1.0) # Heuristic normalization


class Subsquare:
    """A data class to hold all information related to a single detected particle."""
    def __init__(self, img_id, grid_id, path, img, bbox, proc=None, edge=None):
        self.image_id = img_id  # The index of the source image file.
        self.grid_id = grid_id  # A human-readable ID like "A1", "B3", etc.
        self.original_image_path = path # Full path to the source image.
        self.original_subsquare_img = img # The raw pixel data of the particle.
        self.bbox = bbox # The bounding box (x, y, w, h) in the original image.
        self.processed_img = proc # The preprocessed image used for texture correlation.
        self.edge_img = edge # The edge-detected image used for edge correlation.
        
        # NEW: Smart naming for files like 'J1024_...'
        base_name = os.path.basename(path).split('.')[0]
        match = re.match(r"(J\d+)_", base_name)
        if match:
            short_name = match.group(1)
            self.unique_id = f"{short_name}_{grid_id}"
        else:
            self.unique_id = f"{base_name}_{grid_id}"

        self.texture_quality = 0.0
        self.edge_quality = 0.0
        self.overall_quality = 0.0

    def calculate_quality_metrics(self):
        """Calculates and stores the quality scores for this subsquare."""
        if self.processed_img is not None:
            self.texture_quality = QualityMetrics.calculate_texture_quality(self.processed_img)
        if self.edge_img is not None:
            self.edge_quality = QualityMetrics.calculate_edge_quality(self.edge_img)
        self.overall_quality = (self.texture_quality + self.edge_quality) / 2.0

    def to_qpixmap(self, img_data=None, size=100):
        """Converts image data (NumPy array) to a QPixmap for display in the GUI."""
        img = img_data if img_data is not None else self.processed_img
        if img is None:
            pixmap = QPixmap(size, size)
            pixmap.fill(QColor(AppTheme.CURRENT['BORDER']))
            return pixmap

        # Normalize and convert format for QImage.
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        # Scale the pixmap smoothly while preserving its aspect ratio.
        return pixmap.scaled(QSize(size, size), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class EnhancedProgressBar(QWidget):
    """A custom progress bar widget that includes a secondary label for detailed text."""
    def __init__(self, parent=None):
        super().__init__(parent)
        T = AppTheme.CURRENT
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {T['BORDER']}; border-radius: 8px; text-align: center;
                font-weight: bold; background-color: {T['CONTENT_BG']}; color: {T['TEXT']};
            }}
            QProgressBar::chunk {{
                background-color: {T['PROGRESS_CHUNK']}; border-radius: 7px;
            }}
        """)

        self.detail_label = QLabel("Idle")
        self.detail_label.setStyleSheet(f"font-size: 11px; color: {T['TEXT_MUTED']};")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)

    def setValue(self, value):
        self.progress_bar.setValue(value)

    def setRange(self, min_val, max_val):
        self.progress_bar.setRange(min_val, max_val)

    def setDetailText(self, text):
        self.detail_label.setText(text)


# --- Processing Threads ---

class ImageProcessor(QThread):
    """
    Worker thread for the entire particle detection and preprocessing pipeline.
    Running this in a separate thread is essential to prevent the GUI from
    freezing during these potentially long-running operations.
    """
    # Signals to communicate progress and results back to the main GUI thread.
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_detection = pyqtSignal(list)

    def __init__(self, paths, params, ref_square=None):
        super().__init__()
        self.image_paths = paths
        self.params = params
        self.ref_square = ref_square # This holds the reference selection info.
        self.all_subsquares = []
        self._is_running = True # A flag to allow for graceful interruption.

    def stop(self):
        """Requests the thread to stop processing."""
        self._is_running = False
        logging.info("Image processing thread stop requested.")

    def run(self):
        """The main entry point for the thread's execution."""
        try:
            logging.info(f"Image detection thread started. Mode: {'Reference' if self.ref_square else 'Automatic'}")
            # Dispatch to the correct detection method based on user selection.
            if self.ref_square:
                self.detect_using_reference()
            else:
                self.detect_automatically()

            if not self._is_running:
                logging.info("Detection was aborted by user.")
                self.finished_detection.emit([])
                return

            # After detection, preprocess all found particles.
            self.preprocess_all()
            
            if not self._is_running:
                logging.info("Preprocessing was aborted by user.")
                self.finished_detection.emit([])
                return

            # Finally, calculate quality metrics for each preprocessed particle.
            self.progress.emit("🔬 Calculating quality metrics...")
            if self.all_subsquares:
                for i, ss in enumerate(self.all_subsquares):
                    if not self._is_running: break
                    ss.calculate_quality_metrics()
                    # Use the last 5% of the progress bar for this step.
                    self.progress_value.emit(95 + int(((i + 1) / len(self.all_subsquares)) * 5))

            logging.info(f"Detection pipeline finished. Found {len(self.all_subsquares)} subsquares.")
            self.finished_detection.emit(self.all_subsquares)
        except Exception as e:
            logging.error(f"Critical error in detection thread: {e}", exc_info=True)
            self.progress.emit(f"❌ Error: {e}")
            self.finished_detection.emit([])

    def detect_using_reference(self):
        """Performs particle detection by template matching against a user-defined reference."""
        ref_path, rx, ry, rw, rh = self.ref_square
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            logging.error(f"Could not load reference image: {ref_path}")
            return
            
        # Extract the reference template from the source image.
        ref_template = ref_img[ry:ry+rh, rx:rx+rw]
        grid_map = self._create_grid_map()
        
        total_images = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if not self._is_running: return
                
            self.progress.emit(f"🔍 Matching in {os.path.basename(path)}...")
            # Use the first half of the progress bar for matching.
            self.progress_value.emit(int(((i+1) / total_images) * 50))
            
            img = cv2.imread(path)
            if img is not None:
                matches = self._template_match(img, ref_template, path, i, grid_map.get(i))
                self.all_subsquares.extend(matches)
                logging.info(f"Found {len(matches)} matches in {os.path.basename(path)}")

    def _template_match(self, image, template, path, img_id, grid):
        """Core template matching logic, including rotation invariance."""
        squares = []
        g_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = g_template.shape
        
        all_matches = []
        # Check for matches at different rotation angles.
        angles = np.arange(0, 360, 30) # Check every 30 degrees.
        
        for angle in angles:
            if not self._is_running: return []
                
            # Rotate the template.
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated_template = cv2.warpAffine(g_template, M, (w, h))
            
            # Perform the template matching.
            res = cv2.matchTemplate(g_img, rotated_template, cv2.TM_CCOEFF_NORMED)
            # Find all locations where the correlation score exceeds the threshold.
            locs = np.where(res >= self.params['ref_threshold'])
            
            for pt in zip(*locs[::-1]):
                all_matches.append((pt[0], pt[1], w, h, res[pt[1], pt[0]]))
                
        # Sort all found matches by their score in descending order.
        all_matches.sort(key=lambda x: x[4], reverse=True)
        
        # Filter out overlapping matches (non-maximum suppression).
        filtered_squares = []
        for x, y, wm, hm, score in all_matches:
            # Check if this new match significantly overlaps with an already accepted one.
            if not any(self._calc_overlap((x, y, wm, hm), s.bbox) > 0.4 for s in filtered_squares):
                subsquare = Subsquare(
                    img_id, self._get_grid_id(x, y, grid), path, 
                    image[y:y+hm, x:x+wm], (x, y, wm, hm)
                )
                filtered_squares.append(subsquare)
                
        return filtered_squares
        
    def detect_automatically(self):
        """Performs particle detection using contour analysis."""
        grid_map = self._create_grid_map()
        total_images = len(self.image_paths)
        
        for i, path in enumerate(self.image_paths):
            if not self._is_running: return
            self.progress.emit(f"🔍 Detecting in {os.path.basename(path)}...")
            # Use the first half of the progress bar for detection.
            self.progress_value.emit(int(((i + 1) / total_images) * 50))
            logging.info(f"Auto-detecting squares in: {path}")

            detected = self._detect_squares(path, i, grid_map.get(i))
            self.all_subsquares.extend(detected)
            logging.info(f"Found {len(detected)} potential squares in {os.path.basename(path)}")
    
    def preprocess_all(self):
        """Iterates through all detected subsquares and applies preprocessing steps."""
        total = len(self.all_subsquares)
        if total == 0: return
        
        self.progress.emit("⚙️ Preprocessing squares...")
        logging.info(f"Preprocessing {total} squares.")
        
        for i, ss in enumerate(self.all_subsquares):
            if not self._is_running: return
            # Use the second half of the progress bar for preprocessing.
            progress_val = 50 + int(((i + 1) / total) * 45)
            self.progress_value.emit(progress_val)
            if i % 20 == 0:
                self.progress.emit(f"⚙️ Preprocessing square {i+1}/{total}...")
            self._preprocess_subsquare(ss)

    def _create_grid_map(self):
        """Creates a mapping of image index to grid dimensions for naming particles."""
        grid_map = {}
        for i, path in enumerate(self.image_paths):
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    grid_map[i] = {'size': (w / 10, h / 10)} # Assume a 10x10 grid.
            except Exception as e:
                logging.error(f"Could not read image for grid map: {path}, {e}")
        return grid_map

    def _get_grid_id(self, x, y, grid):
        """Converts (x, y) coordinates to a grid ID like 'A1'."""
        if not grid or 'size' not in grid or grid['size'][0] == 0 or grid['size'][1] == 0:
            return "A1"
        col = int(x // grid['size'][0])
        row = int(y // grid['size'][1])
        return f"{chr(ord('A') + min(col, 25))}{row + 1}"

    def _detect_squares(self, path, img_id, grid):
        """Core automatic detection logic using OpenCV's contour finding."""
        try:
            img = cv2.imread(path)
            if img is None:
                logging.error(f"Failed to load image for detection: {path}")
                return []

            # Image processing pipeline to isolate potential particles.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = []
            min_size = self.params['min_square_size']
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_size * min_size: continue

                # Approximate the contour to a polygon.
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Filter based on size, aspect ratio, and "squareness".
                if 0.7 <= aspect_ratio <= 1.4 and w >= min_size and h >= min_size:
                    extent = area / (w * h)
                    if extent > 0.5: # Check how much of the bounding box is filled.
                        subsquare = Subsquare(
                            img_id, self._get_grid_id(x, y, grid), path,
                            img[y:y+h, x:x+w], (x, y, w, h)
                        )
                        detected.append(subsquare)
            
            return self._remove_duplicates(detected)
        except Exception as e:
            logging.error(f"Error during automatic square detection in {path}: {e}", exc_info=True)
            return []

    def _remove_duplicates(self, squares):
        """Filters a list of squares to remove highly overlapping duplicates."""
        # Sort by size, so we keep the largest of a cluster.
        squares.sort(key=lambda s: s.bbox[2] * s.bbox[3], reverse=True)
        filtered = []
        for square in squares:
            overlap = any(self._calc_overlap(square.bbox, accepted.bbox) > 0.5 for accepted in filtered)
            if not overlap:
                filtered.append(square)
        return filtered

    def _calc_overlap(self, bbox1, bbox2):
        """Calculates the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        x_left, y_top = max(x1, x2), max(y1, y2)
        x_right, y_bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if x_right <= x_left or y_bottom <= y_top: return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0

    def _preprocess_subsquare(self, ss):
        """Applies the standard preprocessing pipeline to a single subsquare."""
        target_size = self.params['target_subsquare_size']
        img = ss.original_subsquare_img
        if img is None or img.size == 0: return

        # Resize with aspect ratio preservation.
        h, w = img.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad to the target size, centering the image.
        padded = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)
        y_offset, x_offset = (target_size[1] - new_h) // 2, (target_size[0] - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply CLAHE for localized contrast enhancement.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        ss.processed_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # 2. Perform Canny edge detection for structural analysis.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny_edges = cv2.Canny(blurred, 50, 150)
        ss.edge_img = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)


class EnhancedCorrelationProcessor(QThread):
    """
    Worker thread for the main correlation analysis. It uses a ThreadPoolExecutor
    to parallelize the processing of image pairs, leveraging the GPUProcessor
    for acceleration.
    """
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_correlation = pyqtSignal(object)

    def __init__(self, subsquares, params):
        super().__init__()
        self.subsquares = subsquares
        self.params = params
        self.data = None
        self._is_running = True

    def stop(self):
        """Requests the thread to stop processing."""
        self._is_running = False
        logging.info("Correlation processing thread stop requested.")

    def run(self):
        """The main entry point for the correlation thread."""
        
        # --- NEW: Pair Generation Logic ---
        pairs = self._generate_pairs()
        if not pairs:
            logging.warning("No pairs generated for correlation based on the selected mode.")
            self.finished_correlation.emit(None)
            return

        n = len(self.subsquares)
        self.data = np.empty((n, n), dtype=object)
        total_pairs = len(pairs)
        
        use_gpu = GPUProcessor.is_available()
        prefix = "🚀" if use_gpu else "⚙️"
        msg = f"{prefix} Starting {'GPU' if use_gpu else 'CPU'} correlation for {total_pairs} pairs..."
        self.progress.emit(msg)
        logging.info(msg)

        # Process the pairs in a threaded manner.
        self._process_pairs_threaded(pairs, use_gpu)
        
        if self._is_running:
            logging.info("Correlation analysis finished successfully.")
            self.finished_correlation.emit(self.data)
        else:
            logging.info("Correlation analysis was aborted by user.")
            self.finished_correlation.emit(None)

    def _generate_pairs(self):
        """Generates the list of pairs to compare based on the selected mode."""
        n = len(self.subsquares)
        mode = self.params.get("comparison_mode")
        source_image_path = self.params.get("source_image_path")

        if mode == "All vs. All":
            logging.info("Generating pairs for 'All vs. All' mode.")
            # Create a list of all unique pairs (i, j) where i <= j
            return [(i, j) for i in range(n) for j in range(i, n)]
        
        elif mode == "One Image vs. Others":
            if not source_image_path:
                logging.error("'One Image vs. Others' mode selected but no source image was provided.")
                return []
            
            logging.info(f"Generating pairs for 'One Image vs. Others' mode. Source: {os.path.basename(source_image_path)}")
            source_indices = [i for i, s in enumerate(self.subsquares) if s.original_image_path == source_image_path]
            other_indices = [i for i, s in enumerate(self.subsquares) if s.original_image_path != source_image_path]
            
            # Also include comparisons within the source image itself
            source_vs_source_pairs = [(i, j) for i_idx, i in enumerate(source_indices) for j in source_indices[i_idx:]]
            source_vs_other_pairs = [(i, j) for i in source_indices for j in other_indices]
            
            return source_vs_source_pairs + source_vs_other_pairs
        
        return []

    def _process_pairs_threaded(self, pairs, use_gpu):
        """Manages a thread pool to process correlation pairs in parallel."""
        total_pairs = len(pairs)
        # Use a higher number of workers to try and feed the GPU faster.
        max_workers = 8 if use_gpu else min(16, os.cpu_count() + 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pairs to the thread pool for execution.
            future_to_pair = {executor.submit(self._process_pair, pair, use_gpu): pair for pair in pairs}
            
            completed_pairs = 0
            for future in as_completed(future_to_pair):
                if not self._is_running:
                    # If a stop is requested, attempt to cancel remaining futures.
                    for f in future_to_pair: f.cancel()
                    break
                
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        i, j = pair
                        # The result matrix is symmetric.
                        self.data[i, j] = self.data[j, i] = result
                except Exception as e:
                    logging.error(f"Error processing pair {pair}: {e}", exc_info=True)

                completed_pairs += 1
                progress_val = int((completed_pairs / total_pairs) * 100)
                self.progress_value.emit(progress_val)
                # Update the detail text periodically to avoid overwhelming the GUI event loop.
                if completed_pairs % 50 == 0 or completed_pairs == total_pairs:
                    prefix = "🚀" if use_gpu else "⚙️"
                    self.progress.emit(f"{prefix} Correlated pair {completed_pairs}/{total_pairs}")

    def _process_pair(self, pair, use_gpu):
        """Processes a single pair of subsquares to calculate their correlation."""
        i, j = pair
        # The correlation of an image with itself is always 1.
        if i == j:
            return (1.0, 0.0, {'texture_weight': 0.5, 'edge_weight': 0.5, 'quality_score': 1.0, 'gpu_accelerated': use_gpu})

        s1, s2 = self.subsquares[i], self.subsquares[j]

        # Dynamically weight the final score based on the quality of the inputs.
        # Pairs with high texture get a higher weight for texture correlation, and vice-versa.
        texture_quality_avg = (s1.texture_quality + s2.texture_quality) / 2.0
        edge_quality_avg = (s1.edge_quality + s2.edge_quality) / 2.0
        total_quality = texture_quality_avg + edge_quality_avg
        texture_weight = (texture_quality_avg / total_quality) if total_quality > 0 else 0.5
        edge_weight = 1.0 - texture_weight

        # Calculate correlation for both texture and edges.
        texture_score, texture_angle = self._calculate_correlation_with_rotation(s1.processed_img, s2.processed_img, use_gpu)
        edge_score, edge_angle = self._calculate_correlation_with_rotation(s1.edge_img, s2.edge_img, use_gpu)

        # Combine scores using the dynamic weights.
        final_score = texture_score * texture_weight + edge_score * edge_weight
        # The final angle is taken from the component (texture or edge) with the higher score.
        final_angle = texture_angle if texture_score > edge_score else edge_angle

        metadata = {
            'texture_weight': texture_weight, 'edge_weight': edge_weight,
            'texture_score': texture_score, 'edge_score': edge_score,
            'quality_score': (s1.overall_quality + s2.overall_quality) / 2.0,
            'gpu_accelerated': use_gpu
        }
        return (final_score, final_angle, metadata)

    def _calculate_correlation_with_rotation(self, img1, img2, use_gpu):
        """Finds the best correlation score between two images across a range of rotations."""
        if img1 is None or img2 is None: return 0.0, 0.0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        max_correlation, best_angle = -1.0, 0
        
        rotation_step = self.params.get("rotation_step", 15)

        # Check a range of angles to find the best rotational alignment.
        for angle in range(0, 360, rotation_step):
            if not self._is_running: break
            center = (gray2.shape[1] // 2, gray2.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Rotate the second image.
            rotated = cv2.warpAffine(gray2, M, (gray2.shape[1], gray2.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=128)
            
            # Perform the correlation using the GPUProcessor.
            correlation = GPUProcessor.correlate_2d(gray1, rotated)
            if correlation > max_correlation:
                max_correlation = correlation
                best_angle = angle
        return max_correlation, best_angle


# --- Dialogs and Viewers ---

class ZoomableViewer(QDialog):
    """Dialog for comparing two subsquares with zoom/pan."""
    def __init__(self, s1, s2, score, angle, metadata=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 Pairwise Comparison")
        self.setMinimumSize(600, 400)
        self.resize(800, 600)
        
        self.s1, self.s2 = s1, s2
        self.angle = angle
        self.zoom = 1.0
        self.metadata = metadata or {}
        
        self._setup_ui(score)
        self.display_pair()

    def _setup_ui(self, score):
        T = AppTheme.CURRENT
        self.setStyleSheet(f"background-color: {T['BACKGROUND']}; color: {T['TEXT']};")
        layout = QVBoxLayout(self)
        
        # Info panel
        info_widget = QFrame()
        info_widget.setFrameShape(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout(info_widget)
        score_label = QLabel(f"📊 <b>Score: {score:.4f}</b>")
        score_label.setStyleSheet(f"font-size: 14px; color: {T['SUCCESS']};")
        gpu_indicator = "🚀" if self.metadata.get('gpu_accelerated', False) else "⚙️"
        meta_text = f"T:{self.metadata.get('texture_score', 0):.3f} E:{self.metadata.get('edge_score', 0):.3f} Q:{self.metadata.get('quality_score', 0):.3f}"
        meta_label = QLabel(f"{gpu_indicator} {meta_text}")
        comp_label = QLabel(f"🔄 {self.s1.unique_id} vs {self.s2.unique_id} (↻{self.angle}°)")
        info_layout.addWidget(score_label)
        info_layout.addWidget(meta_label)
        info_layout.addStretch()
        info_layout.addWidget(comp_label)
        layout.addWidget(info_widget)
        
        self.viewer = QLabel()
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setStyleSheet(f"border: 2px solid {T['BORDER']}; border-radius: 8px;")
        layout.addWidget(self.viewer, 1) # Give stretch factor
        
        # Controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self._reset_zoom)
        controls_layout.addWidget(QLabel("🔍 Zoom:"))
        controls_layout.addWidget(self.zoom_slider)
        controls_layout.addWidget(reset_btn)
        layout.addWidget(controls)

    def _on_zoom_changed(self, value):
        self.zoom = value / 100.0
        self.display_pair()

    def _reset_zoom(self):
        self.zoom = 1.0
        self.zoom_slider.setValue(100)
        self.display_pair()

    def display_pair(self):
        if self.s1.processed_img is None or self.s2.processed_img is None: return
        img1, img2 = self.s1.processed_img.copy(), self.s2.processed_img.copy()
        center = (img2.shape[1] // 2, img2.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
        
        combined = cv2.hconcat([img1, rotated])
        h, w = combined.shape[:2]
        cv2.line(combined, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        
        q_img = QImage(combined.data, w, h, w * 3, QImage.Format.Format_BGR888)
        scaled_size = QSize(int(w * self.zoom), int(h * self.zoom))
        pixmap = QPixmap.fromImage(q_img).scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.viewer.setPixmap(pixmap)


class TopPairsGallery(QDialog):
    """Dialog to display a gallery of the most highly correlated pairs."""
    def __init__(self, data, subsquares, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🏆 Top Correlated Pairs")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        self.data = data
        self.subsquares = subsquares
        
        self._setup_ui()
        self.update_display()

    def _setup_ui(self):
        T = AppTheme.CURRENT
        self.setStyleSheet(f"background-color: {T['BACKGROUND']}; color: {T['TEXT']};")
        layout = QVBoxLayout(self)
        
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(QLabel("📈 Show top:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(10, 1000); self.count_spin.setValue(50)
        self.count_spin.valueChanged.connect(self.update_display)
        controls_layout.addWidget(self.count_spin)
        
        controls_layout.addWidget(QLabel("🎯 Min score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0); self.min_score_spin.setValue(0.0); self.min_score_spin.setSingleStep(0.05)
        self.min_score_spin.valueChanged.connect(self.update_display)
        controls_layout.addWidget(self.min_score_spin)
        
        self.quality_check = QCheckBox("Sort by Quality-Score")
        self.quality_check.toggled.connect(self.update_display)
        controls_layout.addWidget(self.quality_check)
        controls_layout.addStretch()
        self.stats_label = QLabel()
        controls_layout.addWidget(self.stats_label)
        layout.addWidget(controls)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.scroll_area.setWidget(self.gallery_widget)
        layout.addWidget(self.scroll_area)

    def update_display(self):
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
                
        n = len(self.subsquares)
        min_score = self.min_score_spin.value()
        
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                # Use self.data here, which is the correlation matrix
                item = self.data[i, j]
                if item:
                    score, angle, meta = item
                    quality = meta.get('quality_score', 0)
                    if score >= min_score:
                        pairs.append((i, j, score, angle, quality, meta))
        
        if self.quality_check.isChecked():
            pairs.sort(key=lambda x: x[2] * x[4], reverse=True) # Score * Quality
        else:
            pairs.sort(key=lambda x: x[2], reverse=True) # Score only
            
        pairs = pairs[:self.count_spin.value()]
        
        if pairs:
            avg_score = np.mean([p[2] for p in pairs])
            self.stats_label.setText(f"📊 Showing {len(pairs)} pairs | Avg Score: {avg_score:.3f}")
        else:
            self.stats_label.setText("📊 No pairs match criteria")
            
        for rank, (i, j, score, angle, quality, meta) in enumerate(pairs, 1):
            pair_widget = self._create_pair_widget(i, j, score, angle, quality, meta, rank)
            self.gallery_layout.addWidget(pair_widget)
        self.gallery_layout.addStretch()

    def _create_pair_widget(self, i, j, score, angle, quality, meta, rank):
        T = AppTheme.CURRENT
        widget = QFrame()
        widget.setFrameShape(QFrame.Shape.StyledPanel)
        widget.setStyleSheet(f"""
            QFrame {{ border: 1px solid {T['BORDER']}; border-radius: 8px; margin: 5px; padding: 10px; background-color: {T['CONTENT_BG']}; }}
            QFrame:hover {{ border-color: {T['PRIMARY']}; }}
        """)
        layout = QHBoxLayout(widget)
        
        rank_label = QLabel(f"#{rank}")
        rank_label.setStyleSheet(f"background-color: {T['PRIMARY']}; color: white; border-radius: 15px; padding: 5px; font-weight: bold;")
        rank_label.setFixedSize(40, 40); rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        s1, s2 = self.subsquares[i], self.subsquares[j]
        pixmap = self._create_comparison_pixmap(s1, s2, angle)
        preview_label = QLabel()
        preview_label.setPixmap(pixmap)
        preview_label.setStyleSheet(f"border: 1px solid {T['BORDER']}; border-radius: 4px;")
        
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"<b>{s1.unique_id} ↔ {s2.unique_id}</b>"))
        info_layout.addWidget(QLabel(f"📊 Score: <b>{score:.4f}</b>"))
        # FIX: Display the rotational information that was previously missing.
        info_layout.addWidget(QLabel(f"🔄 Rotation: {angle}°"))
        info_layout.addWidget(QLabel(f"⭐ Quality: {quality:.3f}"))
        info_layout.addStretch()
        
        view_btn = QPushButton("🔍 View")
        view_btn.clicked.connect(lambda chk, i=i, j=j, s=score, a=angle, m=meta: self._view_pair(i, j, s, a, m))
        
        layout.addWidget(rank_label)
        layout.addWidget(preview_label)
        layout.addLayout(info_layout, 1)
        layout.addWidget(view_btn)
        return widget
    
    def _create_comparison_pixmap(self, s1, s2, angle):
        # FIX: Use OpenCV for rotation to guarantee correct visual alignment.
        img1_data = s1.processed_img
        img2_data = s2.processed_img
        
        if img1_data is None or img2_data is None:
            return QPixmap(210, 100)
            
        # Rotate the second image using OpenCV
        center = (img2_data.shape[1] // 2, img2_data.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img2_data = cv2.warpAffine(img2_data, M, (img2_data.shape[1], img2_data.shape[0]))
        
        # Combine the two images side-by-side
        combined_img = cv2.hconcat([img1_data, rotated_img2_data])
        
        # Convert the combined OpenCV image to a QPixmap
        h, w, ch = combined_img.shape
        bytes_per_line = ch * w
        q_img = QImage(combined_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        return pixmap.scaled(QSize(200, 100), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def _view_pair(self, i, j, score, angle, metadata):
        s1, s2 = self.subsquares[i], self.subsquares[j]
        viewer = ZoomableViewer(s1, s2, score, angle, metadata, self)
        viewer.exec()


class InteractiveImageViewer(QLabel):
    """A label widget that allows interactive selection, panning, and zooming."""
    square_selected = pyqtSignal(int, int, int, int)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap = None
        self.start_pos = None
        self.end_pos = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.pan_start = QPoint()
        T = AppTheme.CURRENT
        self.setStyleSheet(f"border: 2px solid {T['BORDER']}; border-radius: 8px; background-color: {T['CONTENT_BG']};")
        self.setToolTip("Left-click & drag: Select area\nRight-click & drag: Pan\nScroll wheel: Zoom\nDouble-click: Reset")
        
    def set_image(self, path):
        self._pixmap = QPixmap(path)
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()
        
    def paintEvent(self, event):
        if not self._pixmap:
            super().paintEvent(event)
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        scaled_size = self._pixmap.size() * self.zoom_factor
        pos = QPoint((self.width() - scaled_size.width()) // 2 + self.pan_offset.x(), (self.height() - scaled_size.height()) // 2 + self.pan_offset.y())
        
        painter.drawPixmap(QRect(pos, scaled_size), self._pixmap, self._pixmap.rect())
        
        if self.start_pos and self.end_pos:
            painter.setPen(QPen(QColor(AppTheme.CURRENT['ERROR']), 3))
            painter.setBrush(QColor(231, 76, 60, 50))
            painter.drawRect(QRect(self.start_pos, self.end_pos).normalized())
            
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = self.end_pos = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            self.pan_start = event.pos()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.start_pos:
            self.end_pos = event.pos()
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.start_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                scaled_size = self._pixmap.size() * self.zoom_factor
                pos = QPoint((self.width() - scaled_size.width()) // 2 + self.pan_offset.x(), (self.height() - scaled_size.height()) // 2 + self.pan_offset.y())
                
                rel_x = (rect.x() - pos.x()) / self.zoom_factor
                rel_y = (rect.y() - pos.y()) / self.zoom_factor
                rel_w = rect.width() / self.zoom_factor
                rel_h = rect.height() / self.zoom_factor
                
                if (rel_x >= 0 and rel_y >= 0 and rel_x + rel_w <= self._pixmap.width() and rel_y + rel_h <= self._pixmap.height()):
                    self.square_selected.emit(int(rel_x), int(rel_y), int(rel_w), int(rel_h))
                    
            self.start_pos = None
            self.update()
            
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1/1.15
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor * zoom_factor))
        
        if self.zoom_factor != old_zoom:
            mouse_pos = event.position().toPoint()
            zoom_ratio = self.zoom_factor / old_zoom
            self.pan_offset = QPoint(
                int((self.pan_offset.x() - mouse_pos.x()) * zoom_ratio + mouse_pos.x()),
                int((self.pan_offset.y() - mouse_pos.y()) * zoom_ratio + mouse_pos.y())
            )
        self.update()
        
    def mouseDoubleClickEvent(self, event):
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()


# --- NEW WIDGET: Particle Analysis Dialog ---
class ParticleAnalysisDialog(QDialog):
    """
    A new dialog that provides a sortable table view of all particles,
    showing their average correlation score and best match.
    """
    def __init__(self, data, subsquares, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Particle Analysis")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        self.data = data
        self.subsquares = subsquares
        self.analysis_results = []

        self._setup_ui()
        self._calculate_metrics()
        self._populate_table()

    def _setup_ui(self):
        T = AppTheme.CURRENT
        self.setStyleSheet(f"background-color: {T['BACKGROUND']}; color: {T['TEXT']};")
        layout = QVBoxLayout(self)

        instructions = QLabel("Sort by clicking headers. Double-click a row to compare a particle with its best match.")
        layout.addWidget(instructions)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Particle ID", "Avg. Score", "Best Match ID", "Best Match Score"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        
        layout.addWidget(self.table)

    def _calculate_metrics(self):
        """Calculates the average score and best match for each particle."""
        n = len(self.subsquares)
        if n == 0: return

        for i in range(n):
            scores = []
            best_match_idx = -1
            best_match_score = -1.0

            for j in range(n):
                if i == j: continue
                item = self.data[i, j]
                if item:
                    score = item[0]
                    scores.append(score)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = j
            
            avg_score = np.mean(scores) if scores else 0.0
            self.analysis_results.append({
                "index": i,
                "id": self.subsquares[i].unique_id,
                "avg_score": avg_score,
                "best_match_index": best_match_idx,
                "best_match_id": self.subsquares[best_match_idx].unique_id if best_match_idx != -1 else "N/A",
                "best_match_score": best_match_score if best_match_idx != -1 else 0.0
            })

    def _populate_table(self):
        """Fills the QTableWidget with the calculated analysis results."""
        self.table.setRowCount(len(self.analysis_results))
        for row, result in enumerate(self.analysis_results):
            # Create QTableWidgetItems for sorting. Note: For numeric sorting,
            # we must set the data with the correct type.
            id_item = QTableWidgetItem(result["id"])
            
            avg_score_item = QTableWidgetItem()
            avg_score_item.setData(Qt.ItemDataRole.DisplayRole, f"{result['avg_score']:.4f}")
            
            best_match_id_item = QTableWidgetItem(result["best_match_id"])
            
            best_match_score_item = QTableWidgetItem()
            best_match_score_item.setData(Qt.ItemDataRole.DisplayRole, f"{result['best_match_score']:.4f}")

            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, avg_score_item)
            self.table.setItem(row, 2, best_match_id_item)
            self.table.setItem(row, 3, best_match_score_item)

    def _on_cell_double_clicked(self, row, column):
        """Opens the ZoomableViewer for the double-clicked row."""
        # Get the original index from the sorted table view
        original_index = self.table.model().index(row, 0).row()
        
        result = self.analysis_results[original_index]
        
        s1_idx = result["index"]
        s2_idx = result["best_match_index"]

        if s1_idx == -1 or s2_idx == -1:
            return

        s1 = self.subsquares[s1_idx]
        s2 = self.subsquares[s2_idx]
        
        score, angle, metadata = self.data[s1_idx, s2_idx]
        
        viewer = ZoomableViewer(s1, s2, score, angle, metadata, self)
        viewer.exec()


# --- NEW WIDGET: Plotly Viewer Dialog ---
class PlotlyViewerDialog(QDialog):
    """A dedicated dialog window to display a Plotly figure in a QWebEngineView."""
    def __init__(self, fig, title="Plot Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 800)
        
        self.figure = fig
        
        layout = QVBoxLayout(self)
        self.webview = QWebEngineView()
        layout.addWidget(self.webview)
        
        self._render_figure()

    def _render_figure(self):
        """Saves the figure to a temporary HTML file and loads it."""
        try:
            html_content = self.figure.to_html(full_html=True, include_plotlyjs='cdn')
            self.webview.setHtml(html_content)
        except Exception as e:
            logging.error(f"Failed to render Plotly figure: {e}", exc_info=True)
            self.webview.setHtml(f"<h1>Error rendering plot</h1><p>{e}</p>")


# --- Main Application Window ---

class EnhancedImageAnalysisApp(QMainWindow):
    """The main application window, orchestrating the UI and processing threads."""
    def __init__(self):
        super().__init__()
        # Use QSettings to persist application state between sessions.
        self.settings = QSettings("StavrosAzinas", "2DCorrAnalysisTool_v3")
        
        # --- Application State ---
        self.image_paths = []
        self.subsquares = []
        self.filtered_subsquares = [] # For analysis
        self.correlation_data = None
        self.ref_square = None
        self.thread = None

        # --- Initialization ---
        self._init_logging()
        self._apply_theme()
        self._init_ui()
        self._setup_status_bar()
        self._load_settings()

    def _init_logging(self):
        """Sets up the custom logger to redirect messages to the GUI."""
        self.log_emitter = LogEmitter()
        self.log_emitter.log_message.connect(self._append_log_message)
        
        log_handler = QtLogHandler(self.log_emitter)
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        log_handler.setFormatter(log_format)
        
        # Configure the root logger.
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def _apply_theme(self):
        """Applies the stylesheet generated from the current theme."""
        theme_name = self.settings.value("theme", "light", type=str)
        AppTheme.set_theme(theme_name)
        self.setWindowTitle(f"🔬 2D Class Average Analysis Tool v4.1 ({theme_name.title()} Mode)")
        self.setStyleSheet(AppTheme.get_stylesheet())

    def _init_ui(self):
        """Initializes the main UI layout, panels, and widgets."""
        self._create_menu_bar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self._create_left_panel()
        left_panel.setMaximumWidth(380)
        left_panel.setMinimumWidth(320)
        
        right_panel = self._create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        # Restore splitter position from settings.
        splitter_sizes = self.settings.value("splitter_sizes", [350, 950], type=list)
        splitter.setSizes([int(s) for s in splitter_sizes])

    def _create_menu_bar(self):
        """Creates the main application menu bar."""
        menu_bar = self.menuBar()
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        load_action = QAction("📂 &Load Images...", self); load_action.triggered.connect(self._load_images)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        save_settings_action = QAction("💾 Save Settings", self); save_settings_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_settings_action)
        file_menu.addSeparator()
        exit_action = QAction("🚪 &Exit", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("🎨 Theme")
        light_theme_action = QAction("Light", self, checkable=True); light_theme_action.triggered.connect(lambda: self._change_theme("light"))
        dark_theme_action = QAction("Dark", self, checkable=True); dark_theme_action.triggered.connect(lambda: self._change_theme("dark"))
        
        # Add the actions to the menu itself so they are visible.
        theme_menu.addAction(light_theme_action)
        theme_menu.addAction(dark_theme_action)
        
        # Use QActionGroup for exclusive QAction behavior.
        theme_group = QActionGroup(self)
        theme_group.addAction(light_theme_action)
        theme_group.addAction(dark_theme_action)
        theme_group.setExclusive(True)
        
        # Set the initially checked theme based on saved settings.
        current_theme = self.settings.value("theme", "light", type=str)
        (dark_theme_action if current_theme == "dark" else light_theme_action).setChecked(True)

    def _create_left_panel(self):
        """Creates the left control panel with all user inputs and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File Operations Group
        file_group = QGroupBox("📁 File Operations")
        file_layout = QGridLayout(file_group)
        self.load_btn = QPushButton("📂 Load Images"); self.load_btn.clicked.connect(self._load_images)
        self.clear_btn = QPushButton("🗑️ Clear All"); self.clear_btn.clicked.connect(self._clear_all)
        file_layout.addWidget(self.load_btn, 0, 0, 1, 2)
        file_layout.addWidget(self.clear_btn, 1, 0, 1, 2)
        
        # Detection Mode Group (Restored)
        mode_group = QGroupBox("🔍 Detection Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.auto_radio = QRadioButton("🤖 Automatic Detection"); self.auto_radio.setChecked(True)
        self.ref_radio = QRadioButton("🎯 Reference-Based")
        self.mode_group = QButtonGroup(self); self.mode_group.addButton(self.auto_radio); self.mode_group.addButton(self.ref_radio)
        self.mode_group.buttonToggled.connect(self._update_ui_states)
        self.select_ref_btn = QPushButton("🎯 Select Reference Square"); self.select_ref_btn.clicked.connect(self._select_reference)
        mode_layout.addWidget(self.auto_radio); mode_layout.addWidget(self.ref_radio); mode_layout.addWidget(self.select_ref_btn)
        
        # Parameters Group
        params_group = QGroupBox("⚙️ Parameters")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("Min Size:"), 0, 0)
        self.min_size_spin = QSpinBox(); self.min_size_spin.setRange(10, 500); self.min_size_spin.setSuffix(" px")
        params_layout.addWidget(self.min_size_spin, 0, 1)
        params_layout.addWidget(QLabel("Target Size:"), 1, 0)
        self.target_size_spin = QSpinBox(); self.target_size_spin.setRange(32, 512); self.target_size_spin.setSuffix(" px")
        params_layout.addWidget(self.target_size_spin, 1, 1)
        self.ref_thresh_label = QLabel("Ref Threshold:")
        params_layout.addWidget(self.ref_thresh_label, 2, 0)
        self.ref_thresh_spin = QDoubleSpinBox(); self.ref_thresh_spin.setRange(0.1, 1.0); self.ref_thresh_spin.setSingleStep(0.05)
        params_layout.addWidget(self.ref_thresh_spin, 2, 1)
        # NEW: Quality Filter
        params_layout.addWidget(QLabel("Min. Quality Score:"), 3, 0)
        self.min_quality_spin = QDoubleSpinBox(); self.min_quality_spin.setRange(0.0, 1.0); self.min_quality_spin.setSingleStep(0.05); self.min_quality_spin.setValue(0.0)
        params_layout.addWidget(self.min_quality_spin, 3, 1)
        # NEW: Rotation Step
        params_layout.addWidget(QLabel("Rotation Step (°):"), 4, 0)
        self.rotation_step_spin = QSpinBox(); self.rotation_step_spin.setRange(1, 90); self.rotation_step_spin.setValue(15)
        params_layout.addWidget(self.rotation_step_spin, 4, 1)

        # Processing Group
        process_group = QGroupBox("🚀 Processing")
        process_layout = QGridLayout(process_group)
        # NEW: Comparison Mode
        process_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.comparison_mode_combo = QComboBox()
        self.comparison_mode_combo.addItems(["All vs. All", "One Image vs. Others"])
        self.comparison_mode_combo.currentIndexChanged.connect(self._update_ui_states)
        process_layout.addWidget(self.comparison_mode_combo, 0, 1)
        self.source_image_label = QLabel("Source:")
        process_layout.addWidget(self.source_image_label, 1, 0)
        self.source_image_combo = QComboBox()
        process_layout.addWidget(self.source_image_combo, 1, 1)

        self.detect_btn = QPushButton("🔍 Detect Squares"); self.detect_btn.clicked.connect(self._start_detection)
        self.analyze_btn = QPushButton("📊 Analyze Correlations"); self.analyze_btn.clicked.connect(self._start_analysis)
        self.stop_btn = QPushButton("⏹️ Stop"); self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setStyleSheet(f"background-color: {AppTheme.CURRENT['ERROR']};")
        process_layout.addWidget(self.detect_btn, 2, 0); process_layout.addWidget(self.analyze_btn, 2, 1)
        process_layout.addWidget(self.stop_btn, 3, 0, 1, 2)
        
        # Progress Group
        progress_group = QGroupBox("📈 Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = EnhancedProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        # Results Group
        results_group = QGroupBox("📋 Results")
        results_layout = QVBoxLayout(results_group)
        self.heatmap_btn = QPushButton("🔥 Show Heatmap"); self.heatmap_btn.clicked.connect(self._show_heatmap)
        # FIX: Add the new network graph button to the layout
        self.network_btn = QPushButton("🕸️ Show Network Graph"); self.network_btn.clicked.connect(self._show_network_graph)
        self.gallery_btn = QPushButton("🏆 Show Top Pairs"); self.gallery_btn.clicked.connect(self._show_gallery)
        self.analysis_btn = QPushButton("📊 Particle Analysis"); self.analysis_btn.clicked.connect(self._show_particle_analysis)
        self.export_btn = QPushButton("💾 Export Results"); self.export_btn.clicked.connect(self._export_results)
        results_layout.addWidget(self.heatmap_btn)
        results_layout.addWidget(self.network_btn)
        results_layout.addWidget(self.gallery_btn)
        results_layout.addWidget(self.analysis_btn)
        results_layout.addWidget(self.export_btn)
        
        layout.addWidget(file_group); layout.addWidget(mode_group); layout.addWidget(params_group)
        layout.addWidget(process_group); layout.addWidget(progress_group); layout.addWidget(results_group)
        layout.addStretch()
        return panel

    def _create_right_panel(self):
        """Creates the right panel, which contains the results tabs and loading overlay."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # A QStackedWidget is used to easily switch between the results and the loading screen.
        self.results_stack = QStackedWidget()
        layout.addWidget(self.results_stack)
        
        # Page 0: The main results tabs.
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget); results_layout.setContentsMargins(0,0,0,0)
        self.results_tabs = QTabWidget()
        
        # Tab 1: Detected Squares Gallery
        self.detection_scroll = QScrollArea(); self.detection_scroll.setWidgetResizable(True)
        self.detection_widget = QWidget(); self.detection_layout = QVBoxLayout(self.detection_widget)
        self.detection_scroll.setWidget(self.detection_widget)
        
        # Tab 2: Statistics
        self.stats_widget = QTextEdit(); self.stats_widget.setReadOnly(True)
        
        # Tab 3: Log
        self.log_widget = QTextEdit(); self.log_widget.setReadOnly(True)
        
        T = AppTheme.CURRENT
        log_style = f"font-family: 'Courier New', monospace; font-size: 11px; background-color: {T['LOG_BG']}; color: {T['LOG_TEXT']};"
        self.stats_widget.setStyleSheet(log_style)
        self.log_widget.setStyleSheet(log_style)
        
        self.results_tabs.addTab(self.detection_scroll, "🔍 Detected Squares")
        self.results_tabs.addTab(self.stats_widget, "📈 Statistics")
        self.results_tabs.addTab(self.log_widget, "📜 Log")
        results_layout.addWidget(self.results_tabs)
        
        # Page 1: The loading overlay. It sits in the same space as the tabs.
        self.loading_overlay = LoadingOverlay(self.results_stack)
        
        self.results_stack.addWidget(results_widget)
        self.results_stack.addWidget(self.loading_overlay)
        
        return panel

    def _setup_status_bar(self):
        """Creates and configures the status bar at the bottom of the window."""
        status_bar = self.statusBar()
        self.status_label = QLabel("🚀 Ready to analyze images!")
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()
        status_bar.addWidget(self.status_label, 1) # The '1' gives it stretch
        status_bar.addPermanentWidget(self.system_monitor)
        self._update_ui_states()

    def _update_ui_states(self):
        """Enables or disables UI elements based on the current application state."""
        is_processing = self.thread is not None and self.thread.isRunning()
        
        has_images = bool(self.image_paths)
        has_squares = bool(self.subsquares)
        has_correlation = self.correlation_data is not None
        is_ref_mode = self.ref_radio.isChecked()
        is_one_vs_others_mode = self.comparison_mode_combo.currentText() == "One Image vs. Others"

        # Enable/disable buttons
        self.load_btn.setEnabled(not is_processing)
        self.clear_btn.setEnabled(has_images and not is_processing)
        self.detect_btn.setEnabled(has_images and not is_processing)
        self.analyze_btn.setEnabled(bool(self.filtered_subsquares) and not is_processing)
        self.stop_btn.setEnabled(is_processing)
        self.heatmap_btn.setEnabled(has_correlation and not is_processing)
        self.network_btn.setEnabled(has_correlation and not is_processing)
        self.gallery_btn.setEnabled(has_correlation and not is_processing)
        self.analysis_btn.setEnabled(has_correlation and not is_processing)
        self.export_btn.setEnabled(has_correlation and not is_processing)
        
        # Show/hide reference-based controls
        self.select_ref_btn.setVisible(is_ref_mode)
        self.ref_thresh_label.setVisible(is_ref_mode)
        self.ref_thresh_spin.setVisible(is_ref_mode)
        self.select_ref_btn.setEnabled(is_ref_mode and has_images and not is_processing)

        # Show/hide "One vs Others" source image selector
        self.source_image_label.setVisible(is_one_vs_others_mode)
        self.source_image_combo.setVisible(is_one_vs_others_mode)
        
        # Control the loading overlay
        if is_processing:
            self.results_stack.setCurrentWidget(self.loading_overlay)
            self.loading_overlay.start_animation()
            self.progress_bar.setVisible(True)
        else:
            self.loading_overlay.stop_animation()
            self.results_stack.setCurrentIndex(0) # Show main tabs
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setDetailText("Idle")

    def _load_images(self):
        """Opens a file dialog to load images for analysis."""
        last_dir = self.settings.value("last_load_dir", os.path.expanduser("~"))
        paths, _ = QFileDialog.getOpenFileNames(self, "Load Images", last_dir, "Images (*.png *.jpg *.jpeg *.mrc *.tiff)")
        if paths:
            self.settings.setValue("last_load_dir", os.path.dirname(paths[0]))
            self.image_paths = sorted(paths)
            self._clear_results()
            msg = f"📁 Loaded {len(paths)} images."
            self.status_label.setText(msg)
            logging.info(msg)
            # NEW: Populate the source image combo box
            self.source_image_combo.clear()
            self.source_image_combo.addItems([os.path.basename(p) for p in self.image_paths])
            self._update_ui_states()
            self._update_stats()

    def _clear_all(self):
        """Clears all loaded data and results from the application."""
        self.image_paths.clear()
        self._clear_results()
        self.source_image_combo.clear()
        self.status_label.setText("🗑️ All data cleared.")
        logging.info("All data and results have been cleared.")
        self._update_ui_states()
        self._update_stats()

    def _clear_results(self):
        """Clears only the results of processing, keeping loaded images."""
        self.subsquares.clear()
        self.filtered_subsquares.clear()
        self.correlation_data = None
        self.ref_square = None # Also clear reference selection
        # Clear the detected squares gallery
        while self.detection_layout.count():
            child = self.detection_layout.takeAt(0)
            if child and child.widget():
                child.widget().deleteLater()
        self.stats_widget.clear()

    def _select_reference(self):
        """Opens a dialog to allow the user to select a reference square."""
        if not self.image_paths: return
        dialog = QDialog(self)
        dialog.setWindowTitle("🎯 Select Reference Square")
        dialog.setMinimumSize(900, 700)
        layout = QVBoxLayout(dialog)
        
        # ... (Implementation of this dialog is restored from original code) ...
        instructions = QLabel("<b>Instructions:</b><br>1. Choose an image<br>2. Pan/Zoom as needed<br>3. Left-click & drag to select reference<br>4. Click OK to confirm")
        layout.addWidget(instructions)
        
        image_combo = QComboBox()
        image_combo.addItems([os.path.basename(p) for p in self.image_paths])
        layout.addWidget(image_combo)
        
        viewer = InteractiveImageViewer()
        layout.addWidget(viewer, 1)
        
        self.selection_info = QLabel("📏 No selection")
        layout.addWidget(self.selection_info)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        selection = {}
        def on_image_changed(index):
            viewer.set_image(self.image_paths[index])
            selection.clear()
            self.selection_info.setText("📏 No selection")
        def on_selection(x, y, w, h):
            selection.update({'rect': (x, y, w, h)})
            self.selection_info.setText(f"📏 Selected: {w}×{h} at ({x}, {y})")
            # Automatically update size parameters based on selection
            self.target_size_spin.setValue(max(w, h))
            self.min_size_spin.setValue(int(min(w, h) * 0.8))
            
        image_combo.currentIndexChanged.connect(on_image_changed)
        viewer.square_selected.connect(on_selection)
        viewer.set_image(self.image_paths[0])
        
        if dialog.exec() == QDialog.DialogCode.Accepted and 'rect' in selection:
            self.ref_square = (self.image_paths[image_combo.currentIndex()], *selection['rect'])
            msg = f"🎯 Reference square selected from {os.path.basename(self.ref_square[0])}."
            self.status_label.setText(msg)
            logging.info(msg)

    def _start_detection(self):
        """Starts the ImageProcessor thread to detect and preprocess squares."""
        self._clear_results()
        params = self._get_detection_parameters()
        # Pass the reference square only if in reference mode.
        ref_square = self.ref_square if self.ref_radio.isChecked() else None
        
        self.thread = ImageProcessor(self.image_paths, params, ref_square)
        self.thread.progress.connect(self.progress_bar.setDetailText)
        self.thread.progress_value.connect(self.progress_bar.setValue)
        self.thread.finished_detection.connect(self._on_detection_finished)
        self.thread.finished.connect(self._on_thread_finished)
        
        self.progress_bar.setRange(0, 100)
        self.thread.start()
        self._update_ui_states()

    def _start_analysis(self):
        """Starts the EnhancedCorrelationProcessor thread."""
        if not self.filtered_subsquares:
            logging.warning("Analysis started with no particles passing the quality filter.")
            self.status_label.setText("⚠️ No particles passed the quality filter. Adjust parameters.")
            return

        params = self._get_analysis_parameters()
        
        self.thread = EnhancedCorrelationProcessor(self.filtered_subsquares, params)
        self.thread.progress.connect(self.progress_bar.setDetailText)
        self.thread.progress_value.connect(self.progress_bar.setValue)
        self.thread.finished_correlation.connect(self._on_analysis_finished)
        self.thread.finished.connect(self._on_thread_finished)
        
        self.progress_bar.setRange(0, 100)
        self.thread.start()
        self._update_ui_states()

    def _stop_processing(self):
        """Stops the currently running worker thread."""
        if self.thread and self.thread.isRunning():
            self.status_label.setText("⏹️ Stopping process...")
            logging.warning("User requested to stop the current process.")
            self.thread.stop()
    
    @pyqtSlot(list)
    def _on_detection_finished(self, squares):
        """Handles the results when the detection thread finishes."""
        if self.thread: # Check if not stopped
            self.subsquares = sorted(squares, key=lambda s: s.unique_id)
            
            # NEW: Apply quality filter
            min_quality = self.min_quality_spin.value()
            if min_quality > 0.0:
                self.filtered_subsquares = [s for s in self.subsquares if s.overall_quality >= min_quality]
                logging.info(f"Quality filter applied (>{min_quality:.2f}). Kept {len(self.filtered_subsquares)} of {len(self.subsquares)} particles.")
            else:
                self.filtered_subsquares = self.subsquares
            
            self._display_detected_squares()
            self._update_stats()
            self.status_label.setText(f"✅ Detection complete. Found {len(squares)} particles ({len(self.filtered_subsquares)} passed filter).")

    @pyqtSlot(object)
    def _on_analysis_finished(self, data):
        """Handles the results when the analysis thread finishes."""
        if self.thread:
            self.correlation_data = data
            self._update_stats()
            self.status_label.setText("✅ Correlation analysis complete.")

    @pyqtSlot()
    def _on_thread_finished(self):
        """Common cleanup logic for when any worker thread finishes."""
        self.thread = None
        self._update_ui_states()

    def _display_detected_squares(self):
        """Populates the gallery in the 'Detected Squares' tab."""
        # ... (Implementation restored from original, with styling updates) ...
        while self.detection_layout.count():
            child = self.detection_layout.takeAt(0)
            if child and child.widget(): child.widget().deleteLater()

        groups = {}
        for square in self.subsquares:
            groups.setdefault(square.original_image_path, []).append(square)
            
        for image_path in sorted(groups.keys()):
            squares = groups[image_path]
            header = QLabel(f"📁 <b>{os.path.basename(image_path)}</b> ({len(squares)} squares)")
            header.setStyleSheet("font-size: 14px; padding: 8px; margin: 5px 0;")
            self.detection_layout.addWidget(header)
            
            gallery_widget = QWidget()
            gallery_layout = QHBoxLayout(gallery_widget)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedHeight(150); scroll.setWidget(gallery_widget)
            
            for square in squares:
                gallery_layout.addWidget(self._create_thumbnail_widget(square))
            gallery_layout.addStretch()
            self.detection_layout.addWidget(scroll)
            
        self.detection_layout.addStretch()

    def _create_thumbnail_widget(self, square):
        """Creates a single thumbnail widget for the detected squares gallery."""
        # ... (Implementation restored from original, with styling updates) ...
        T = AppTheme.CURRENT
        widget = QFrame(); widget.setFixedSize(120, 140)
        widget.setStyleSheet(f"QFrame {{ border: 1px solid {T['BORDER']}; border-radius: 8px; background-color: {T['CONTENT_BG']}; margin: 2px; }} QFrame:hover {{ border-color: {T['PRIMARY']}; }}")
        layout = QVBoxLayout(widget); layout.setSpacing(2); layout.setContentsMargins(5, 5, 5, 5)
        
        thumb_label = QLabel(); thumb_label.setPixmap(square.to_qpixmap(size=100)); thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumb_label)
        
        info_label = QLabel(f"<b>{square.grid_id}</b>"); info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        quality = square.overall_quality
        color = T['SUCCESS'] if quality > 0.7 else T['WARNING'] if quality > 0.4 else T['ERROR']
        quality_label = QLabel(f"⭐ {quality:.2f}"); quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quality_label.setStyleSheet(f"font-size: 10px; color: {color}; font-weight: bold;")
        layout.addWidget(quality_label)
        return widget

    def _update_stats(self):
        """Updates the text in the 'Statistics' tab."""
        stats = f"📊 ANALYSIS STATISTICS\n{'='*50}\n\n"
        stats += f"📁 Images loaded: {len(self.image_paths)}\n"
        stats += f"🔍 Squares detected: {len(self.subsquares)}\n"
        stats += f"✅ Squares passing filter: {len(self.filtered_subsquares)}\n"
        if self.filtered_subsquares:
            qualities = [s.overall_quality for s in self.filtered_subsquares]
            stats += f"⭐ Average quality (filtered): {np.mean(qualities):.3f}\n\n"
        if self.correlation_data is not None:
            n = len(self.filtered_subsquares)
            scores = [self.correlation_data[i, j][0] for i in range(n) for j in range(i+1, n) if self.correlation_data[i,j] is not None]
            if scores:
                stats += f"🔗 Correlation pairs: {len(scores)}\n"
                stats += f"📊 Average correlation: {np.mean(scores):.3f}\n"
                stats += f"📈 Correlation range: [{np.min(scores):.3f} - {np.max(scores):.3f}]\n"
        self.stats_widget.setText(stats)

    def _get_detection_parameters(self):
        """Collects all detection parameters from the UI into a dictionary."""
        return {
            'min_square_size': self.min_size_spin.value(),
            'target_subsquare_size': (self.target_size_spin.value(), self.target_size_spin.value()),
            'ref_threshold': self.ref_thresh_spin.value()
        }

    def _get_analysis_parameters(self):
        """Collects all analysis-specific parameters from the UI."""
        params = {
            'rotation_step': self.rotation_step_spin.value(),
            'comparison_mode': self.comparison_mode_combo.currentText()
        }
        if params['comparison_mode'] == "One Image vs. Others":
            if self.source_image_combo.currentIndex() >= 0:
                params['source_image_path'] = self.image_paths[self.source_image_combo.currentIndex()]
            else:
                params['source_image_path'] = None
        return params

    def _show_heatmap(self):
        """
        Generates the heatmap, saves it as a temporary HTML file, and opens
        it in the user's default web browser. This is the stable, cross-platform
        solution that avoids using the unstable QWebEngineView.
        """
        if self.correlation_data is None:
            logging.warning("Heatmap requested but no correlation data is available.")
            self.status_label.setText("⚠️ No correlation data to display.")
            return

        try:
            self.status_label.setText("🔥 Generating heatmap...")
            QApplication.processEvents() # Allow GUI to update

            scores = np.array([[d[0] if isinstance(d, tuple) else 0 for d in row] for row in self.correlation_data])
            labels = [s.unique_id for s in self.filtered_subsquares]
            
            fig = go.Figure(data=go.Heatmap(
                z=scores,
                x=labels,
                y=labels,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='<b>%{y} vs %{x}</b><br>Score: %{z:.4f}<extra></extra>'
            ))
            
            T = AppTheme.CURRENT
            fig.update_layout(
                title={'text': "Cross-Correlation Matrix", 'x': 0.5},
                yaxis_autorange='reversed',
                font=dict(color=T['TEXT']),
                paper_bgcolor=T['BACKGROUND'],
                plot_bgcolor=T['CONTENT_BG'],
            )

            viewer = PlotlyViewerDialog(fig, "🔥 Heatmap Viewer", self)
            viewer.exec()
            msg = "🔥 Heatmap closed."
            self.status_label.setText(msg)
            logging.info(msg)

        except Exception as e:
            msg = f"❌ Failed to generate heatmap: {e}"
            self.status_label.setText(msg)
            logging.error(msg, exc_info=True)

    def _show_gallery(self):
        """Shows the top pairs gallery dialog if correlation data exists."""
        if self.correlation_data is not None:
            gallery = TopPairsGallery(self.correlation_data, self.filtered_subsquares, self)
            gallery.exec()

    def _show_particle_analysis(self):
        """Shows the new particle analysis dialog."""
        if self.correlation_data is not None:
            analysis_dialog = ParticleAnalysisDialog(self.correlation_data, self.filtered_subsquares, self)
            analysis_dialog.exec()
            
    def _show_network_graph(self):
        """
        Generates an interactive network graph of particle relationships
        and opens it in the user's default web browser.
        """
        if self.correlation_data is None:
            logging.warning("Network graph requested but no correlation data is available.")
            self.status_label.setText("⚠️ No correlation data to display.")
            return

        try:
            self.status_label.setText("🕸️ Generating network graph...")
            QApplication.processEvents()

            # --- Graph Data Preparation ---
            nodes = self.filtered_subsquares
            n = len(nodes)
            # A threshold to decide if a connection is strong enough to be drawn.
            correlation_threshold = 0.8 

            edge_x, edge_y = [], []
            
            for i in range(n):
                for j in range(i + 1, n):
                    item = self.correlation_data[i, j]
                    if item and item[0] > correlation_threshold:
                        # For each edge, we need the start and end coordinates.
                        # We'll assign coordinates later. For now, store indices.
                        edge_x.extend([i, j, None]) # 'None' creates a break in the line
                        edge_y.extend([i, j, None]) # Using indices as placeholders for now

            # FIX: Check if any edges were created before trying to plot.
            if not edge_x:
                msg = "No pairs found above the correlation threshold of 0.8 to build a network graph."
                QMessageBox.information(self, "Network Graph", msg)
                logging.warning(msg)
                self.status_label.setText("🕸️ No strong correlations to graph.")
                return

            # --- Node Layout (Simple Circle) ---
            # Arrange nodes in a circle for a clean initial layout.
            node_x = [np.cos(2 * np.pi * i / n) for i in range(n)]
            node_y = [np.sin(2 * np.pi * i / n) for i in range(n)]
            
            # Now, replace the placeholder indices in edge_x/y with actual coordinates
            final_edge_x = [node_x[i] if i is not None else None for i in edge_x]
            final_edge_y = [node_y[i] if i is not None else None for i in edge_y]

            # --- Create Plotly Traces ---
            T = AppTheme.CURRENT
            edge_trace = go.Scatter(
                x=final_edge_x, y=final_edge_y,
                line=dict(width=0.5, color=T['BORDER']),
                hoverinfo='none',
                mode='lines')

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=[node.unique_id for node in nodes],
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[s.overall_quality for s in nodes],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Particle Quality',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2))

            # --- Create Figure ---
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title='Particle Correlation Network Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(
                                text=f"Edges shown for correlations > {correlation_threshold}",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            paper_bgcolor=T['BACKGROUND'],
                            plot_bgcolor=T['CONTENT_BG'],
                            font=dict(color=T['TEXT'])
                        ))
            
            viewer = PlotlyViewerDialog(fig, "🕸️ Network Graph Viewer", self)
            viewer.exec()
            msg = "🕸️ Network graph closed."
            self.status_label.setText(msg)
            logging.info(msg)

        except Exception as e:
            msg = f"❌ Failed to generate network graph: {e}"
            self.status_label.setText(msg)
            logging.error(msg, exc_info=True)


    def _export_results(self):
        """Exports the correlation results to a CSV file."""
        if self.correlation_data is None: return
        last_dir = self.settings.value("last_export_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", os.path.join(last_dir, "correlation_results.csv"), "CSV Files (*.csv)")
        if path:
            self.settings.setValue("last_export_dir", os.path.dirname(path))
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Square1_ID', 'Square2_ID', 'Correlation_Score', 'Rotation_Angle', 'GPU_Accelerated'])
                    n = len(self.filtered_subsquares)
                    for i in range(n):
                        for j in range(i + 1, n):
                            item = self.correlation_data[i,j]
                            if item:
                                s1, s2 = self.filtered_subsquares[i], self.filtered_subsquares[j]
                                score, angle, meta = item
                                writer.writerow([s1.unique_id, s2.unique_id, f"{score:.6f}", f"{angle:.1f}", meta.get('gpu_accelerated', False)])
                msg = f"💾 Results exported to {os.path.basename(path)}"
                self.status_label.setText(msg); logging.info(msg)
            except Exception as e:
                logging.error(f"Export failed: {e}", exc_info=True)
                self.status_label.setText(f"❌ Export failed: {e}")

    def _change_theme(self, theme_name):
        """Changes the application theme and prompts for a restart."""
        self.settings.setValue("theme", theme_name)
        dlg = QDialog(self); dlg.setWindowTitle("Theme Changed")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Please restart the application for the theme change to take full effect."))
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec()

    @pyqtSlot(str)
    def _append_log_message(self, message):
        """Appends a message to the log widget in the GUI."""
        self.log_widget.append(message)

    def _load_settings(self):
        """Loads application settings from the previous session."""
        geometry_setting = self.settings.value("geometry")
        if geometry_setting and isinstance(geometry_setting, QByteArray):
            self.restoreGeometry(geometry_setting)
        
        # Load other settings. These are safe as they use standard types.
        self.min_size_spin.setValue(self.settings.value("min_size", 30, type=int))
        self.target_size_spin.setValue(self.settings.value("target_size", 128, type=int))
        self.ref_thresh_spin.setValue(self.settings.value("ref_thresh", 0.7, type=float))
        self.min_quality_spin.setValue(self.settings.value("min_quality", 0.0, type=float))
        self.rotation_step_spin.setValue(self.settings.value("rotation_step", 15, type=int))
        logging.info("Loaded application settings.")

    def _save_settings(self):
        """Saves the current application settings for the next session."""
        self.settings.setValue("geometry", self.saveGeometry())
        # The findChild method is used to safely get the splitter widget.
        splitter = self.centralWidget().findChild(QSplitter)
        if splitter:
            self.settings.setValue("splitter_sizes", splitter.sizes())
        self.settings.setValue("min_size", self.min_size_spin.value())
        self.settings.setValue("target_size", self.target_size_spin.value())
        self.settings.setValue("ref_thresh", self.ref_thresh_spin.value())
        self.settings.setValue("min_quality", self.min_quality_spin.value())
        self.settings.setValue("rotation_step", self.rotation_step_spin.value())
        self.status_label.setText("💾 Settings saved.")
        logging.info("Application settings saved.")

    def closeEvent(self, event):
        """Handles the application close event."""
        self._save_settings()
        self._stop_processing()
        self.system_monitor.stop_monitoring()
        if self.thread and self.thread.isRunning():
            self.thread.wait(2000) # Wait up to 2s for thread to finish gracefully
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("2D Class Correlation Analysis Tool")
    app.setApplicationVersion("2 (Stable)")
    app.setOrganizationName("StavrosAzinas")
    
    # --- FIX: Robustly set the application icon ---
    # This determines the script's directory and builds an absolute path to the icon.
    # This ensures the icon is found regardless of where the script is run from.
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle (e.g., PyInstaller)
        script_dir = sys._MEIPASS
    else:
        # If the application is run as a script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    icon_path = os.path.join(script_dir, "app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"⚠️ app_icon.png not found at '{icon_path}'. No icon will be set.")

    window = EnhancedImageAnalysisApp()
    window.show()
    
    window.status_label.setText("🎉 Welcome to the 2D Correlation Analysis Tool v2!")
    logging.info("="*40)
    logging.info("      Application Started Successfully")
    logging.info(f"      GPU Available: {GPU_AVAILABLE} (Backend: {GPU_BACKEND or 'N/A'})")
    logging.info("="*40)
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
