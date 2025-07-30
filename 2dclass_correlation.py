import sys
import os
import cv2
import numpy as np
import psutil
import GPUtil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QRadioButton, QButtonGroup,
    QComboBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QFrame,
    QSplitter, QTabWidget, QTextEdit, QCheckBox, QSlider
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QColor, QTransform, QFont,
    QLinearGradient, QPalette, QBrush
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QPoint, QRect, QObject, pyqtSlot, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class AnimatedLoadingWidget(QWidget):
    """Enhanced loading widget with multiple animation styles"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.setFixedSize(200, 30)
        self.setVisible(False)
        self.animation_style = 'wave'  # 'wave', 'pulse', 'dots'
        
    def set_style(self, style):
        self.animation_style = style
        
    def start_animation(self):
        self.phase = 0
        self.timer.start(50)
        self.setVisible(True)
        
    def stop_animation(self):
        self.timer.stop()
        self.setVisible(False)
        
    def update_animation(self):
        self.phase += 0.2
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.animation_style == 'wave':
            self._draw_wave(painter)
        elif self.animation_style == 'pulse':
            self._draw_pulse(painter)
        else:
            self._draw_dots(painter)
            
    def _draw_wave(self, painter):
        painter.setPen(QPen(QColor("#3498db"), 3))
        w, h = self.width(), self.height()
        mid_y, amp = h / 2, (h / 2) - 5
        x = np.linspace(0, w, 100)
        y = mid_y + amp * np.sin(x / 15 + self.phase)
        
        for i in range(len(x) - 1):
            painter.drawLine(int(x[i]), int(y[i]), int(x[i+1]), int(y[i+1]))
            
    def _draw_pulse(self, painter):
        w, h = self.width(), self.height()
        radius = 8 + 5 * abs(np.sin(self.phase))
        painter.setBrush(QBrush(QColor("#e74c3c")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(w/2 - radius), int(h/2 - radius), int(2*radius), int(2*radius))
        
    def _draw_dots(self, painter):
        w, h = self.width(), self.height()
        painter.setBrush(QBrush(QColor("#2ecc71")))
        painter.setPen(Qt.PenStyle.NoPen)
        
        for i in range(5):
            x = w / 6 * (i + 1)
            y = h / 2 + 5 * np.sin(self.phase + i * 0.5)
            painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

class SystemMonitor(QWidget):
    """Widget to monitor system resources"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.cpu_label = QLabel("CPU: 0%")
        self.ram_label = QLabel("RAM: 0%")
        self.gpu_label = QLabel("GPU: N/A")
        
        layout.addWidget(QLabel("📊"))
        layout.addWidget(self.cpu_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.ram_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.gpu_label)
        layout.addStretch()
        
        self.setStyleSheet("""
            QLabel { color: #666; font-size: 11px; }
        """)
        
    def start_monitoring(self):
        self.timer.start(2000)  # Update every 2 seconds
        
    def stop_monitoring(self):
        self.timer.stop()
        
    def update_stats(self):
        # CPU and RAM
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
        self.ram_label.setText(f"RAM: {ram_percent:.1f}%")
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.gpu_label.setText(f"GPU: {gpu.load * 100:.1f}%")
            else:
                self.gpu_label.setText("GPU: N/A")
        except:
            self.gpu_label.setText("GPU: N/A")

class QualityMetrics:
    """Enhanced quality metrics for dynamic weighting"""
    
    @staticmethod
    def calculate_texture_quality(img):
        """Calculate texture quality using multiple metrics"""
        if img is None:
            return 0.0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Standard deviation (contrast)
        std_dev = np.std(gray.astype(np.float32))
        
        # Local binary pattern variance (texture)
        lbp_var = QualityMetrics._calculate_lbp_variance(gray)
        
        # Gradient magnitude (edge density)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(grad_mag)
        
        # Combine metrics (normalized)
        texture_score = (std_dev / 255.0) * 0.4 + (lbp_var / 100.0) * 0.3 + (grad_mean / 255.0) * 0.3
        return min(texture_score, 1.0)
    
    @staticmethod
    def calculate_edge_quality(edge_img):
        """Calculate edge quality with multiple metrics"""
        if edge_img is None:
            return 0.0
            
        gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY) if len(edge_img.shape) == 3 else edge_img
        
        # Mean edge strength
        mean_strength = np.mean(gray) / 255.0
        
        # Edge density (percentage of pixels that are edges)
        edge_density = np.sum(gray > 50) / gray.size
        
        # Edge continuity (using morphological operations)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        continuity = np.sum(closed > 50) / np.sum(gray > 50) if np.sum(gray > 50) > 0 else 0
        
        # Combine metrics
        edge_score = mean_strength * 0.5 + edge_density * 0.3 + continuity * 0.2
        return min(edge_score, 1.0)
    
    @staticmethod
    def _calculate_lbp_variance(gray):
        """Calculate Local Binary Pattern variance for texture analysis"""
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0
            
        # Simple LBP implementation
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                        
                lbp[i-1, j-1] = code
        
        return np.var(lbp.astype(np.float32))

class WebBridge(QObject):
    heatmap_clicked = pyqtSignal(int, int)
    
    @pyqtSlot(int, int)
    def on_heatmap_click(self, i, j):
        self.heatmap_clicked.emit(i, j)

class Subsquare:
    def __init__(self, img_id, grid_id, path, img, bbox, proc=None, edge=None):
        self.image_id = img_id
        self.grid_id = grid_id
        self.original_image_path = path
        self.original_subsquare_img = img
        self.bbox = bbox
        self.processed_img = proc
        self.edge_img = edge
        self.unique_id = f"{os.path.basename(path).split('.')[0]}_{grid_id}"
        
        # Quality metrics
        self.texture_quality = 0.0
        self.edge_quality = 0.0
        self.overall_quality = 0.0
        
    def calculate_quality_metrics(self):
        """Calculate and store quality metrics"""
        if self.processed_img is not None:
            self.texture_quality = QualityMetrics.calculate_texture_quality(self.processed_img)
        
        if self.edge_img is not None:
            self.edge_quality = QualityMetrics.calculate_edge_quality(self.edge_img)
            
        self.overall_quality = (self.texture_quality + self.edge_quality) / 2.0
        
    def to_qpixmap(self, img_data=None, size=100):
        img = img_data if img_data is not None else self.processed_img
        if img is None:
            return QPixmap(size, size)
            
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        return pixmap.scaled(
            QSize(size, size),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

class EnhancedProgressBar(QWidget):
    """Custom progress bar with additional info"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 6px;
            }
        """)
        
        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("font-size: 11px; color: #666;")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)
        
    def setValue(self, value):
        self.progress_bar.setValue(value)
        
    def setRange(self, min_val, max_val):
        self.progress_bar.setRange(min_val, max_val)
        
    def setDetailText(self, text):
        self.detail_label.setText(text)

class ImageProcessor(QThread):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_detection = pyqtSignal(list)
    
    def __init__(self, paths, params, ref_square=None):
        super().__init__()
        self.image_paths = paths
        self.params = params
        self.ref_square = ref_square
        self.all_subsquares = []
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            if self.ref_square:
                self.detect_using_reference()
            else:
                self.detect_automatically()
                
            # Calculate quality metrics for all subsquares
            self.progress.emit("Calculating quality metrics...")
            for i, ss in enumerate(self.all_subsquares):
                if not self._is_running:
                    break
                ss.calculate_quality_metrics()
                progress_val = int((i / len(self.all_subsquares)) * 100)
                self.progress_value.emit(progress_val)
                
            self.finished_detection.emit(self.all_subsquares)
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            
    def detect_using_reference(self):
        ref_path, rx, ry, rw, rh = self.ref_square
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return
            
        ref_template = ref_img[ry:ry+rh, rx:rx+rw]
        grid_map = self._create_grid_map()
        
        total_images = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if not self._is_running:
                break
                
            self.progress.emit(f"Matching in {os.path.basename(path)}...")
            self.progress_value.emit(int((i / total_images) * 50))  # First 50% for detection
            
            img = cv2.imread(path)
            if img is not None:
                matches = self._template_match(img, ref_template, path, i, grid_map[i])
                self.all_subsquares.extend(matches)
                
        self.preprocess_all()
        
    def _template_match(self, image, template, path, img_id, grid):
        squares = []
        g_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = g_template.shape
        
        matches = []
        angles = np.arange(0, 360, 30)  # Reduced angle steps for speed
        
        for angle in angles:
            if not self._is_running:
                break
                
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(g_template, M, (w, h))
            
            res = cv2.matchTemplate(g_img, rotated, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= self.params['ref_threshold'])
            
            for pt in zip(*locs[::-1]):
                matches.append((pt[0], pt[1], w, h, res[pt[1], pt[0]]))
                
        # Sort by score and filter overlaps
        matches.sort(key=lambda x: x[4], reverse=True)
        filtered = []
        
        for x, y, wm, hm, score in matches:
            if not any(self._calc_overlap((x, y, wm, hm), s.bbox) > 0.4 for s in filtered):
                subsquare = Subsquare(
                    img_id, 
                    self._get_grid_id(x, y, grid), 
                    path, 
                    image[y:y+hm, x:x+wm], 
                    (x, y, wm, hm)
                )
                filtered.append(subsquare)
                
        return filtered
        
    def detect_automatically(self):
        grid_map = self._create_grid_map()
        total_images = len(self.image_paths)
        
        for i, path in enumerate(self.image_paths):
            if not self._is_running:
                break
                
            self.progress.emit(f"Detecting in {os.path.basename(path)}...")
            self.progress_value.emit(int((i / total_images) * 50))
            
            detected = self._detect_squares(path, i, grid_map[i])
            self.all_subsquares.extend(detected)
            
        self.preprocess_all()
        
    def _create_grid_map(self):
        grid_map = {}
        for i, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                grid_map[i] = {'size': (w / 10, h / 10)}
        return grid_map
        
    def _get_grid_id(self, x, y, grid):
        col = int(x // grid['size'][0])
        row = int(y // grid['size'][1])
        return f"{chr(ord('A') + min(col, 25))}{row + 1}"
        
    def preprocess_all(self):
        total = len(self.all_subsquares)
        for i, ss in enumerate(self.all_subsquares):
            if not self._is_running:
                break
                
            self.progress.emit(f"Processing square {i+1}/{total}...")
            self.progress_value.emit(50 + int((i / total) * 50))  # Second 50%
            
            self._preprocess_subsquare(ss)
            
    def _detect_squares(self, path, img_id, grid):
        img = cv2.imread(path)
        if img is None:
            return []
            
        # Enhanced square detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        min_size = self.params['min_square_size']
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < min_size * min_size:
                continue
                
            # Approximate contour to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly square/rectangular
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # More lenient aspect ratio for squares
            if 0.7 <= aspect_ratio <= 1.4 and w >= min_size and h >= min_size:
                # Additional quality checks
                extent = area / (w * h)  # How much of bounding rect is filled
                
                if extent > 0.5:  # At least 50% filled
                    subsquare = Subsquare(
                        img_id,
                        self._get_grid_id(x, y, grid),
                        path,
                        img[y:y+h, x:x+w],
                        (x, y, w, h)
                    )
                    detected.append(subsquare)
                    
        return self._remove_duplicates(detected)
        
    def _remove_duplicates(self, squares):
        # Sort by area (largest first)
        squares.sort(key=lambda s: s.bbox[2] * s.bbox[3], reverse=True)
        
        filtered = []
        for square in squares:
            # Check overlap with already accepted squares
            overlap = any(
                self._calc_overlap(square.bbox, accepted.bbox) > 0.5 
                for accepted in filtered
            )
            
            if not overlap:
                filtered.append(square)
                
        return filtered
        
    def _calc_overlap(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _preprocess_subsquare(self, ss):
        target = self.params['target_subsquare_size']
        img = ss.original_subsquare_img
        
        if img is None or img.size == 0:
            return
            
        h, w = img.shape[:2]
        
        # Calculate scaling
        scale = min(target[0] / w, target[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize with high-quality interpolation
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image
        padded = np.full((target[1], target[0], 3), 128, dtype=np.uint8)
        y_offset = (target[1] - new_h) // 2
        x_offset = (target[0] - new_w) // 2
        
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Enhanced preprocessing
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        ss.processed_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Enhanced edge detection
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Combine multiple edge detection methods
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine edge responses
        combined_edges = np.sqrt(laplacian**2 + sobel_x**2 + sobel_y**2)
        
        # Normalize and convert
        edge_normalized = cv2.normalize(combined_edges, None, 0, 255, cv2.NORM_MINMAX)
        ss.edge_img = cv2.cvtColor(edge_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

class EnhancedCorrelationProcessor(QThread):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_correlation = pyqtSignal(object)
    
    def __init__(self, subsquares):
        super().__init__()
        self.subsquares = subsquares
        self.data = None
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        n = len(self.subsquares)
        if n < 2:
            self.finished_correlation.emit(None)
            return
            
        # Initialize correlation matrix
        self.data = np.empty((n, n), dtype=object)
        
        # Generate all pairs
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        total_pairs = len(pairs)
        
        self.progress.emit(f"Starting correlation analysis for {total_pairs} pairs...")
        
        # Process pairs with threading
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            future_to_pair = {executor.submit(self._process_pair, pair): pair for pair in pairs}
            
            completed = 0
            for future in as_completed(future_to_pair):
                if not self._is_running:
                    break
                    
                pair = future_to_pair[future]
                i, j = pair
                
                try:
                    result = future.result()
                    self.data[i, j] = self.data[j, i] = result
                    
                    completed += 1
                    progress_val = int((completed / total_pairs) * 100)
                    self.progress_value.emit(progress_val)
                    self.progress.emit(f"Processed pair {completed}/{total_pairs}")
                    
                except Exception as e:
                    print(f"Error processing pair {pair}: {e}")
                    self.data[i, j] = self.data[j, i] = (0.0, 0.0, {})
                    
        self.finished_correlation.emit(self.data)
        
    def _process_pair(self, pair):
        """Enhanced correlation processing with dynamic weighting"""
        i, j = pair
        
        if i == j:
            return (1.0, 0.0, {'texture_weight': 0.5, 'edge_weight': 0.5, 'quality_score': 1.0})
            
        s1, s2 = self.subsquares[i], self.subsquares[j]
        
        # Calculate dynamic weights based on quality metrics
        texture_quality_avg = (s1.texture_quality + s2.texture_quality) / 2.0
        edge_quality_avg = (s1.edge_quality + s2.edge_quality) / 2.0
        
        total_quality = texture_quality_avg + edge_quality_avg
        if total_quality > 0:
            texture_weight = texture_quality_avg / total_quality
            edge_weight = edge_quality_avg / total_quality
        else:
            texture_weight = edge_weight = 0.5
            
        # Calculate correlation scores
        texture_score, texture_angle = self._calculate_correlation(
            s1.processed_img, s2.processed_img, method='texture'
        )
        
        edge_score, edge_angle = self._calculate_correlation(
            s1.edge_img, s2.edge_img, method='edge'
        )
        
        # Combine scores with dynamic weights
        final_score = texture_score * texture_weight + edge_score * edge_weight
        
        # Use the angle from the higher-scoring method
        final_angle = texture_angle if texture_score > edge_score else edge_angle
        
        # Additional metadata
        metadata = {
            'texture_weight': texture_weight,
            'edge_weight': edge_weight,
            'texture_score': texture_score,
            'edge_score': edge_score,
            'quality_score': (s1.overall_quality + s2.overall_quality) / 2.0
        }
        
        return (final_score, final_angle, metadata)
        
    def _calculate_correlation(self, img1, img2, method='texture'):
        """Enhanced correlation calculation with multiple methods"""
        if img1 is None or img2 is None:
            return 0.0, 0.0
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Normalize images
        gray1 = gray1.astype(np.float32) / 255.0
        gray2 = gray2.astype(np.float32) / 255.0
        
        max_correlation = -1.0
        best_angle = 0
        
        # Test different angles (reduced for speed but still comprehensive)
        angles = np.arange(0, 360, 10) if method == 'texture' else np.arange(0, 360, 15)
        
        for angle in angles:
            # Rotate second image
            center = (gray2.shape[1] // 2, gray2.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray2, M, gray2.shape[:2])
            
            # Calculate correlation using multiple methods
            if method == 'texture':
                correlation = self._texture_correlation(gray1, rotated)
            else:
                correlation = self._edge_correlation(gray1, rotated)
                
            if correlation > max_correlation:
                max_correlation = correlation
                best_angle = angle
                
        return max_correlation, best_angle
        
    def _texture_correlation(self, img1, img2):
        """Enhanced texture correlation using multiple metrics"""
        # Template matching (normalized cross-correlation)
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Structural similarity (simplified SSIM)
        ssim = self._calculate_ssim(img1, img2)
        
        # Mutual information (simplified)
        mi = self._calculate_mutual_information(img1, img2)
        
        # Combine metrics
        combined_score = max_val * 0.5 + ssim * 0.3 + mi * 0.2
        return combined_score
        
    def _edge_correlation(self, img1, img2):
        """Enhanced edge correlation"""
        # Direct template matching on edge images
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Edge density similarity
        edge1_density = np.mean(img1 > 0.2)
        edge2_density = np.mean(img2 > 0.2)
        density_similarity = 1.0 - abs(edge1_density - edge2_density)
        
        # Combine metrics
        combined_score = max_val * 0.7 + density_similarity * 0.3
        return combined_score
        
    def _calculate_ssim(self, img1, img2):
        """Simplified Structural Similarity Index"""
        # Mean values
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # Variances and covariance
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM calculation
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        
        return numerator / denominator if denominator > 0 else 0.0
        
    def _calculate_mutual_information(self, img1, img2):
        """Simplified mutual information calculation"""
        # Quantize images to reduce computation
        bins = 32
        img1_q = (img1 * (bins - 1)).astype(np.uint8)
        img2_q = (img2 * (bins - 1)).astype(np.uint8)
        
        # Joint histogram
        joint_hist = np.histogram2d(img1_q.flatten(), img2_q.flatten(), bins=bins)[0]
        joint_hist = joint_hist / np.sum(joint_hist)
        
        # Marginal histograms
        hist1 = np.sum(joint_hist, axis=1)
        hist2 = np.sum(joint_hist, axis=0)
        
        # Calculate MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 0 and hist1[i] > 0 and hist2[j] > 0:
                    mi += joint_hist[i, j] * np.log(joint_hist[i, j] / (hist1[i] * hist2[j]))
                    
        return mi / np.log(2)  # Convert to bits

class ZoomableViewer(QDialog):
    """Enhanced comparison viewer with better controls"""
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
        layout = QVBoxLayout(self)
        
        # Info panel
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        
        # Score info
        score_label = QLabel(f"📊 <b>Score: {score:.4f}</b>")
        score_label.setStyleSheet("font-size: 14px; color: #2c3e50;")
        
        # Metadata info
        if self.metadata:
            meta_text = f"T:{self.metadata.get('texture_score', 0):.3f} " \
                       f"E:{self.metadata.get('edge_score', 0):.3f} " \
                       f"Q:{self.metadata.get('quality_score', 0):.3f}"
            meta_label = QLabel(f"📈 {meta_text}")
            meta_label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        else:
            meta_label = QLabel()
            
        # Comparison info
        comp_label = QLabel(f"🔄 {self.s1.unique_id} vs {self.s2.unique_id} (↻{self.angle}°)")
        comp_label.setStyleSheet("font-size: 12px; color: #34495e;")
        
        info_layout.addWidget(score_label)
        info_layout.addWidget(meta_label)
        info_layout.addStretch()
        info_layout.addWidget(comp_label)
        
        layout.addWidget(info_widget)
        
        # Image viewer
        self.viewer = QLabel()
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 8px;")
        layout.addWidget(self.viewer)
        
        # Controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        zoom_label = QLabel("🔍 Zoom:")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self._reset_zoom)
        
        controls_layout.addWidget(zoom_label)
        controls_layout.addWidget(self.zoom_slider)
        controls_layout.addWidget(reset_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls)
        
    def _on_zoom_changed(self, value):
        self.zoom = value / 100.0
        self.display_pair()
        
    def _reset_zoom(self):
        self.zoom = 1.0
        self.zoom_slider.setValue(100)
        self.display_pair()
        
    def display_pair(self):
        if self.s1.processed_img is None or self.s2.processed_img is None:
            return
            
        img1 = self.s1.processed_img.copy()
        img2 = self.s2.processed_img.copy()
        
        # Rotate second image
        center = (img2.shape[1] // 2, img2.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(img2, M, img2.shape[:2])
        
        # Create comparison view
        combined = cv2.hconcat([img1, rotated])
        
        # Add separator line
        h, w = combined.shape[:2]
        cv2.line(combined, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        
        # Convert to QPixmap
        h, w, ch = combined.shape
        q_img = QImage(combined.data, w, h, ch * w, QImage.Format.Format_BGR888)
        
        # Scale according to zoom
        scaled_size = QSize(int(w * self.zoom), int(h * self.zoom))
        pixmap = QPixmap.fromImage(q_img).scaled(
            scaled_size, 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.viewer.setPixmap(pixmap)
        
    def wheelEvent(self, event):
        # Mouse wheel zoom
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1/1.15
        new_zoom = self.zoom * zoom_factor
        
        if 0.25 <= new_zoom <= 4.0:
            self.zoom = new_zoom
            self.zoom_slider.setValue(int(self.zoom * 100))
            self.display_pair()

class EnhancedHeatmapViewer(QDialog):
    """Enhanced heatmap viewer with better interactivity"""
    def __init__(self, subsquares, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🌡️ Correlation Heatmap")
        self.setMinimumSize(900, 700)
        self.resize(1200, 800)
        
        self.subsquares = subsquares
        self.data = data
        
        self._setup_ui()
        self.display_heatmap()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        # Colorscale selection
        controls_layout.addWidget(QLabel("🎨 Colorscale:"))
        self.colorscale_combo = QComboBox()
        self.colorscale_combo.addItems([
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'RdYlBu', 'RdBu', 'Spectral', 'Turbo'
        ])
        self.colorscale_combo.currentTextChanged.connect(self.display_heatmap)
        controls_layout.addWidget(self.colorscale_combo)
        
        # Threshold slider
        controls_layout.addWidget(QLabel("🎯 Min Score:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.display_heatmap)
        controls_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.00")
        controls_layout.addWidget(self.threshold_label)
        
        controls_layout.addStretch()
        
        # Stats
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 11px; color: #666;")
        controls_layout.addWidget(self.stats_label)
        
        layout.addWidget(controls)
        
        # Web view for heatmap
        self.webview = QWebEngineView()
        self.channel = QWebChannel()
        self.bridge = WebBridge(self)
        
        self.channel.registerObject("py_bridge", self.bridge)
        self.webview.page().setWebChannel(self.channel)
        self.bridge.heatmap_clicked.connect(self.on_heatmap_clicked)
        
        layout.addWidget(self.webview)
        
    def display_heatmap(self):
        # Extract scores and apply threshold
        threshold = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
        scores = np.array([[d[0] if isinstance(d, tuple) else d for d in row] for row in self.data])
        
        # Apply threshold
        scores_filtered = np.where(scores >= threshold, scores, np.nan)
        
        # Calculate statistics
        valid_scores = scores[~np.isnan(scores_filtered)]
        if len(valid_scores) > 0:
            stats_text = f"📊 Mean: {np.mean(valid_scores):.3f} | " \
                        f"Max: {np.max(valid_scores):.3f} | " \
                        f"Valid: {len(valid_scores)}/{scores.size}"
        else:
            stats_text = "📊 No data above threshold"
            
        self.stats_label.setText(stats_text)
        
        # Create labels
        labels = [s.unique_id for s in self.subsquares]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=scores_filtered,
            x=labels,
            y=labels,
            colorscale=self.colorscale_combo.currentText(),
            showscale=True,
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Score: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "🌡️ Cross-Correlation Matrix (Click to Compare)",
                'x': 0.5,
                'font': {'size': 16}
            },
            yaxis_autorange='reversed',
            width=1000,
            height=600,
            font=dict(size=10)
        )
        
        # JavaScript for interactivity
        js_code = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        var py_bridge;
        new QWebChannel(qt.webChannelTransport, function(channel) {
            py_bridge = channel.objects.py_bridge;
        });
        
        document.addEventListener('plotly_click', function(data) {
            if (data.points && data.points.length > 0) {
                var point = data.points[0];
                py_bridge.on_heatmap_click(point.y, point.x);
            }
        });
        </script>
        """
        
        html_content = fig.to_html(include_plotlyjs='cdn') + js_code
        self.webview.setHtml(html_content)
        
    def on_heatmap_clicked(self, i, j):
        if i < len(self.subsquares) and j < len(self.subsquares):
            s1, s2 = self.subsquares[i], self.subsquares[j]
            
            if isinstance(self.data[i, j], tuple):
                score, angle, metadata = self.data[i, j]
            else:
                score, angle, metadata = self.data[i, j], 0, {}
                
            viewer = ZoomableViewer(s1, s2, score, angle, metadata, self)
            viewer.exec()

class TopPairsGallery(QDialog):
    """Enhanced gallery with filtering and sorting options"""
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
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        # Number of pairs
        controls_layout.addWidget(QLabel("📈 Show top:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(10, 1000)
        self.count_spin.setValue(50)
        self.count_spin.valueChanged.connect(self.update_display)
        controls_layout.addWidget(self.count_spin)
        
        # Minimum score filter
        controls_layout.addWidget(QLabel("🎯 Min score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setValue(0.0)
        self.min_score_spin.setSingleStep(0.05)
        self.min_score_spin.valueChanged.connect(self.update_display)
        controls_layout.addWidget(self.min_score_spin)
        
        # Sort by quality
        self.quality_check = QCheckBox("Sort by quality")
        self.quality_check.toggled.connect(self.update_display)
        controls_layout.addWidget(self.quality_check)
        
        controls_layout.addStretch()
        
        # Stats
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 11px; color: #666;")
        controls_layout.addWidget(self.stats_label)
        
        layout.addWidget(controls)
        
        # Gallery
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        
        self.scroll_area.setWidget(self.gallery_widget)
        layout.addWidget(self.scroll_area)
        
    def update_display(self):
        # Clear existing items
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        n = len(self.subsquares)
        min_score = self.min_score_spin.value()
        
        # Collect all pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                data_item = self.data[i, j]
                
                if isinstance(data_item, tuple):
                    score, angle, metadata = data_item
                    quality = metadata.get('quality_score', 0)
                else:
                    score, angle, quality = data_item, 0, 0
                    
                if score >= min_score:
                    pairs.append((i, j, score, angle, quality))
                    
        # Sort pairs
        if self.quality_check.isChecked():
            pairs.sort(key=lambda x: (x[2] * x[4]), reverse=True)  # score * quality
        else:
            pairs.sort(key=lambda x: x[2], reverse=True)  # just score
            
        # Limit number of pairs
        pairs = pairs[:self.count_spin.value()]
        
        # Update stats
        if pairs:
            avg_score = np.mean([p[2] for p in pairs])
            max_score = max(p[2] for p in pairs)
            stats_text = f"📊 Showing {len(pairs)} pairs | Avg: {avg_score:.3f} | Max: {max_score:.3f}"
        else:
            stats_text = "📊 No pairs match criteria"
            
        self.stats_label.setText(stats_text)
        
        # Create gallery items
        for idx, (i, j, score, angle, quality) in enumerate(pairs):
            pair_widget = self._create_pair_widget(i, j, score, angle, quality, idx + 1)
            self.gallery_layout.addWidget(pair_widget)
            
        self.gallery_layout.addStretch()
        
    def _create_pair_widget(self, i, j, score, angle, quality, rank):
        """Create a widget for displaying a pair"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setStyleSheet("""
            QFrame {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #f8f9fa;
            }
            QFrame:hover {
                border-color: #3498db;
                background-color: #e3f2fd;
            }
        """)
        
        layout = QHBoxLayout(widget)
        
        # Rank badge
        rank_label = QLabel(f"#{rank}")
        rank_label.setStyleSheet("""
            QLabel {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        rank_label.setFixedSize(40, 30)
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Image previews
        s1, s2 = self.subsquares[i], self.subsquares[j]
        
        # Create combined preview
        if s1.processed_img is not None and s2.processed_img is not None:
            img1 = s1.processed_img.copy()
            img2 = s2.processed_img.copy()
            
            # Rotate second image
            center = (img2.shape[1] // 2, img2.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img2, M, img2.shape[:2])
            
            # Combine images
            combined = cv2.hconcat([img1, rotated])
            
            # Convert to QPixmap
            h, w, ch = combined.shape
            q_img = QImage(combined.data, w, h, ch * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                QSize(200, 100),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            preview_label = QLabel()
            preview_label.setPixmap(pixmap)
            preview_label.setStyleSheet("border: 1px solid #ccc; border-radius: 4px;")
        else:
            preview_label = QLabel("No preview")
            preview_label.setFixedSize(200, 100)
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
            
        # Info panel
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(5)
        
        # Title
        title_label = QLabel(f"<b>{s1.grid_id} ↔ {s2.grid_id}</b>")
        title_label.setStyleSheet("font-size: 14px; color: #2c3e50;")
        
        # Score info
        score_label = QLabel(f"📊 Score: <b>{score:.4f}</b>")
        score_label.setStyleSheet("font-size: 12px; color: #27ae60;")
        
        # Additional info
        angle_label = QLabel(f"🔄 Rotation: {angle}°")
        quality_label = QLabel(f"⭐ Quality: {quality:.3f}")
        
        for label in [angle_label, quality_label]:
            label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
            
        info_layout.addWidget(title_label)
        info_layout.addWidget(score_label)
        info_layout.addWidget(angle_label)
        info_layout.addWidget(quality_label)
        info_layout.addStretch()
        
        # View button
        view_btn = QPushButton("🔍 View Details")
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        view_btn.clicked.connect(lambda: self._view_pair(i, j, score, angle))
        
        # Layout assembly
        layout.addWidget(rank_label)
        layout.addWidget(preview_label)
        layout.addWidget(info_widget, 1)
        layout.addWidget(view_btn)
        
        return widget
        
    def _view_pair(self, i, j, score, angle):
        """Open detailed view for a pair"""
        s1, s2 = self.subsquares[i], self.subsquares[j]
        
        data_item = self.data[i, j]
        metadata = data_item[2] if isinstance(data_item, tuple) and len(data_item) > 2 else {}
        
        viewer = ZoomableViewer(s1, s2, score, angle, metadata, self)
        viewer.exec()

class InteractiveImageViewer(QLabel):
    """Enhanced image viewer with better interaction"""
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
        
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        
        # Instructions
        self.setToolTip(
            "🖱️ Left-click & drag: Select area\n"
            "🖱️ Right-click & drag: Pan image\n"
            "🖱️ Scroll wheel: Zoom in/out\n"
            "Double-click: Reset view"
        )
        
    def set_image(self, path):
        """Load and display image"""
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
        
        # Calculate scaled size
        scaled_size = self._pixmap.size() * self.zoom_factor
        
        # Calculate position (centered + pan offset)
        pos = QPoint(
            (self.width() - scaled_size.width()) // 2 + self.pan_offset.x(),
            (self.height() - scaled_size.height()) // 2 + self.pan_offset.y()
        )
        
        # Draw image
        painter.drawPixmap(
            QRect(pos, scaled_size),
            self._pixmap,
            self._pixmap.rect()
        )
        
        # Draw selection rectangle
        if self.start_pos and self.end_pos:
            painter.setPen(QPen(QColor("#e74c3c"), 3))
            painter.setBrush(QColor(231, 76, 60, 50))  # Semi-transparent red
            
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.drawRect(rect)
            
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
                # Convert screen coordinates to image coordinates
                scaled_size = self._pixmap.size() * self.zoom_factor
                pos = QPoint(
                    (self.width() - scaled_size.width()) // 2 + self.pan_offset.x(),
                    (self.height() - scaled_size.height()) // 2 + self.pan_offset.y()
                )
                
                # Calculate relative position in original image
                rel_x = (rect.x() - pos.x()) / self.zoom_factor
                rel_y = (rect.y() - pos.y()) / self.zoom_factor
                rel_w = rect.width() / self.zoom_factor
                rel_h = rect.height() / self.zoom_factor
                
                # Ensure coordinates are within image bounds
                if (rel_x >= 0 and rel_y >= 0 and 
                    rel_x + rel_w <= self._pixmap.width() and 
                    rel_y + rel_h <= self._pixmap.height()):
                    
                    self.square_selected.emit(int(rel_x), int(rel_y), int(rel_w), int(rel_h))
                    
            self.start_pos = None
            self.update()
            
    def wheelEvent(self, event):
        # Zoom with mouse wheel
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1/1.15
        
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor * zoom_factor))
        
        # Adjust pan to zoom towards mouse position
        if self.zoom_factor != old_zoom:
            mouse_pos = event.position().toPoint()
            zoom_ratio = self.zoom_factor / old_zoom
            
            self.pan_offset = QPoint(
                int((self.pan_offset.x() - mouse_pos.x()) * zoom_ratio + mouse_pos.x()),
                int((self.pan_offset.y() - mouse_pos.y()) * zoom_ratio + mouse_pos.y())
            )
            
        self.update()
        
    def mouseDoubleClickEvent(self, event):
        # Reset view
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

class EnhancedImageAnalysisApp(QMainWindow):
    """Main application with enhanced UI and features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 2D Class Average Analysis Tool v2.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.image_paths = []
        self.subsquares = []
        self.correlation_data = None
        self.ref_square = None
        self.thread = None
        
        # Apply modern styling
        self._apply_modern_style()
        self._init_ui()
        self._setup_status_bar()
        
    def _apply_modern_style(self):
        """Apply modern dark/light theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 5px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #3498db;
            }
            QScrollArea {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                background-color: white;
            }
        """)
        
    def _init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self._create_left_panel()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        
        # Right panel (results)
        right_panel = self._create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Initial sizes
        
    def _create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File operations
        file_group = QGroupBox("📁 File Operations")
        file_layout = QGridLayout(file_group)
        
        self.load_btn = QPushButton("📂 Load Images")
        self.load_btn.clicked.connect(self._load_images)
        self.load_btn.setToolTip("Load multiple image files for analysis")
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        self.clear_btn.setToolTip("Clear all loaded data")
        
        file_layout.addWidget(self.load_btn, 0, 0, 1, 2)
        file_layout.addWidget(self.clear_btn, 1, 0, 1, 2)
        
        # Detection mode
        mode_group = QGroupBox("🔍 Detection Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.auto_radio = QRadioButton("🤖 Automatic Detection")
        self.ref_radio = QRadioButton("🎯 Reference-Based")
        self.auto_radio.setChecked(True)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.ref_radio)
        self.mode_group.buttonToggled.connect(self._update_ui_states)
        
        self.select_ref_btn = QPushButton("🎯 Select Reference Square")
        self.select_ref_btn.clicked.connect(self._select_reference)
        self.select_ref_btn.setToolTip("Select a reference square for template matching")
        
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.ref_radio)
        mode_layout.addWidget(self.select_ref_btn)
        
        # Parameters
        params_group = QGroupBox("⚙️ Parameters")
        params_layout = QGridLayout(params_group)
        
        # Min square size
        params_layout.addWidget(QLabel("Min Size:"), 0, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 500)
        self.min_size_spin.setValue(30)
        self.min_size_spin.setSuffix(" px")
        params_layout.addWidget(self.min_size_spin, 0, 1)
        
        # Target size
        params_layout.addWidget(QLabel("Target Size:"), 1, 0)
        self.target_size_spin = QSpinBox()
        self.target_size_spin.setRange(32, 512)
        self.target_size_spin.setValue(128)
        self.target_size_spin.setSuffix(" px")
        params_layout.addWidget(self.target_size_spin, 1, 1)
        
        # Reference threshold
        params_layout.addWidget(QLabel("Ref Threshold:"), 2, 0)
        self.ref_thresh_spin = QDoubleSpinBox()
        self.ref_thresh_spin.setRange(0.1, 1.0)
        self.ref_thresh_spin.setValue(0.7)
        self.ref_thresh_spin.setSingleStep(0.05)
        self.ref_thresh_spin.setDecimals(2)
        params_layout.addWidget(self.ref_thresh_spin, 2, 1)
        
        # Processing controls
        process_group = QGroupBox("🚀 Processing")
        process_layout = QGridLayout(process_group)
        
        self.detect_btn = QPushButton("🔍 Detect Squares")
        self.detect_btn.clicked.connect(self._start_detection)
        self.detect_btn.setToolTip("Start automatic square detection")
        
        self.analyze_btn = QPushButton("📊 Analyze Correlations")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.analyze_btn.setToolTip("Calculate cross-correlations between squares")
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setToolTip("Stop current processing")
        
        process_layout.addWidget(self.detect_btn, 0, 0)
        process_layout.addWidget(self.analyze_btn, 0, 1)
        process_layout.addWidget(self.stop_btn, 1, 0, 1, 2)
        
        # Progress
        progress_group = QGroupBox("📈 Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = EnhancedProgressBar()
        self.loading_widget = AnimatedLoadingWidget()
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.loading_widget)
        
        # Results
        results_group = QGroupBox("📋 Results")
        results_layout = QVBoxLayout(results_group)
        
        self.heatmap_btn = QPushButton("🌡️ Show Heatmap")
        self.heatmap_btn.clicked.connect(self._show_heatmap)
        self.heatmap_btn.setToolTip("Display correlation heatmap")
        
        self.gallery_btn = QPushButton("🏆 Show Top Pairs")
        self.gallery_btn.clicked.connect(self._show_gallery)
        self.gallery_btn.setToolTip("View gallery of best matches")
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setToolTip("Export correlation data to CSV")
        
        results_layout.addWidget(self.heatmap_btn)
        results_layout.addWidget(self.gallery_btn)
        results_layout.addWidget(self.export_btn)
        
        # Add all groups to layout
        layout.addWidget(file_group)
        layout.addWidget(mode_group)
        layout.addWidget(params_group)
        layout.addWidget(process_group)
        layout.addWidget(progress_group)
        layout.addWidget(results_group)
        layout.addStretch()
        
        return panel
        
    def _create_right_panel(self):
        """Create the right results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("📊 Analysis Results")
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Tabbed interface for results
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
            }
        """)
        
        # Detection results tab
        self.detection_scroll = QScrollArea()
        self.detection_scroll.setWidgetResizable(True)
        self.detection_widget = QWidget()
        self.detection_layout = QVBoxLayout(self.detection_widget)
        self.detection_scroll.setWidget(self.detection_widget)
        
        # Statistics tab
        self.stats_widget = QTextEdit()
        self.stats_widget.setReadOnly(True)
        self.stats_widget.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 11px;
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                border-radius: 4px;
            }
        """)
        
        self.results_tabs.addTab(self.detection_scroll, "🔍 Detected Squares")
        self.results_tabs.addTab(self.stats_widget, "📈 Statistics")
        
        layout.addWidget(self.results_tabs)
        
        return panel
        
    def _setup_status_bar(self):
        """Setup enhanced status bar"""
        status_bar = self.statusBar()
        
        # Main status label
        self.status_label = QLabel("🚀 Ready to analyze images!")
        self.status_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        
        # System monitor
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()
        
        # Add widgets to status bar
        status_bar.addWidget(self.status_label, 1)
        status_bar.addPermanentWidget(self.system_monitor)
        
        # Initial state update
        self._update_ui_states()
        
    def _update_ui_states(self):
        """Update UI element states based on current conditions"""
        has_images = bool(self.image_paths)
        has_squares = bool(self.subsquares)
        has_correlation = self.correlation_data is not None
        is_processing = self.thread is not None and self.thread.isRunning()
        
        # File operations
        self.load_btn.setEnabled(not is_processing)
        self.clear_btn.setEnabled(has_images and not is_processing)
        
        # Detection
        self.detect_btn.setEnabled(has_images and not is_processing)
        self.select_ref_btn.setEnabled(
            self.ref_radio.isChecked() and has_images and not is_processing
        )
        
        # Analysis
        self.analyze_btn.setEnabled(has_squares and not is_processing)
        self.stop_btn.setEnabled(is_processing)
        
        # Results
        self.heatmap_btn.setEnabled(has_correlation and not is_processing)
        self.gallery_btn.setEnabled(has_correlation and not is_processing)
        self.export_btn.setEnabled(has_correlation and not is_processing)
        
        # Reference threshold visibility
        self.ref_thresh_spin.setVisible(self.ref_radio.isChecked())
        
        # Animation
        if is_processing:
            self.loading_widget.start_animation()
            self.progress_bar.setVisible(True)
        else:
            self.loading_widget.stop_animation()
            self.progress_bar.setVisible(False)
            
    def _load_images(self):
        """Load image files"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        
        if file_dialog.exec():
            paths = file_dialog.selectedFiles()
            if paths:
                self.image_paths = sorted(paths)
                self.ref_square = None
                self._clear_results()
                
                self.status_label.setText(f"📁 Loaded {len(paths)} images")
                self._update_ui_states()
                self._update_stats()
                
    def _clear_all(self):
        """Clear all data"""
        self.image_paths.clear()
        self.subsquares.clear()
        self.correlation_data = None
        self.ref_square = None
        
        self._clear_results()
        self.status_label.setText("🗑️ All data cleared")
        self._update_ui_states()
        self._update_stats()
        
    def _clear_results(self):
        """Clear result displays"""
        # Clear detection results
        while self.detection_layout.count():
            child = self.detection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Clear stats
        self.stats_widget.clear()
        
    def _select_reference(self):
        """Select reference square for template matching"""
        if not self.image_paths:
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("🎯 Select Reference Square")
        dialog.setMinimumSize(900, 700)
        dialog.resize(1000, 800)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("""
        <b>📋 Instructions:</b><br>
        1. 📂 Choose an image from the dropdown<br>
        2. 🖱️ Right-click & drag to pan the image<br>
        3. 🖱️ Scroll wheel to zoom in/out<br>
        4. 🖱️ Left-click & drag to select reference square<br>
        5. ✅ Click OK to confirm selection
        """)
        instructions.setStyleSheet("""
            QLabel {
                background-color: #e8f4f8;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        layout.addWidget(instructions)
        
        # Image selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("📁 Image:"))
        
        image_combo = QComboBox()
        image_combo.addItems([os.path.basename(p) for p in self.image_paths])
        selector_layout.addWidget(image_combo)
        selector_layout.addStretch()
        
        layout.addLayout(selector_layout)
        
        # Image viewer
        viewer = InteractiveImageViewer()
        viewer.setMinimumHeight(400)
        layout.addWidget(viewer)
        
        # Selection info
        self.selection_info = QLabel("📏 No selection")
        self.selection_info.setStyleSheet("font-weight: bold; color: #e74c3c;")
        layout.addWidget(self.selection_info)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Connect signals
        selection = {}
        
        def on_image_changed(index):
            viewer.set_image(self.image_paths[index])
            selection.clear()
            self.selection_info.setText("📏 No selection")
            
        def on_selection(x, y, w, h):
            selection.update({'rect': (x, y, w, h)})
            self.selection_info.setText(f"📏 Selected: {w}×{h} at ({x}, {y})")
            
            # Auto-adjust parameters
            self.target_size_spin.setValue(max(w, h))
            self.min_size_spin.setValue(int(min(w, h) * 0.8))
            
        image_combo.currentIndexChanged.connect(on_image_changed)
        viewer.square_selected.connect(on_selection)
        
        # Initialize
        viewer.set_image(self.image_paths[0])
        
        # Execute dialog
        if dialog.exec() == QDialog.DialogCode.Accepted and 'rect' in selection:
            self.ref_square = (self.image_paths[image_combo.currentIndex()], *selection['rect'])
            self.status_label.setText("🎯 Reference square selected")
            
    def _start_detection(self):
        """Start square detection process"""
        self._clear_results()
        self.subsquares.clear()
        self.correlation_data = None
        
        params = self._get_detection_parameters()
        ref_square = self.ref_square if self.ref_radio.isChecked() else None
        
        self.thread = ImageProcessor(self.image_paths, params, ref_square)
        self.thread.progress.connect(self.status_label.setText)
        self.thread.progress_value.connect(self.progress_bar.setValue)
        self.thread.finished_detection.connect(self._on_detection_finished)
        self.thread.finished.connect(self._on_thread_finished)
        
        self.progress_bar.setRange(0, 100)
        self.thread.start()
        self._update_ui_states()
        
    def _start_analysis(self):
        """Start correlation analysis"""
        if not self.subsquares:
            return
            
        self.thread = EnhancedCorrelationProcessor(self.subsquares)
        self.thread.progress.connect(self.status_label.setText)
        self.thread.progress_value.connect(self.progress_bar.setValue)
        self.thread.finished_correlation.connect(self._on_analysis_finished)
        self.thread.finished.connect(self._on_thread_finished)
        
        self.progress_bar.setRange(0, 100)
        self.thread.start()
        self._update_ui_states()
        
    def _stop_processing(self):
        """Stop current processing"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.status_label.setText("⏹️ Stopping process...")
            
    def _on_detection_finished(self, squares):
        """Handle detection completion"""
        self.subsquares = sorted(squares, key=lambda s: s.unique_id)
        self._display_detected_squares()
        self._update_stats()
        
    def _on_analysis_finished(self, data):
        """Handle analysis completion"""
        self.correlation_data = data
        self._update_stats()
        
    def _on_thread_finished(self):
        """Handle thread completion"""
        self.status_label.setText("✅ Process completed successfully!")
        self.thread = None
        self._update_ui_states()
        
    def _display_detected_squares(self):
        """Display detected squares in the results panel"""
        # Group squares by image
        groups = {}
        for square in self.subsquares:
            path = square.original_image_path
            if path not in groups:
                groups[path] = []
            groups[path].append(square)
            
        # Display each group
        for image_path in sorted(groups.keys()):
            squares = groups[image_path]
            
            # Image header
            header = QLabel(f"📁 <b>{os.path.basename(image_path)}</b> ({len(squares)} squares)")
            header.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    border: 1px solid #bdc3c7;
                    border-radius: 6px;
                    padding: 8px;
                    margin: 5px 0;
                }
            """)
            self.detection_layout.addWidget(header)
            
            # Thumbnail gallery
            gallery_widget = QWidget()
            gallery_layout = QHBoxLayout(gallery_widget)
            gallery_layout.setSpacing(5)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFixedHeight(150)
            scroll_area.setWidget(gallery_widget)
            
            for square in squares:
                thumb_widget = self._create_thumbnail_widget(square)
                gallery_layout.addWidget(thumb_widget)
                
            gallery_layout.addStretch()
            self.detection_layout.addWidget(scroll_area)
            
        self.detection_layout.addStretch()
        
    def _create_thumbnail_widget(self, square):
        """Create thumbnail widget for a square"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setFixedSize(120, 140)
        widget.setStyleSheet("""
            QFrame {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: white;
                margin: 2px;
            }
            QFrame:hover {
                border-color: #3498db;
                background-color: #e3f2fd;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail image
        thumb_label = QLabel()
        pixmap = square.to_qpixmap(size=100)
        thumb_label.setPixmap(pixmap)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumb_label)
        
        # Info
        info_label = QLabel(f"<b>{square.grid_id}</b>")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-size: 11px; color: #2c3e50;")
        layout.addWidget(info_label)
        
        # Quality indicator
        quality_color = "#27ae60" if square.overall_quality > 0.7 else "#f39c12" if square.overall_quality > 0.4 else "#e74c3c"
        quality_label = QLabel(f"⭐ {square.overall_quality:.2f}")
        quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quality_label.setStyleSheet(f"font-size: 10px; color: {quality_color}; font-weight: bold;")
        layout.addWidget(quality_label)
        
        return widget
        
    def _update_stats(self):
        """Update statistics display"""
        stats_text = "📊 ANALYSIS STATISTICS\n"
        stats_text += "=" * 50 + "\n\n"
        
        # Image statistics
        stats_text += f"📁 Images loaded: {len(self.image_paths)}\n"
        if self.image_paths:
            total_size = sum(os.path.getsize(p) for p in self.image_paths if os.path.exists(p))
            stats_text += f"💾 Total size: {total_size / (1024*1024):.1f} MB\n"
            
        stats_text += "\n"
        
        # Square statistics
        stats_text += f"🔍 Squares detected: {len(self.subsquares)}\n"
        if self.subsquares:
            qualities = [s.overall_quality for s in self.subsquares]
            texture_qualities = [s.texture_quality for s in self.subsquares]
            edge_qualities = [s.edge_quality for s in self.subsquares]
            
            stats_text += f"⭐ Average quality: {np.mean(qualities):.3f}\n"
            stats_text += f"📈 Quality range: {np.min(qualities):.3f} - {np.max(qualities):.3f}\n"
            stats_text += f"🎨 Texture quality: {np.mean(texture_qualities):.3f}\n"
            stats_text += f"📐 Edge quality: {np.mean(edge_qualities):.3f}\n"
            
        stats_text += "\n"
        
        # Correlation statistics
        if self.correlation_data is not None:
            # Extract correlation scores
            n = len(self.subsquares)
            scores = []
            texture_scores = []
            edge_scores = []
            
            for i in range(n):
                for j in range(i+1, n):
                    data_item = self.correlation_data[i, j]
                    if isinstance(data_item, tuple):
                        score, _, metadata = data_item
                        scores.append(score)
                        texture_scores.append(metadata.get('texture_score', 0))
                        edge_scores.append(metadata.get('edge_score', 0))
                    else:
                        scores.append(data_item)
                        
            if scores:
                stats_text += f"🔗 Correlation pairs: {len(scores)}\n"
                stats_text += f"📊 Average correlation: {np.mean(scores):.3f}\n"
                stats_text += f"📈 Correlation range: {np.min(scores):.3f} - {np.max(scores):.3f}\n"
                stats_text += f"🏆 High correlations (>0.8): {sum(1 for s in scores if s > 0.8)}\n"
                stats_text += f"⚠️ Low correlations (<0.3): {sum(1 for s in scores if s < 0.3)}\n"
                
                if texture_scores and edge_scores:
                    stats_text += f"🎨 Avg texture score: {np.mean(texture_scores):.3f}\n"
                    stats_text += f"📐 Avg edge score: {np.mean(edge_scores):.3f}\n"
                    
        stats_text += "\n"
        
        # System information
        stats_text += "🖥️ SYSTEM INFO\n"
        stats_text += "-" * 30 + "\n"
        stats_text += f"💻 CPU cores: {os.cpu_count()}\n"
        stats_text += f"🧠 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB\n"
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats_text += f"🎮 GPU: {gpu.name}\n"
                stats_text += f"📊 GPU Memory: {gpu.memoryTotal} MB\n"
        except:
            stats_text += f"🎮 GPU: Not detected\n"
            
        self.stats_widget.setText(stats_text)
        
    def _get_detection_parameters(self):
        """Get current detection parameters"""
        return {
            'min_square_size': self.min_size_spin.value(),
            'target_subsquare_size': (self.target_size_spin.value(), self.target_size_spin.value()),
            'ref_threshold': self.ref_thresh_spin.value()
        }
        
    def _show_heatmap(self):
        """Show correlation heatmap"""
        if self.correlation_data is not None:
            heatmap_viewer = EnhancedHeatmapViewer(self.subsquares, self.correlation_data, self)
            heatmap_viewer.exec()
            
    def _show_gallery(self):
        """Show top pairs gallery"""
        if self.correlation_data is not None:
            gallery = TopPairsGallery(self.correlation_data, self.subsquares, self)
            gallery.exec()
            
    def _export_results(self):
        """Export correlation results to CSV"""
        if self.correlation_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "correlation_results.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Header
                    writer.writerow([
                        'Square1_ID', 'Square1_Path', 'Square1_Grid',
                        'Square2_ID', 'Square2_Path', 'Square2_Grid',
                        'Correlation_Score', 'Rotation_Angle',
                        'Texture_Score', 'Edge_Score', 'Quality_Score',
                        'Texture_Weight', 'Edge_Weight'
                    ])
                    
                    # Data
                    n = len(self.subsquares)
                    for i in range(n):
                        for j in range(i + 1, n):
                            s1, s2 = self.subsquares[i], self.subsquares[j]
                            data_item = self.correlation_data[i, j]
                            
                            if isinstance(data_item, tuple):
                                score, angle, metadata = data_item
                                texture_score = metadata.get('texture_score', 0)
                                edge_score = metadata.get('edge_score', 0)
                                quality_score = metadata.get('quality_score', 0)
                                texture_weight = metadata.get('texture_weight', 0.5)
                                edge_weight = metadata.get('edge_weight', 0.5)
                            else:
                                score, angle = data_item, 0
                                texture_score = edge_score = quality_score = 0
                                texture_weight = edge_weight = 0.5
                                
                            writer.writerow([
                                s1.unique_id, os.path.basename(s1.original_image_path), s1.grid_id,
                                s2.unique_id, os.path.basename(s2.original_image_path), s2.grid_id,
                                f"{score:.6f}", f"{angle:.1f}",
                                f"{texture_score:.6f}", f"{edge_score:.6f}", f"{quality_score:.6f}",
                                f"{texture_weight:.6f}", f"{edge_weight:.6f}"
                            ])
                            
                self.status_label.setText(f"💾 Results exported to {os.path.basename(file_path)}")
                
            except Exception as e:
                self.status_label.setText(f"❌ Export failed: {str(e)}")
                
    def closeEvent(self, event):
        """Handle application close"""
        self._stop_processing()
        self.system_monitor.stop_monitoring()
        
        # Wait for thread to finish
        if self.thread and self.thread.isRunning():
            self.thread.wait(3000)  # Wait up to 3 seconds
            
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("2D Correlation Analysis Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Scientific Analysis Tools")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    # Create and show main window
    window = EnhancedImageAnalysisApp()
    window.show()
    
    # Add splash screen effect
    window.status_label.setText("🎉 Welcome to 2D Correlation Analysis Tool v2.0!")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
