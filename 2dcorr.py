import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QListWidget, QListWidgetItem,
    QProgressDialog, QSpinBox, QSlider, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QPoint
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
import plotly.graph_objects as go
from skimage.transform import warp_polar, rotate
from skimage.registration import phase_cross_correlation
from skimage.feature import match_template
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration not available. Install CuPy for GPU support: pip install cupy-cuda12x")

class Subsquare:
    """Represents a detected square region from a 2D class average image."""
    
    def __init__(self, image_id, original_image_path, original_subsquare_img, bbox, processed_img=None):
        self.image_id = image_id
        self.original_image_path = original_image_path
        self.original_subsquare_img = original_subsquare_img
        self.bbox = bbox  # (x, y, w, h)
        self.processed_img = processed_img if processed_img is not None else original_subsquare_img.copy()
        # Create a robust unique ID
        basename = os.path.basename(original_image_path).split('.')[0]
        self.unique_id = f"{basename}_x{bbox[0]}y{bbox[1]}w{bbox[2]}h{bbox[3]}"

    def to_qpixmap(self, img_data=None, size=None):
        """Convert image data to QPixmap for display."""
        img_to_display = img_data if img_data is not None else self.processed_img
        
        # Ensure image data is in uint8 format (0-255)
        if img_to_display.dtype != np.uint8:
            img_to_display = cv2.normalize(img_to_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Ensure image is 3-channel BGR for QImage
        if len(img_to_display.shape) == 2:
            img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_GRAY2BGR)

        h, w, ch = img_to_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_to_display.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        
        if size:
            qt_image = qt_image.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, 
                                     Qt.TransformationMode.SmoothTransformation)
        
        return QPixmap.fromImage(qt_image)

class ImageProcessor(QThread):
    """Main processing thread for detecting squares and calculating correlations."""
    
    # Signals for progress and results
    detection_progress = pyqtSignal(int, int, str)
    processing_progress = pyqtSignal(int, int, str)
    correlation_progress = pyqtSignal(int, int, str)
    finished_processing = pyqtSignal(list, np.ndarray)

    def __init__(self, image_paths, target_subsquare_size=(128, 128), min_square_size=30, 
                 aspect_ratio_tolerance=0.3, reference_square=None, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.target_subsquare_size = target_subsquare_size
        self.min_square_size = min_square_size
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.reference_square = reference_square  # (image_path, x, y, w, h)
        self.all_subsquares = []
        self.correlation_matrix = None
        self.use_gpu = GPU_AVAILABLE

    def run(self):
        """Main processing pipeline."""
        self.all_subsquares = []
        
        if self.reference_square:
            # Use reference-based detection
            self.detect_using_reference()
        else:
            # Use automatic detection
            self.detect_and_preprocess_all_images()
            
        if self.all_subsquares:
            self.calculate_all_correlations()
        self.finished_processing.emit(self.all_subsquares, self.correlation_matrix)

    def detect_using_reference(self):
        """Detect squares using the reference square as a template."""
        ref_img_path, ref_x, ref_y, ref_w, ref_h = self.reference_square
        
        # Load reference image and extract reference square
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            print(f"Error: Could not load reference image {ref_img_path}")
            return
            
        ref_square = ref_img[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]
        
        # Update target size based on reference
        self.target_subsquare_size = (ref_w, ref_h)
        self.min_square_size = min(ref_w, ref_h) * 0.7  # Allow 30% size variation
        
        # Add the reference square itself
        ref_subsquare = Subsquare(0, ref_img_path, ref_square, (ref_x, ref_y, ref_w, ref_h))
        self.all_subsquares.append(ref_subsquare)
        
        total_images = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths):
            self.detection_progress.emit(i + 1, total_images, 
                                       f"Template matching in {os.path.basename(img_path)}...")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Skip the reference image since we already added it
            if img_path == ref_img_path:
                continue
                
            subsquares_in_image = self._template_match_reference(
                img, ref_square, img_path, i + 1
            )
            self.all_subsquares.extend(subsquares_in_image)
        
        # Preprocess all detected subsquares
        total_subsquares = len(self.all_subsquares)
        for i, subsquare_obj in enumerate(self.all_subsquares):
            self.processing_progress.emit(i + 1, total_subsquares, 
                                        f"Processing subsquare {i+1}/{total_subsquares}...")
            self._preprocess_subsquare(subsquare_obj)
            self._resize_subsquare(subsquare_obj)

    def _template_match_reference(self, image, reference_template, image_path, image_id):
        """Find squares similar to the reference template."""
        subsquares = []
        
        # Convert to grayscale for template matching
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(reference_template, cv2.COLOR_BGR2GRAY)
        
        # Get template dimensions
        template_h, template_w = gray_template.shape
        
        # Try multiple scales to account for size variations
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        all_matches = []
        
        for scale in scales:
            # Resize template
            new_w = int(template_w * scale)
            new_h = int(template_h * scale)
            
            if new_w < 10 or new_h < 10 or new_w > gray_img.shape[1] or new_h > gray_img.shape[0]:
                continue
                
            scaled_template = cv2.resize(gray_template, (new_w, new_h))
            
            # Template matching
            result = cv2.matchTemplate(gray_img, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.6  # Adjust this threshold as needed
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = new_w, new_h
                score = result[y, x] if y < result.shape[0] and x < result.shape[1] else 0
                all_matches.append((x, y, w, h, score))
        
        # Remove overlapping detections (Non-Maximum Suppression)
        if all_matches:
            # Sort by score (descending)
            all_matches.sort(key=lambda x: x[4], reverse=True)
            
            filtered_matches = []
            for match in all_matches:
                x, y, w, h, score = match
                bbox = (x, y, w, h)
                
                # Check overlap with existing matches
                overlaps = False
                for existing in filtered_matches:
                    if self._calculate_overlap_ratio(bbox, existing[:4]) > 0.3:
                        overlaps = True
                        break
                
                if not overlaps:
                    filtered_matches.append(match)
                    
                    # Extract the square
                    if (x + w <= image.shape[1] and y + h <= image.shape[0] and 
                        x >= 0 and y >= 0):
                        square_img = image[y:y+h, x:x+w]
                        subsquares.append(Subsquare(image_id, image_path, square_img, bbox))
        
        return subsquares

    def detect_and_preprocess_all_images(self):
        """Detect squares in all images and preprocess them."""
        total_images = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths):
            self.detection_progress.emit(i + 1, total_images, 
                                       f"Detecting squares in {os.path.basename(img_path)}...")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not load image {img_path}")
                continue

            subsquares_in_image = self._detect_squares(img, img_path, i)
            self.all_subsquares.extend(subsquares_in_image)

        # Preprocess all detected subsquares
        total_subsquares = len(self.all_subsquares)
        for i, subsquare_obj in enumerate(self.all_subsquares):
            self.processing_progress.emit(i + 1, total_subsquares, 
                                        f"Processing subsquare {i+1}/{total_subsquares}...")
            self._preprocess_subsquare(subsquare_obj)
            self._resize_subsquare(subsquare_obj)

    def _detect_squares(self, image, image_path, image_id):
        """Detect square-like regions in an image using contour analysis."""
        subsquares = []
        
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple detection strategies
        subsquares.extend(self._detect_squares_contours(image, gray, image_path, image_id))
        subsquares.extend(self._detect_squares_template(image, gray, image_path, image_id))
        
        # Remove duplicates based on overlap
        subsquares = self._remove_duplicate_detections(subsquares)
        
        return subsquares

    def _detect_squares_contours(self, image, gray, image_path, image_id):
        """Detect squares using contour-based method."""
        subsquares = []
        
        # Apply different thresholding methods
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            cv2.Canny(gray, 50, 150)
        ]
        
        for binary in methods:
            # Apply morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Filter for 4-sided polygons (quadrilaterals)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Check if it's square-like and meets size requirements
                    if (abs(aspect_ratio - 1.0) <= self.aspect_ratio_tolerance and 
                        w >= self.min_square_size and h >= self.min_square_size):
                        subsquare_img = image[y:y+h, x:x+w]
                        subsquares.append(Subsquare(image_id, image_path, subsquare_img, (x, y, w, h)))
        
        return subsquares

    def _detect_squares_template(self, image, gray, image_path, image_id):
        """Detect squares using template matching with different square templates."""
        subsquares = []
        
        # Create square templates of different sizes
        template_sizes = [32, 48, 64, 96, 128]
        
        for size in template_sizes:
            # Create a simple square template (white square on black background)
            template = np.zeros((size, size), dtype=np.uint8)
            template[2:-2, 2:-2] = 255
            
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.3)  # Threshold for detection
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = size, size
                
                # Check if region is within image bounds
                if x + w <= image.shape[1] and y + h <= image.shape[0]:
                    subsquare_img = image[y:y+h, x:x+w]
                    subsquares.append(Subsquare(image_id, image_path, subsquare_img, (x, y, w, h)))
        
        return subsquares

    def _remove_duplicate_detections(self, subsquares):
        """Remove overlapping detections to avoid duplicates."""
        if not subsquares:
            return subsquares
        
        # Sort by area (larger first)
        subsquares.sort(key=lambda s: s.bbox[2] * s.bbox[3], reverse=True)
        
        filtered = []
        for current in subsquares:
            overlap_found = False
            for existing in filtered:
                # Calculate overlap
                overlap_ratio = self._calculate_overlap_ratio(current.bbox, existing.bbox)
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(current)
        
        return filtered

    def _calculate_overlap_ratio(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0

    def _preprocess_subsquare(self, subsquare_obj):
        """Apply preprocessing to enhance the subsquare for correlation analysis."""
        img = subsquare_obj.processed_img.copy()
        
        # Convert to grayscale for processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Noise reduction
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Convert back to BGR for consistency
        subsquare_obj.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _resize_subsquare(self, subsquare_obj):
        """Resize subsquare to uniform target size maintaining aspect ratio."""
        img = subsquare_obj.processed_img
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        target_w, target_h = self.target_subsquare_size
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # Resize
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        subsquare_obj.processed_img = padded

    def calculate_all_correlations(self):
        """Calculate rotation-invariant cross-correlation matrix for all subsquares."""
        num_subsquares = len(self.all_subsquares)
        if num_subsquares == 0:
            self.correlation_matrix = np.array([])
            return

        # Initialize correlation matrix
        self.correlation_matrix = np.zeros((num_subsquares, num_subsquares))
        
        # Calculate total number of unique pairs
        total_pairs = num_subsquares * (num_subsquares - 1) // 2
        processed_pairs = 0
        
        # Use parallel processing for correlation calculation
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            future_to_pair = {}
            
            for i in range(num_subsquares):
                for j in range(i + 1, num_subsquares):
                    future = executor.submit(
                        self._calculate_rotation_invariant_correlation,
                        self.all_subsquares[i].processed_img,
                        self.all_subsquares[j].processed_img
                    )
                    future_to_pair[future] = (i, j)
            
            for future in as_completed(future_to_pair):
                i, j = future_to_pair[future]
                try:
                    correlation_score = future.result()
                    self.correlation_matrix[i, j] = correlation_score
                    self.correlation_matrix[j, i] = correlation_score
                except Exception as exc:
                    print(f'Correlation calculation for pair ({i}, {j}) failed: {exc}')
                    self.correlation_matrix[i, j] = 0
                    self.correlation_matrix[j, i] = 0
                
                processed_pairs += 1
                self.correlation_progress.emit(
                    processed_pairs, total_pairs,
                    f"Calculating correlations: {processed_pairs}/{total_pairs}"
                )
        
        # Set diagonal to 1 (self-correlation)
        np.fill_diagonal(self.correlation_matrix, 1.0)

    def _calculate_rotation_invariant_correlation(self, img1, img2):
        """Calculate rotation-invariant correlation between two images."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if self.use_gpu:
            return self._gpu_rotation_invariant_correlation(gray1, gray2)
        else:
            return self._cpu_rotation_invariant_correlation(gray1, gray2)

    def _gpu_rotation_invariant_correlation(self, img1, img2):
        """GPU-accelerated rotation-invariant correlation using CuPy."""
        try:
            # Transfer to GPU
            gpu_img1 = cp.asarray(img1)
            gpu_img2 = cp.asarray(img2)
            
            # Compute FFT magnitude spectra
            f1 = cp.fft.fft2(gpu_img1)
            f2 = cp.fft.fft2(gpu_img2)
            mag1 = cp.abs(cp.fft.fftshift(f1))
            mag2 = cp.abs(cp.fft.fftshift(f2))
            
            # Convert to polar coordinates for rotation invariance
            center = (img1.shape[1] // 2, img1.shape[0] // 2)
            max_radius = min(center)
            
            # Create coordinate grids
            y, x = cp.ogrid[:img1.shape[0], :img1.shape[1]]
            y = y - center[1]
            x = x - center[0]
            
            # Convert to polar
            r = cp.sqrt(x*x + y*y)
            theta = cp.arctan2(y, x)
            
            # Resample in polar coordinates
            r_max = max_radius
            theta_max = 2 * cp.pi
            
            # Simple polar transform (can be optimized)
            polar1 = self._simple_polar_transform_gpu(mag1, center, max_radius)
            polar2 = self._simple_polar_transform_gpu(mag2, center, max_radius)
            
            # Phase correlation
            f_polar1 = cp.fft.fft2(polar1)
            f_polar2 = cp.fft.fft2(polar2)
            
            cross_power_spectrum = f_polar1 * cp.conj(f_polar2)
            normalized_cps = cross_power_spectrum / cp.abs(cross_power_spectrum)
            correlation = cp.abs(cp.fft.ifft2(normalized_cps))
            
            # Find maximum correlation
            max_corr = float(cp.max(correlation))
            
            return max_corr
            
        except Exception as e:
            print(f"GPU correlation failed, falling back to CPU: {e}")
            return self._cpu_rotation_invariant_correlation(img1, img2)

    def _simple_polar_transform_gpu(self, img, center, max_radius):
        """Simple polar transform on GPU."""
        h, w = img.shape
        cx, cy = center
        
        # Create output image
        polar_h, polar_w = h // 2, w // 2
        polar_img = cp.zeros((polar_h, polar_w), dtype=img.dtype)
        
        for r_idx in range(polar_h):
            for theta_idx in range(polar_w):
                r = (r_idx / polar_h) * max_radius
                theta = (theta_idx / polar_w) * 2 * cp.pi
                
                x = int(cx + r * cp.cos(theta))
                y = int(cy + r * cp.sin(theta))
                
                if 0 <= x < w and 0 <= y < h:
                    polar_img[r_idx, theta_idx] = img[y, x]
        
        return polar_img

    def _cpu_rotation_invariant_correlation(self, img1, img2):
        """CPU-based rotation-invariant correlation."""
        max_correlation = 0
        
        # Test multiple rotations
        angles = np.arange(0, 360, 15)  # Test every 15 degrees
        
        for angle in angles:
            # Rotate img2
            center = (img2.shape[1] // 2, img2.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
            
            # Calculate normalized cross-correlation
            result = cv2.matchTemplate(img1, rotated_img2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            max_correlation = max(max_correlation, max_val)
        
        return max_correlation

class InteractiveImageViewer(QLabel):
    """Interactive image viewer for manual square selection."""
    
    # Signal emitted when a square is selected (x, y, width, height)
    square_selected = pyqtSignal(int, int, int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 2px solid #333; background-color: #f0f0f0;")
        
        self._original_pixmap = None
        self._display_pixmap = None
        self._zoom_factor = 1.0
        self._selection_start = None
        self._selection_end = None
        self._selecting = False
        self._image_offset = QPoint(0, 0)
        
    def set_image(self, image_path):
        """Load and display an image for selection."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self._original_pixmap = pixmap
            self._fit_to_widget()
            self.update_display()
            return True
        return False
    
    def _fit_to_widget(self):
        """Fit image to widget size while maintaining aspect ratio."""
        if not self._original_pixmap:
            return
            
        widget_size = self.size()
        pixmap_size = self._original_pixmap.size()
        
        # Calculate zoom to fit
        zoom_w = widget_size.width() / pixmap_size.width()
        zoom_h = widget_size.height() / pixmap_size.height()
        self._zoom_factor = min(zoom_w, zoom_h) * 0.9  # Leave some margin
        
    def update_display(self):
        """Update the displayed image."""
        if not self._original_pixmap:
            self.setText("Click 'Select Reference Image' to load an image for square selection")
            return
            
        # Scale the pixmap
        scaled_size = self._original_pixmap.size() * self._zoom_factor
        self._display_pixmap = self._original_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Create a pixmap with selection overlay
        display_pixmap = self._display_pixmap.copy()
        
        if self._selection_start and self._selection_end:
            from PyQt6.QtGui import QPainter, QPen
            painter = QPainter(display_pixmap)
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            
            # Draw selection rectangle
            rect = self._get_selection_rect()
            painter.drawRect(rect)
            painter.end()
        
        self.setPixmap(display_pixmap)
    
    def _get_selection_rect(self):
        """Get the current selection rectangle in display coordinates."""
        if not (self._selection_start and self._selection_end):
            return None
            
        from PyQt6.QtCore import QRect
        x1, y1 = self._selection_start.x(), self._selection_start.y()
        x2, y2 = self._selection_end.x(), self._selection_end.y()
        
        return QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    
    def mousePressEvent(self, event):
        """Start square selection."""
        if event.button() == Qt.MouseButton.LeftButton and self._display_pixmap:
            # Convert to image coordinates
            pos = self._widget_to_image_coords(event.pos())
            if pos:
                self._selection_start = pos
                self._selection_end = pos
                self._selecting = True
                self.update_display()
    
    def mouseMoveEvent(self, event):
        """Update selection during drag."""
        if self._selecting and self._display_pixmap:
            pos = self._widget_to_image_coords(event.pos())
            if pos:
                self._selection_end = pos
                self.update_display()
    
    def mouseReleaseEvent(self, event):
        """Finish square selection."""
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._selecting = False
            
            if self._selection_start and self._selection_end:
                # Convert to original image coordinates
                rect = self._get_selection_rect()
                if rect and rect.width() > 10 and rect.height() > 10:
                    # Convert display coordinates to original image coordinates
                    scale_factor = 1.0 / self._zoom_factor
                    orig_x = int(rect.x() * scale_factor)
                    orig_y = int(rect.y() * scale_factor)
                    orig_w = int(rect.width() * scale_factor)
                    orig_h = int(rect.height() * scale_factor)
                    
                    # Emit the selection
                    self.square_selected.emit(orig_x, orig_y, orig_w, orig_h)
    
    def _widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates."""
        if not self._display_pixmap:
            return None
            
        # Get the position of the pixmap within the label
        pixmap_rect = self._display_pixmap.rect()
        label_rect = self.rect()
        
        # Center the pixmap in the label
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        # Convert to pixmap coordinates
        pixmap_x = widget_pos.x() - x_offset
        pixmap_y = widget_pos.y() - y_offset
        
        # Check if within pixmap bounds
        if (0 <= pixmap_x <= pixmap_rect.width() and 
            0 <= pixmap_y <= pixmap_rect.height()):
            return QPoint(pixmap_x, pixmap_y)
        
        return None
    
    def wheelEvent(self, event):
        """Zoom functionality."""
        if self._original_pixmap:
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
            self._zoom_factor *= zoom_factor
            self._zoom_factor = max(0.1, min(5.0, self._zoom_factor))
            self.update_display()
    
    def reset_selection(self):
        """Clear current selection."""
        self._selection_start = None
        self._selection_end = None
        self.update_display()

class PhotoViewer(QLabel):
    """Enhanced image viewer with zoom and pan capabilities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        
        self._pixmap = None
        self._zoom_factor = 1.0
        self._pan_start_pos = None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(
                int(self._pixmap.width() * self._zoom_factor),
                int(self._pixmap.height() * self._zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.adjustSize()

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self._zoom_factor *= zoom_factor
        self._zoom_factor = max(0.1, min(5.0, self._zoom_factor))  # Limit zoom range
        self.update_display()

    def reset_view(self):
        self._zoom_factor = 1.0
        self.update_display()

class CorrelationHeatmapViewer(QWidget):
    """Interactive correlation heatmap viewer using Plotly."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.webview = QWebEngineView()
        self.layout.addWidget(self.webview)

    def display_heatmap(self, correlation_matrix, subsquares):
        if correlation_matrix is None or correlation_matrix.size == 0:
            self.webview.setHtml("<h1>No correlation data to display.</h1>")
            return

        # Create labels for axes
        labels = [ss.unique_id for ss in subsquares]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False
        ))

        fig.update_layout(
            title="Cross-Correlation Matrix",
            xaxis_title="Subsquares",
            yaxis_title="Subsquares",
            width=800,
            height=600
        )

        self.webview.setHtml(fig.to_html(include_plotlyjs='cdn'))

class TopPairsGallery(QListWidget):
    """Gallery showing top correlated pairs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(150, 75))
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.itemClicked.connect(self._on_item_clicked)

    def display_top_pairs(self, correlation_matrix, subsquares, num_pairs=20):
        self.clear()
        
        if correlation_matrix is None or len(subsquares) == 0:
            return

        # Find top pairs
        n = correlation_matrix.shape[0]
        pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, correlation_matrix[i, j]))
        
        # Sort by correlation score
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Display top pairs
        for i, j, score in pairs[:num_pairs]:
            subsquare1 = subsquares[i]
            subsquare2 = subsquares[j]
            
            # Create side-by-side image
            img1 = subsquare1.processed_img
            img2 = subsquare2.processed_img
            
            # Resize for display
            display_size = 100
            img1_resized = cv2.resize(img1, (display_size, display_size))
            img2_resized = cv2.resize(img2, (display_size, display_size))
            
            # Combine images
            combined = np.hstack((img1_resized, img2_resized))
            
            # Convert to QPixmap
            h, w, ch = combined.shape
            bytes_per_line = ch * w
            qt_image = QImage(combined.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Create list item
            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap))
            item.setText(f"Score: {score:.3f}\n{subsquare1.unique_id}\nvs\n{subsquare2.unique_id}")
            item.setData(Qt.ItemDataRole.UserRole, (i, j))
            self.addItem(item)

    def _on_item_clicked(self, item):
        """Handle item click to show detailed view."""
        # This can be implemented to show a detailed comparison window
        pass

class ImageAnalysisApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Class Average Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        self.image_paths = []
        self.all_subsquares = []
        self.correlation_matrix = None
        self.processor_thread = None
        self.reference_square = None  # (image_path, x, y, w, h)
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout(control_group)
        
        # Load images button
        self.load_btn = QPushButton("Load 2D Class Images")
        self.load_btn.clicked.connect(self._load_images)
        control_layout.addWidget(self.load_btn, 0, 0)
        
        # Reference selection
        self.select_ref_btn = QPushButton("Select Reference Square")
        self.select_ref_btn.clicked.connect(self._select_reference)
        self.select_ref_btn.setEnabled(False)
        control_layout.addWidget(self.select_ref_btn, 0, 1)
        
        self.clear_ref_btn = QPushButton("Clear Reference")
        self.clear_ref_btn.clicked.connect(self._clear_reference)
        self.clear_ref_btn.setEnabled(False)
        control_layout.addWidget(self.clear_ref_btn, 0, 2)
        
        # Detection mode selection
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self.auto_mode_radio = QRadioButton("Automatic Detection")
        self.ref_mode_radio = QRadioButton("Reference-Based Detection")
        self.auto_mode_radio.setChecked(True)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto_mode_radio, 0)
        self.mode_group.addButton(self.ref_mode_radio, 1)
        
        mode_layout.addWidget(self.auto_mode_radio)
        mode_layout.addWidget(self.ref_mode_radio)
        control_layout.addWidget(mode_group, 1, 0, 1, 3)
        
        # Parameters (only shown in automatic mode)
        self.params_group = QGroupBox("Automatic Detection Parameters")
        params_layout = QHBoxLayout(self.params_group)
        
        params_layout.addWidget(QLabel("Min Square Size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(20, 200)
        self.min_size_spin.setValue(30)
        params_layout.addWidget(self.min_size_spin)
        
        params_layout.addWidget(QLabel("Target Size:"))
        self.target_size_spin = QSpinBox()
        self.target_size_spin.setRange(64, 512)
        self.target_size_spin.setValue(128)
        params_layout.addWidget(self.target_size_spin)
        
        control_layout.addWidget(self.params_group, 2, 0, 1, 3)
        
        # Reference info
        self.ref_info_label = QLabel("No reference square selected")
        self.ref_info_label.setStyleSheet("color: #666; font-style: italic;")
        control_layout.addWidget(self.ref_info_label, 3, 0, 1, 3)
        
        # Process button
        self.process_btn = QPushButton("Detect & Analyze")
        self.process_btn.clicked.connect(self._start_processing)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn, 4, 0)
        
        # View buttons
        self.heatmap_btn = QPushButton("Show Correlation Heatmap")
        self.heatmap_btn.clicked.connect(self._show_heatmap)
        self.heatmap_btn.setEnabled(False)
        control_layout.addWidget(self.heatmap_btn, 4, 1)
        
        self.gallery_btn = QPushButton("Show Top Pairs")
        self.gallery_btn.clicked.connect(self._show_gallery)
        self.gallery_btn.setEnabled(False)
        control_layout.addWidget(self.gallery_btn, 4, 2)
        
        main_layout.addWidget(control_group)
        
        # Connect radio button signals
        self.mode_group.buttonToggled.connect(self._on_mode_changed)
        
        # Display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_widget = QWidget()
        self.display_layout = QVBoxLayout(self.display_widget)
        self.scroll_area.setWidget(self.display_widget)
        main_layout.addWidget(self.scroll_area)
        
        # Status bar
        self.status_label = QLabel("Ready to load 2D class average images...")
        main_layout.addWidget(self.status_label)
        
        # Progress dialog
        self.progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.hide()

    def _on_mode_changed(self):
        """Handle detection mode change."""
        auto_mode = self.auto_mode_radio.isChecked()
        self.params_group.setVisible(auto_mode)
        self.select_ref_btn.setEnabled(not auto_mode and len(self.image_paths) > 0)
        self.clear_ref_btn.setEnabled(not auto_mode and self.reference_square is not None)
        
        # Update process button state
        self._update_process_button_state()

    def _update_process_button_state(self):
        """Update the state of the process button based on current settings."""
        if not self.image_paths:
            self.process_btn.setEnabled(False)
            return
            
        if self.auto_mode_radio.isChecked():
            # Automatic mode - just need images
            self.process_btn.setEnabled(True)
        else:
            # Reference mode - need reference square
            self.process_btn.setEnabled(self.reference_square is not None)

    def _load_images(self):
        """Load 2D class average images."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.mrc)")
        
        if file_dialog.exec():
            self.image_paths = file_dialog.selectedFiles()
            if self.image_paths:
                self.status_label.setText(f"Loaded {len(self.image_paths)} images.")
                self.select_ref_btn.setEnabled(not self.auto_mode_radio.isChecked())
                self._clear_display()
                self._clear_reference()
                self._update_process_button_state()
            else:
                self.status_label.setText("No images selected.")

    def _select_reference(self):
        """Open reference selection dialog."""
        if not self.image_paths:
            self.status_label.setText("Please load images first.")
            return
            
        # Create reference selection dialog
        ref_dialog = QMainWindow(self)
        ref_dialog.setWindowTitle("Select Reference Square")
        ref_dialog.setGeometry(150, 150, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        ref_dialog.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Instructions
        instructions = QLabel(
            "<h3>Select Reference Square</h3>"
            "<p>1. Choose an image from the dropdown</p>"
            "<p>2. Click and drag to select a square region</p>"
            "<p>3. Click 'Use This Selection' to confirm</p>"
            "<p>Use mouse wheel to zoom in/out</p>"
        )
        layout.addWidget(instructions)
        
        # Image selection dropdown
        from PyQt6.QtWidgets import QComboBox
        image_combo = QComboBox()
        for img_path in self.image_paths:
            image_combo.addItem(os.path.basename(img_path), img_path)
        layout.addWidget(image_combo)
        
        # Interactive viewer
        self.ref_viewer = InteractiveImageViewer()
        layout.addWidget(self.ref_viewer)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        use_btn = QPushButton("Use This Selection")
        use_btn.setEnabled(False)
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(use_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Variables to store selection
        current_selection = None
        current_image_path = None
        
        def on_image_changed():
            nonlocal current_image_path
            current_image_path = image_combo.currentData()
            if current_image_path:
                if self.ref_viewer.set_image(current_image_path):
                    self.ref_viewer.reset_selection()
                    use_btn.setEnabled(False)
        
        def on_square_selected(x, y, w, h):
            nonlocal current_selection
            current_selection = (x, y, w, h)
            use_btn.setEnabled(True)
            self.status_label.setText(f"Selected square: {w}x{h} at ({x}, {y})")
        
        def on_use_selection():
            nonlocal current_selection, current_image_path
            if current_selection and current_image_path:
                x, y, w, h = current_selection
                self.reference_square = (current_image_path, x, y, w, h)
                self._update_reference_info()
                ref_dialog.close()
                self._update_process_button_state()
        
        # Connect signals
        image_combo.currentTextChanged.connect(on_image_changed)
        self.ref_viewer.square_selected.connect(on_square_selected)
        use_btn.clicked.connect(on_use_selection)
        cancel_btn.clicked.connect(ref_dialog.close)
        
        # Load first image
        on_image_changed()
        
        # Show dialog
        ref_dialog.show()

    def _clear_reference(self):
        """Clear the current reference square selection."""
        self.reference_square = None
        self._update_reference_info()
        self._update_process_button_state()

    def _update_reference_info(self):
        """Update the reference square information display."""
        if self.reference_square:
            img_path, x, y, w, h = self.reference_square
            filename = os.path.basename(img_path)
            self.ref_info_label.setText(
                f"Reference: {w}×{h} square at ({x},{y}) in {filename}"
            )
            self.ref_info_label.setStyleSheet("color: #0a7d0a; font-weight: bold;")
            self.clear_ref_btn.setEnabled(True)
        else:
            self.ref_info_label.setText("No reference square selected")
            self.ref_info_label.setStyleSheet("color: #666; font-style: italic;")
            self.clear_ref_btn.setEnabled(False) # Corrected line
        
        # Parameters
        control_layout.addWidget(QLabel("Min Square Size:"), 0, 1)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(20, 200)
        self.min_size_spin.setValue(30)
        control_layout.addWidget(self.min_size_spin, 0, 2)
        
        control_layout.addWidget(QLabel("Target Size:"), 0, 3)
        self.target_size_spin = QSpinBox()
        self.target_size_spin.setRange(64, 512)
        self.target_size_spin.setValue(128)
        control_layout.addWidget(self.target_size_spin, 0, 4)
        
        # Process button
        self.process_btn = QPushButton("Detect & Analyze")
        self.process_btn.clicked.connect(self._start_processing)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn, 1, 0)
        
        # View buttons
        self.heatmap_btn = QPushButton("Show Correlation Heatmap")
        self.heatmap_btn.clicked.connect(self._show_heatmap)
        self.heatmap_btn.setEnabled(False)
        control_layout.addWidget(self.heatmap_btn, 1, 1)
        
        self.gallery_btn = QPushButton("Show Top Pairs")
        self.gallery_btn.clicked.connect(self._show_gallery)
        self.gallery_btn.setEnabled(False)
        control_layout.addWidget(self.gallery_btn, 1, 2)
        
        main_layout.addWidget(control_group)
        
        # Display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_widget = QWidget()
        self.display_layout = QVBoxLayout(self.display_widget)
        self.scroll_area.setWidget(self.display_widget)
        main_layout.addWidget(self.scroll_area)
        
        # Status bar
        self.status_label = QLabel("Ready to load 2D class average images...")
        main_layout.addWidget(self.status_label)
        
        # Progress dialog
        self.progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.hide()

    def _load_images(self):
        """Load 2D class average images."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.mrc)")
        
        if file_dialog.exec():
            self.image_paths = file_dialog.selectedFiles()
            if self.image_paths:
                self.status_label.setText(f"Loaded {len(self.image_paths)} images. Click 'Detect & Analyze' to start.")
                self.process_btn.setEnabled(True)
                self._clear_display()
            else:
                self.status_label.setText("No images selected.")

    def _clear_display(self):
        """Clear the display area."""
        while self.display_layout.count():
            item = self.display_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _start_processing(self):
        """Start the image processing pipeline."""
        if not self.image_paths:
            self.status_label.setText("Please load images first.")
            return

        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.heatmap_btn.setEnabled(False)
        self.gallery_btn.setEnabled(False)
        
        # Clear previous results
        self._clear_display()
        self.all_subsquares = []
        self.correlation_matrix = None
        
        # Setup progress dialog
        self.progress_dialog.setValue(0)
        self.progress_dialog.setLabelText("Starting processing...")
        self.progress_dialog.show()
        
        # Get parameters
        min_size = self.min_size_spin.value()
        target_size = self.target_size_spin.value()
        
        # Start processing thread
        self.processor_thread = ImageProcessor(
            self.image_paths,
            target_subsquare_size=(target_size, target_size),
            min_square_size=min_size
        )
        
        # Connect signals
        self.processor_thread.detection_progress.connect(self._update_progress)
        self.processor_thread.processing_progress.connect(self._update_progress)
        self.processor_thread.correlation_progress.connect(self._update_progress)
        self.processor_thread.finished_processing.connect(self._on_processing_finished)
        
        self.processor_thread.start()

    def _update_progress(self, current, total, message):
        """Update progress dialog."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_dialog.setValue(progress)
        self.progress_dialog.setLabelText(message)
        self.status_label.setText(message)

    def _on_processing_finished(self, subsquares, correlation_matrix):
        """Handle completion of processing."""
        self.all_subsquares = subsquares
        self.correlation_matrix = correlation_matrix
        
        # Update UI
        self.progress_dialog.hide()
        self.process_btn.setEnabled(True)
        
        if len(subsquares) > 0:
            self.heatmap_btn.setEnabled(True)
            self.gallery_btn.setEnabled(True)
            self.status_label.setText(f"Processing complete! Found {len(subsquares)} squares.")
            self._display_detected_squares()
        else:
            self.status_label.setText("No squares detected. Try adjusting parameters.")

    def _display_detected_squares(self):
        """Display detected squares grouped by source image."""
        self._clear_display()
        
        if not self.all_subsquares:
            self.display_layout.addWidget(QLabel("No squares detected."))
            return

        # Group by source image
        image_groups = {}
        for subsquare in self.all_subsquares:
            img_path = subsquare.original_image_path
            if img_path not in image_groups:
                image_groups[img_path] = []
            image_groups[img_path].append(subsquare)

        # Display each group
        for img_path, subsquares in image_groups.items():
            # Image header
            header = QLabel(f"<h3>Detected squares from: {os.path.basename(img_path)} ({len(subsquares)} found)</h3>")
            self.display_layout.addWidget(header)
            
            # Create horizontal layout for squares
            squares_widget = QWidget()
            squares_layout = QHBoxLayout(squares_widget)
            
            for subsquare in subsquares[:10]:  # Limit display to first 10
                viewer = PhotoViewer()
                viewer.set_pixmap(subsquare.to_qpixmap(size=120))
                viewer.setFixedSize(140, 140)
                viewer.setToolTip(f"ID: {subsquare.unique_id}\nBBox: {subsquare.bbox}")
                viewer.setStyleSheet("border: 1px solid gray; margin: 2px;")
                squares_layout.addWidget(viewer)
            
            squares_layout.addStretch()
            self.display_layout.addWidget(squares_widget)
        
        self.display_layout.addStretch()

    def _show_heatmap(self):
        """Show correlation heatmap in new window."""
        if self.correlation_matrix is None:
            self.status_label.setText("No correlation data available.")
            return

        self.heatmap_window = QMainWindow(self)
        self.heatmap_window.setWindowTitle("Correlation Heatmap")
        self.heatmap_window.setGeometry(200, 200, 900, 700)
        
        heatmap_viewer = CorrelationHeatmapViewer()
        heatmap_viewer.display_heatmap(self.correlation_matrix, self.all_subsquares)
        self.heatmap_window.setCentralWidget(heatmap_viewer)
        self.heatmap_window.show()

    def _show_gallery(self):
        """Show top correlated pairs gallery."""
        if self.correlation_matrix is None:
            self.status_label.setText("No correlation data available.")
            return

        self.gallery_window = QMainWindow(self)
        self.gallery_window.setWindowTitle("Top Correlated Pairs")
        self.gallery_window.setGeometry(250, 250, 1000, 600)
        
        gallery_viewer = TopPairsGallery()
        gallery_viewer.display_top_pairs(self.correlation_matrix, self.all_subsquares)
        self.gallery_window.setCentralWidget(gallery_viewer)
        self.gallery_window.show()

    def closeEvent(self, event):
        """Handle application closing."""
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.terminate()
            self.processor_thread.wait()
        event.accept()


def main():
    """Main application entry point."""
    # Check for required dependencies
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        print("Error: PyQt6-WebEngine is required but not installed.")
        print("Install it with: pip install PyQt6-WebEngine")
        sys.exit(1)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("2D Class Average Analysis Tool")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = ImageAnalysisApp()
    window.show()
    
    # Print startup information
    print("2D Class Average Analysis Tool Started")
    print(f"GPU Acceleration: {'Available' if GPU_AVAILABLE else 'Not Available'}")
    if not GPU_AVAILABLE:
        print("For GPU acceleration, install CuPy: pip install cupy-cuda12x")
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
