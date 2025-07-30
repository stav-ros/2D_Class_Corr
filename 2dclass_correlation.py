import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import threading
import os
from queue import Queue

class CorrelationAnalyzer:
    """Main correlation analysis class"""
    
    def __init__(self):
        self.images = []
        self.correlation_matrix = None
        self.top_pairs = []
        self.image_paths = []
        self.square_info = []  # Store info about each square (source image, position)
        
    def normalized_cross_correlation(self, img1, img2):
        """Calculate NCC between two images"""
        # Convert to float and flatten
        img1_flat = img1.astype(np.float64).flatten()
        img2_flat = img2.astype(np.float64).flatten()
        
        # Calculate means
        mean1 = np.mean(img1_flat)
        mean2 = np.mean(img2_flat)
        
        # Calculate standard deviations
        std1 = np.std(img1_flat)
        std2 = np.std(img2_flat)
        
        # Avoid division by zero
        if std1 == 0 or std2 == 0:
            return 0.0
            
        # Calculate NCC
        numerator = np.mean((img1_flat - mean1) * (img2_flat - mean2))
        ncc = numerator / (std1 * std2)
        
        return ncc
    
    def load_and_split_images(self, file_paths, progress_callback=None):
        """Load and split images into squares"""
        if len(file_paths) != 3:
            raise ValueError("Please select exactly 3 images")
        
        self.images = []
        self.image_paths = file_paths
        self.square_info = []
        
        for img_idx, img_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(f"Loading image {img_idx + 1}/3...")
            
            # Load image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img)
            
            # Split into 10x10 grid (100 squares per image)
            h, w = img_array.shape
            square_h, square_w = h // 10, w // 10
            
            for i in range(10):
                for j in range(10):
                    y1, y2 = i * square_h, (i + 1) * square_h
                    x1, x2 = j * square_w, (j + 1) * square_w
                    square = img_array[y1:y2, x1:x2]
                    self.images.append(square)
                    
                    # Store square information
                    square_idx = img_idx * 100 + i * 10 + j
                    self.square_info.append({
                        'index': square_idx,
                        'source_image': img_idx,
                        'source_file': os.path.basename(img_path),
                        'grid_row': i,
                        'grid_col': j,
                        'global_row': i,
                        'global_col': j
                    })
        
        return len(self.images)
    
    def compute_correlation_matrix(self, progress_callback=None):
        """Compute full correlation matrix"""
        n_images = len(self.images)
        self.correlation_matrix = np.zeros((n_images, n_images))
        
        # Calculate upper triangle (matrix is symmetric)
        total_pairs = (n_images * (n_images - 1)) // 2
        completed = 0
        
        for i in range(n_images):
            for j in range(i, n_images):
                if i == j:
                    self.correlation_matrix[i, j] = 1.0  # Perfect self-correlation
                else:
                    ncc = self.normalized_cross_correlation(self.images[i], self.images[j])
                    self.correlation_matrix[i, j] = ncc
                    self.correlation_matrix[j, i] = ncc  # Symmetric
                    completed += 1
                    
                # Update progress
                if j > i and progress_callback:  # Only count off-diagonal elements
                    progress = int((completed / total_pairs) * 100)
                    progress_callback(f"Computing correlations... {progress}%")
        
        # Find top correlations
        self.top_pairs = []
        for i in range(n_images):
            for j in range(i + 1, n_images):
                correlation = self.correlation_matrix[i, j]
                self.top_pairs.append((i, j, correlation))
        
        # Sort by correlation value (descending)
        self.top_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return self.correlation_matrix, self.top_pairs

class SquareInspector:
    """Pop-up window for detailed square inspection"""
    
    def __init__(self, parent, analyzer, square1_idx, square2_idx, correlation):
        self.parent = parent
        self.analyzer = analyzer
        self.square1_idx = square1_idx
        self.square2_idx = square2_idx
        self.correlation = correlation
        
        self.create_window()
        
    def create_window(self):
        """Create the inspector window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Square Inspector - Correlation: {self.correlation:.4f}")
        self.window.geometry('800x600')
        
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Square Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create info text
        info1 = self.analyzer.square_info[self.square1_idx]
        info2 = self.analyzer.square_info[self.square2_idx]
        
        info_text = f"""Square 1 (Index {self.square1_idx}):
Source: Image {info1['source_image'] + 1} ({info1['source_file']})
Grid Position: Row {info1['grid_row']}, Column {info1['grid_col']}

Square 2 (Index {self.square2_idx}):
Source: Image {info2['source_image'] + 1} ({info2['source_file']})
Grid Position: Row {info2['grid_row']}, Column {info2['grid_col']}

Normalized Cross-Correlation: {self.correlation:.6f}
Cross-Image Match: {'Yes' if info1['source_image'] != info2['source_image'] else 'No'}"""
        
        info_label = ttk.Label(info_frame, text=info_text, font=('Courier', 10))
        info_label.pack(anchor=tk.W)
        
        # Image comparison frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for side-by-side comparison
        self.fig = Figure(figsize=(12, 6))
        canvas = FigureCanvasTkAgg(self.fig, image_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar for zooming
        toolbar = NavigationToolbar2Tk(canvas, image_frame)
        toolbar.update()
        
        self.plot_comparison()
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)
        
    def plot_comparison(self):
        """Plot side-by-side comparison of the two squares"""
        self.fig.clear()
        
        # Square 1
        ax1 = self.fig.add_subplot(121)
        ax1.imshow(self.analyzer.images[self.square1_idx], cmap='gray', interpolation='nearest')
        info1 = self.analyzer.square_info[self.square1_idx]
        ax1.set_title(f'Square {self.square1_idx}\nImage {info1["source_image"] + 1}, '
                     f'Row {info1["grid_row"]}, Col {info1["grid_col"]}')
        ax1.axis('off')
        
        # Square 2
        ax2 = self.fig.add_subplot(122)
        ax2.imshow(self.analyzer.images[self.square2_idx], cmap='gray', interpolation='nearest')
        info2 = self.analyzer.square_info[self.square2_idx]
        ax2.set_title(f'Square {self.square2_idx}\nImage {info2["source_image"] + 1}, '
                     f'Row {info2["grid_row"]}, Col {info2["grid_col"]}')
        ax2.axis('off')
        
        self.fig.suptitle(f'Correlation Analysis - NCC: {self.correlation:.4f}')
        self.fig.tight_layout()

class ImageCrossCorrelationGUI:
    def __init__(self, root):
        self.root = root
        self.analyzer = CorrelationAnalyzer()
        self.current_threshold = 0.0
        self.current_n_pairs = 20
        self.selected_square = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.root.title('Enhanced Image Feature Cross-Correlation Analyzer')
        self.root.geometry('1600x1000')
        
        # Create main frames
        self.create_left_panel()
        self.create_right_panel()
        
    def create_left_panel(self):
        """Create left control panel"""
        # Left frame
        left_frame = ttk.Frame(self.root, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        # File loading section
        file_frame = ttk.LabelFrame(left_frame, text="Load Images", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.load_btn = ttk.Button(file_frame, text="Load 3 Grid Images (10x10 each)", 
                                  command=self.load_images)
        self.load_btn.pack(fill=tk.X)
        
        self.status_label = ttk.Label(file_frame, text="No images loaded")
        self.status_label.pack(pady=(5, 0))
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(analysis_frame, text="Run Cross-Correlation Analysis",
                                     command=self.run_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(analysis_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=(5, 0))
        
        self.progress_bar = ttk.Progressbar(analysis_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Visualization controls
        viz_frame = ttk.LabelFrame(left_frame, text="Visualization Controls", padding=10)
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Threshold control
        threshold_frame = ttk.Frame(viz_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(threshold_frame, text="Correlation Threshold:").pack(anchor=tk.W)
        
        threshold_control_frame = ttk.Frame(threshold_frame)
        threshold_control_frame.pack(fill=tk.X)
        
        self.threshold_var = tk.DoubleVar(value=0.0)
        self.threshold_scale = ttk.Scale(threshold_control_frame, from_=0.0, to=1.0, 
                                        variable=self.threshold_var, orient=tk.HORIZONTAL,
                                        command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.threshold_label = ttk.Label(threshold_control_frame, text="0.00")
        self.threshold_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Top pairs control
        pairs_frame = ttk.Frame(viz_frame)
        pairs_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(pairs_frame, text="Top pairs to show:").pack(anchor=tk.W)
        
        pairs_control_frame = ttk.Frame(pairs_frame)
        pairs_control_frame.pack(fill=tk.X)
        
        self.pairs_var = tk.IntVar(value=20)
        self.pairs_spinbox = ttk.Spinbox(pairs_control_frame, from_=5, to=100, 
                                        textvariable=self.pairs_var, width=10,
                                        command=self.update_top_pairs)
        self.pairs_spinbox.pack(side=tk.LEFT)
        
        # Update buttons
        update_btn = ttk.Button(viz_frame, text="Update Plots", command=self.update_plots)
        update_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Interaction help
        help_frame = ttk.LabelFrame(left_frame, text="Interaction Help", padding=10)
        help_frame.pack(fill=tk.X, pady=(0, 10))
        
        help_text = """• Click on correlation matrix to inspect squares
• Use mouse wheel to zoom in plots
• Click on top correlation pairs to view details
• Use navigation toolbar for pan/zoom"""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(anchor=tk.W)
        
        # Results section
        results_frame = ttk.LabelFrame(left_frame, text="Results Summary", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, height=15, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_right_panel(self):
        """Create right panel with plots"""
        # Right frame with notebook for tabs
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Correlation matrix tab
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="Correlation Matrix (Click to Inspect)")
        
        self.matrix_fig = Figure(figsize=(12, 10))
        self.matrix_canvas = FigureCanvasTkAgg(self.matrix_fig, matrix_frame)
        self.matrix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        matrix_toolbar = NavigationToolbar2Tk(self.matrix_canvas, matrix_frame)
        matrix_toolbar.update()
        
        # Bind click event
        self.matrix_canvas.mpl_connect('button_press_event', self.on_matrix_click)
        
        # Top pairs tab
        pairs_frame = ttk.Frame(self.notebook)
        self.notebook.add(pairs_frame, text="Top Correlations (Click to Inspect)")
        
        self.pairs_fig = Figure(figsize=(12, 10))
        self.pairs_canvas = FigureCanvasTkAgg(self.pairs_fig, pairs_frame)
        self.pairs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        pairs_toolbar = NavigationToolbar2Tk(self.pairs_canvas, pairs_frame)
        pairs_toolbar.update()
        
        # Bind click event
        self.pairs_canvas.mpl_connect('button_press_event', self.on_pairs_click)
        
    def on_matrix_click(self, event):
        """Handle clicks on correlation matrix"""
        if event.inaxes is None or self.analyzer.correlation_matrix is None:
            return
        
        # Get clicked coordinates
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        # Check bounds
        if 0 <= x < 300 and 0 <= y < 300:
            correlation = self.analyzer.correlation_matrix[y, x]
            if abs(correlation) >= self.current_threshold and x != y:
                # Open inspector window
                SquareInspector(self.root, self.analyzer, y, x, correlation)
    
    def on_pairs_click(self, event):
        """Handle clicks on top pairs plot"""
        if event.inaxes is None or not self.analyzer.top_pairs:
            return
        
        # Find which subplot was clicked
        for ax in self.pairs_fig.get_axes():
            if ax == event.inaxes:
                # Get subplot index
                subplot_idx = self.pairs_fig.get_axes().index(ax)
                pair_idx = subplot_idx // 2  # Each pair uses 2 subplots
                
                if pair_idx < len(self.analyzer.top_pairs):
                    i, j, correlation = self.analyzer.top_pairs[pair_idx]
                    SquareInspector(self.root, self.analyzer, i, j, correlation)
                break
    
    def load_images(self):
        """Load and split the 3 grid images"""
        file_paths = filedialog.askopenfilenames(
            title="Select 3 Grid Images (10x10)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if len(file_paths) != 3:
            messagebox.showwarning("Warning", "Please select exactly 3 images")
            return
        
        try:
            def progress_callback(message):
                self.progress_var.set(message)
                self.root.update_idletasks()
            
            n_squares = self.analyzer.load_and_split_images(file_paths, progress_callback)
            
            self.status_label.config(text=f"Loaded {n_squares} squares from 3 images")
            self.analyze_btn.config(state=tk.NORMAL)
            self.progress_var.set("Ready for analysis")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images: {str(e)}")
            self.status_label.config(text="Error loading images")
    
    def run_analysis(self):
        """Start the correlation analysis in a separate thread"""
        if not self.analyzer.images:
            return
        
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_bar.config(value=0)
        
        def analysis_thread():
            try:
                def progress_callback(message):
                    # Extract percentage if present
                    if '%' in message:
                        try:
                            percent = int(message.split()[-1].replace('%', ''))
                            self.root.after(0, lambda: self.progress_bar.config(value=percent))
                        except:
                            pass
                    self.root.after(0, lambda: self.progress_var.set(message))
                
                # Run analysis
                correlation_matrix, top_pairs = self.analyzer.compute_correlation_matrix(progress_callback)
                
                # Update GUI in main thread
                self.root.after(0, self.analysis_complete)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
        
        # Start analysis thread
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def analysis_complete(self):
        """Handle completion of correlation analysis"""
        self.progress_bar.config(value=100)
        self.progress_var.set("Analysis complete - Click on plots to inspect squares")
        
        # Update visualizations
        self.update_plots()
        
        # Update results summary
        self.update_results_summary()
        
        self.analyze_btn.config(state=tk.NORMAL)
    
    def update_plots(self):
        """Update all plots with current settings"""
        if self.analyzer.correlation_matrix is None:
            return
        
        self.current_threshold = self.threshold_var.get()
        self.current_n_pairs = self.pairs_var.get()
        
        # Update correlation matrix plot
        self.plot_correlation_matrix()
        
        # Update top pairs plot
        self.plot_top_correlations()
    
    def plot_correlation_matrix(self):
        """Plot the correlation matrix as a heatmap"""
        self.matrix_fig.clear()
        ax = self.matrix_fig.add_subplot(111)
        
        correlation_matrix = self.analyzer.correlation_matrix
        
        # Mask values below threshold
        masked_matrix = np.where(np.abs(correlation_matrix) >= self.current_threshold, 
                                correlation_matrix, np.nan)
        
        # Create heatmap
        im = ax.imshow(masked_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                      interpolation='nearest')
        
        # Add colorbar
        cbar = self.matrix_fig.colorbar(im, ax=ax, label='Normalized Cross-Correlation')
        
        ax.set_title(f'Interactive Correlation Matrix (300x300)\nThreshold: |NCC| >= {self.current_threshold:.2f}\nClick on squares to inspect')
        ax.set_xlabel('Square Index')
        ax.set_ylabel('Square Index')
        
        # Add grid lines every 100 images (for the 3 source images)
        for i in [100, 200]:
            ax.axhline(y=i-0.5, color='black', linewidth=2)
            ax.axvline(x=i-0.5, color='black', linewidth=2)
        
        # Add labels for image groups
        ax.text(50, -15, 'Image 1\n(0-99)', ha='center', fontweight='bold')
        ax.text(150, -15, 'Image 2\n(100-199)', ha='center', fontweight='bold')
        ax.text(250, -15, 'Image 3\n(200-299)', ha='center', fontweight='bold')
        
        self.matrix_fig.tight_layout()
        self.matrix_canvas.draw()
    
    def plot_top_correlations(self):
        """Plot top N most correlated pairs"""
        self.pairs_fig.clear()
        
        if not self.analyzer.top_pairs:
            return
        
        # Calculate grid size for subplots
        n_pairs = min(self.current_n_pairs, len(self.analyzer.top_pairs), 15)  # Limit display
        n_cols = 4  # 2 images per pair
        n_rows = n_pairs
        
        for idx, (i, j, correlation) in enumerate(self.analyzer.top_pairs[:n_pairs]):
            info_i = self.analyzer.square_info[i]
            info_j = self.analyzer.square_info[j]
            
            # Image i
            ax1 = self.pairs_fig.add_subplot(n_rows, n_cols, idx * 2 + 1)
            ax1.imshow(self.analyzer.images[i], cmap='gray', interpolation='nearest')
            ax1.set_title(f'#{i}: Img{info_i["source_image"]+1} R{info_i["grid_row"]}C{info_i["grid_col"]}', 
                         fontsize=8)
            ax1.axis('off')
            
            # Image j  
            ax2 = self.pairs_fig.add_subplot(n_rows, n_cols, idx * 2 + 2)
            ax2.imshow(self.analyzer.images[j], cmap='gray', interpolation='nearest')
            ax2.set_title(f'#{j}: Img{info_j["source_image"]+1} R{info_j["grid_row"]}C{info_j["grid_col"]}\nNCC: {correlation:.3f}', 
                         fontsize=8)
            ax2.axis('off')
        
        self.pairs_fig.suptitle(f'Top {n_pairs} Most Correlated Pairs (Click to Inspect)', fontsize=12)
        self.pairs_fig.tight_layout()
        self.pairs_canvas.draw()
    
    def update_threshold(self, value=None):
        """Update threshold display"""
        threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.current_threshold = threshold
    
    def update_top_pairs(self):
        """Update current n_pairs setting"""
        self.current_n_pairs = self.pairs_var.get()
    
    def update_results_summary(self):
        """Update the results summary text"""
        if not self.analyzer.top_pairs:
            return
        
        # Calculate statistics
        correlations = [pair[2] for pair in self.analyzer.top_pairs]
        max_corr = max(correlations)
        mean_corr = np.mean(correlations)
        high_corr_count = len([c for c in correlations if c > 0.8])
        
        # Count cross-image correlations (between different source images)
        cross_image_pairs = []
        for i, j, corr in self.analyzer.top_pairs:
            source_i = i // 100
            source_j = j // 100
            if source_i != source_j:
                cross_image_pairs.append((i, j, corr, source_i, source_j))
        
        # Count high cross-image correlations
        high_cross_corr = len([p for p in cross_image_pairs if p[2] > 0.5])
        
        # Generate summary
        summary = f"""ANALYSIS RESULTS:
========================
Total squares analyzed: {len(self.analyzer.images)}
Maximum correlation: {max_corr:.4f}
Average correlation: {mean_corr:.4f}
High correlations (>0.8): {high_corr_count}

CROSS-IMAGE ANALYSIS:
====================
Total cross-image pairs: {len(cross_image_pairs)}
High cross-image correlations (>0.5): {high_cross_corr}

TOP 15 CROSS-IMAGE MATCHES:
===========================
"""
        
        for idx, (i, j, corr, src_i, src_j) in enumerate(cross_image_pairs[:15]):
            info_i = self.analyzer.square_info[i]
            info_j = self.analyzer.square_info[j]
            summary += f"{idx+1:2d}. #{i:3d} (Img{src_i+1} R{info_i['grid_row']}C{info_i['grid_col']}) ↔ #{j:3d} (Img{src_j+1} R{info_j['grid_row']}C{info_j['grid_col']}): {corr:.4f}\n"
        
        summary += f"""
INTERACTION TIPS:
================
• Click correlation matrix to inspect any pair
• Click top correlation images for detailed view
• Use zoom/pan tools for detailed examination
• Adjust threshold to filter weak correlations
"""
        
        # Clear and insert new text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, summary)

def main():
    root = tk.Tk()
    app = ImageCrossCorrelationGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
