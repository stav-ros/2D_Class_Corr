# 2D_Class_Corr
A simple gui that loads 2D class images and runs correlation score between them


Required Dependencies:

bash pip install numpy matplotlib seaborn pillow

Features:

✅ Load & Split Images: Automatically splits 3 images into 300 squares

✅ NCC Calculation: Full 300x300 correlation matrix

✅ Interactive Visualization: Correlation matrix heatmap and top pairs display

✅ Threshold Control: Filter correlations by minimum strength

✅ Cross-Image Detection: Identifies features appearing across different source images

✅ Progress Tracking: Real-time progress updates during analysis

✅ Results Summary: Statistical analysis with cross-image match identification


Usage:

Save the code as feature_correlator.py
Install dependencies: pip install numpy matplotlib seaborn pillow
Run: python feature_correlator.py
Load your 3 grid images
Click "Run Cross-Correlation Analysis"
Explore results in the two tabs

The application will identify common features across your images, helping you find repeated structures, similar objects, or correlated patterns. The correlation matrix clearly shows relationships between all 300 squares, with visual separation between the three source images.

🆕 New Interactive Features:
🔍 Clickable Correlation Matrix

Click any square in the 300x300 correlation matrix to open a detailed inspector
Visual feedback shows which squares correlate with which
Grid boundaries clearly separate the 3 source images
Zoom/Pan tools for detailed matrix exploration

📊 Square Inspector Window

Side-by-side comparison of any two squares
Detailed information including:

Source image (1, 2, or 3)
Grid position (row/column)
Exact correlation value
Cross-image match detection


Zoomable images with navigation toolbar

🎯 Enhanced Top Correlations

Click any image pair to open detailed inspector
Better labeling showing source image and grid position
More pairs displayed (up to 100)

🔧 Improved Interface

Navigation toolbars on both plots for zooming/panning
Better labeling with square indices and positions
Help section explaining all interactions
Enhanced results with cross-image analysis

🎮 How to Use:

Load Images: Select your 3 grid images
Run Analysis: Wait for correlation computation
Explore Results:

Matrix Tab: Click anywhere to inspect square pairs


RetryThis response paused because Claude reached its max length for a message. Hit continue to nudge Claude along.ContinueClaude can make mistakes. Please double-check responses.
