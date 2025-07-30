🔬 Cryo-Correlator v2.0: The Ultimate 2D Class Detective 🕵️
Ever stared at hundreds of fuzzy gray squares from your cryo-EM run and wondered, "Are... are you two related?" Now you can find out, with more power and precision than ever before!

Cryo-Correlator is a desktop tool with an enhanced GUI that dives into your 2D class average images and finds the hidden similarities between them. It's like a dating app for your protein views, but with more math and less awkward small talk.

(You can replace this with a screenshot of your actual GUI!)

🤔 What's the Big Idea?
In cryo-electron microscopy, we get thousands of images of individual particles, which are then averaged into "2D classes." These classes represent different views of a molecule. Sometimes, you get very similar-looking classes from different images or even within the same one.

Finding these similarities is crucial for:

Validating your structure: Do different subsets of data produce the same views?

Sorting through heterogeneity: Are there different conformations or populations of your particle?

Cleaning up your dataset: Identifying and grouping the best, most consistent views.

Doing this by eye is slow, tedious, and subjective. This tool automates and supercharges the process!

✨ Key Features
🤖 Dual-Mode Detection:

Automatic Mode: Uses OpenCV with advanced filters to automatically find every single class average square in your images, no matter the layout.

Reference-Based Mode: Select a specific particle view, and the app will hunt for matches across all your images using template matching.

🔬 Sophisticated Quality Metrics: It doesn't just look at pixels; it understands them. Each detected square is scored for texture and edge quality, giving you a quantifiable measure of how good it is.

🧠 Dynamically Weighted Correlation: The analysis intelligently weighs correlation scores. A match between two high-quality squares is considered more significant than a match between two noisy, low-quality ones.

🔄 Advanced Rotational Matching: Is your particle view slightly tilted? No problem. The script rotates each image to find the best possible alignment, using a blend of Normalized Cross-Correlation (NCC), Structural Similarity Index (SSIM), and Mutual Information for a highly robust score.

🌡️ Interactive Heatmap (Powered by Plotly): Visualize the entire NxN correlation matrix. The brighter the spot, the hotter the match! Click any pixel to instantly open a detailed comparison viewer for the two corresponding squares.

🏆 Enhanced Top Pairs Gallery: Instantly see a gallery of the most similar pairs, with powerful filtering and sorting options. Sort by correlation score, quality, or a combination of both!

🔍 Zoomable Pairwise Viewer: Click on any pair in the heatmap or gallery to open a detailed inspector. Zoom, pan, and scrutinize the two squares side-by-side, with their optimal rotation and detailed stats.

📊 System Resource Monitor: Keep an eye on your CPU, RAM, and GPU usage right in the status bar, so you know how hard the tool is working for you.

💾 Export Your Findings: Export the complete correlation matrix and all metadata to a CSV file for further analysis or documentation.

⚙️ How It Works: The Secret Sauce v2.0
The tool follows a state-of-the-art image processing pipeline to deliver the most accurate results.

      [Image 1]      [Image 2]      [Image N]
           |              |              |
           V              V              V
+---------------------------------------------------+
|     1. LOAD & DETECT SQUARES (OpenCV)             |
|  Automatically finds all particle squares using   |
|  advanced filtering and contour detection.        |
+---------------------------------------------------+
           |
           V
+---------------------------------------------------+
|      2. PRE-PROCESS & SCORE QUALITY               |
|                                                   |
|   [Square] -> [Enhance] -> [Calc. Quality Score]  |
|      |           |                |               |
|   Original   (CLAHE)     (Texture & Edge Metrics) |
+---------------------------------------------------+
           |
           V
+---------------------------------------------------+
|      3. DYNAMIC ROTATIONAL CORRELATION            |
|                                                   |
|      FOR each pair of squares (A, B):             |
|          Weight = f(Quality(A), Quality(B))       |
|          Score = Correlate(A, B) * Weight         |
|                                                   |
|  *Correlation uses NCC, SSIM, and Mutual Info* |
+---------------------------------------------------+
           |
           V
+---------------------------------------------------+
|            4. VISUALIZE & EXPLORE                 |
|                                                   |
|  [Interactive Heatmap] & [Filterable Gallery]     |
|         & [Detailed Zoom Viewer]                  |
+---------------------------------------------------+

🚀 Getting Started
Ready to play detective with your own data? It's easy to get started.

1. Prerequisites
You'll need Python 3. It is recommended to set up a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Then, install the required libraries from the requirements.txt file:

pip install -r requirements.txt

2. Run the Script
Save the code as cryo_correlator_v2.py and run it from your terminal:

python cryo_correlator_v2.py

3. Using the App
Click "📂 Load Images" and select one or more of your 2D class average files.

Choose your Detection Mode (Automatic is a great start).

Click "🔍 Detect Squares". The status bar will confirm how many squares were found.

Once detection is complete, click "📊 Analyze Correlations". This may take a moment, especially with many squares, as it's performing thousands of
