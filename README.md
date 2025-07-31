# **🔬 Cryo-Correlator v2.0: The Ultimate 2D Class Detective 🕵️**

**Ever** stared at hundreds of fuzzy gray squares **from your cryo-EM run and wondered, "Are... are you two related?" Now you can find out, with more power and precision than ever before!**

Cryo-Correlator is a desktop tool with an enhanced GUI that dives into your 2D class average images and finds the hidden similarities between them. It's like a dating app for your protein views, but with more math and less awkward small talk.

_(You can replace this with a screenshot of your actual GUI!)_

### **🤔 What's the Big Idea?**

In cryo-electron microscopy, we get thousands of images of individual particles, which are then averaged into "2D classes." These classes represent different views of a molecule. Sometimes, you get very similar-looking classes from different images or even within the same one.

Finding these similarities is crucial for:

- **Validating your structure:** Do different subsets of data produce the same views?
- **Sorting through heterogeneity:** Are there different conformations or populations of your particle?
- **Cleaning up your dataset:** Identifying and grouping the best, most consistent views.

Doing this by eye is slow, tedious, and subjective. This tool automates and supercharges the process!

### **✨ Key Features**

- **🤖 Dual-Mode Detection:**
  - **Automatic Mode:** Uses **OpenCV** with advanced filters to automatically find every single class average square in your images, no matter the layout.
  - **Reference-Based Mode:** Select a specific particle view, and the app will hunt for matches across all your images using template matching.
- **🔬 Sophisticated Quality Metrics:** It doesn't just look at pixels; it _understands_ them. Each detected square is scored for **texture and edge quality**, giving you a quantifiable measure of how good it is.
- **🧠 Dynamically Weighted Correlation:** The analysis intelligently weighs correlation scores. A match between two high-quality squares is considered more significant than a match between two noisy, low-quality ones.
- **🔄 Advanced Rotational Matching:** Is your particle view slightly tilted? No problem. The script rotates each image to find the **best possible alignment**, using a blend of Normalized Cross-Correlation (NCC), Structural Similarity Index (SSIM), and Mutual Information for a highly robust score.
- **🌡️ Interactive Heatmap (Powered by Plotly):** Visualize the entire NxN correlation matrix. The brighter the spot, the hotter the match! **Click any pixel** to instantly open a detailed comparison viewer for the two corresponding squares.
- **🏆 Enhanced Top Pairs Gallery:** Instantly see a gallery of the most similar pairs, with powerful filtering and sorting options. Sort by correlation score, quality, or a combination of both!
- **🔍 Zoomable Pairwise Viewer:** Click on any pair in the heatmap or gallery to open a detailed inspector. Zoom, pan, and scrutinize the two squares side-by-side, with their optimal rotation and detailed stats.
- **📊 System Resource Monitor:** Keep an eye on your **CPU, RAM, and GPU** usage right in the status bar, so you know how hard the tool is working for you.
- **💾 Export Your Findings:** Export the complete correlation matrix and all metadata to a **CSV file** for further analysis or documentation.

### **⚙️ How It Works: The Secret Sauce v2.0**

The tool follows a state-of-the-art image processing pipeline to deliver the most accurate results.

\[Image 1\] \[Image 2\] \[Image N\]  
| | |  
V V V  
+---------------------------------------------------+  
| 1. LOAD & DETECT SQUARES (OpenCV) |  
| Automatically finds all particle squares using |  
| advanced filtering and contour detection. |  
+---------------------------------------------------+  
|  
V  
+---------------------------------------------------+  
| 2. PRE-PROCESS & SCORE QUALITY |  
| |  
| \[Square\] -> \[Enhance\] -> \[Calc. Quality Score\] |  
| | | | |  
| Original (CLAHE) (Texture & Edge Metrics) |  
+---------------------------------------------------+  
|  
V  
+---------------------------------------------------+  
| 3. DYNAMIC ROTATIONAL CORRELATION |  
| |  
| FOR each pair of squares (A, B): |  
| Weight = f(Quality(A), Quality(B)) |  
| Score = Correlate(A, B) \* Weight |  
| |  
| \*Correlation uses NCC, SSIM, and Mutual Info\* |  
+---------------------------------------------------+  
|  
V  
+---------------------------------------------------+  
| 4. VISUALIZE & EXPLORE |  
| |  
| \[Interactive Heatmap\] & \[Filterable Gallery\] |  
| & \[Detailed Zoom Viewer\] |  
+---------------------------------------------------+  
(Scroll to the end of this document for further explanation of the methods used)

### **🚀 Getting Started**

Ready to play detective with your own data? It's easy to get started.

#### **1\. Prerequisites**

You'll need Python 3. It is recommended to set up a virtual environment.

python -m venv venv  
source venv/bin/activate # On Windows, use \`venv\\Scripts\\activate\`  

Then, install the required libraries from the requirements.txt file:

pip install -r requirements.txt  

#### **2\. Run the Script**

Save the code as cryo_correlator_v2.py and run it from your terminal:

python cryo_2d_corr_v2.py  

#### **3\. Using the App**

1. Click **"📂 Load Images"** and select one or more of your 2D class average files.
2. Choose your **Detection Mode** (Automatic is a great start).
3. Click **"🔍 Detect Squares"**. The status bar will confirm how many squares were found.
4. Once detection is complete, click **"📊 Analyze Correlations"**. This may take a moment, especially with many squares, as it's performing thousands of

The 2D Correlation Method Explained 🔬
The application measures how different parts of an image have moved between two photos. This technique is a form of Digital Image Correlation (DIC). The specific method used in your script is GPU-Accelerated, Subset-Based Normalized Cross-Correlation (NCC) performed in the frequency domain.

Here is a step-by-step breakdown of how it works:

Step 1: Subsetting the Image
The first image (your reference image) is not analyzed as a whole. Instead, it's broken down into many small, square sections called subsets or blocks.

In the Script: The block_size parameter controls the dimensions of these squares (e.g., 32x32 pixels).

Step 2: Searching for a Match
For each subset from Image 1, the algorithm's goal is to find where that exact pattern of pixels has moved to in Image 2.

It defines a larger search area in Image 2, centered on the original subset's coordinates. This gives the algorithm a region to "look around" in.

In the Script: The search_area_multiplier determines the size of this search area. A value of 2.0 means the search area is twice the width and height of the subset.

Step 3: Normalized Cross-Correlation (NCC)
This is the mathematical heart of the matching process. Cross-correlation is a metric that measures the similarity between the subset from Image 1 and every possible corresponding area within the search area of Image 2.

Why "Normalized"? Simple correlation is sensitive to changes in lighting. If Image 2 is brighter or dimmer than Image 1, the correlation values would be skewed. Normalization fixes this by first subtracting the mean brightness from both the subset and the search area before comparing them. This makes the algorithm robust to simple lighting variations.

In the Script: The lines block_norm = block - cp.mean(block) and search_area_norm = search_area - cp.mean(search_area) perform this crucial normalization step.

Step 4: Acceleration with the Fast Fourier Transform (FFT)
Calculating correlation pixel-by-pixel is extremely slow. The script uses a massive shortcut called the Convolution Theorem. This theorem states that the complex math of correlation in the pixel domain is equivalent to simple element-wise multiplication in the frequency domain.

The Fast Fourier Transform (FFT) is an incredibly efficient algorithm for converting an image into its frequency representation.

The script uses the GPU (via cupy) to perform FFTs on both the subset and the search area, multiplies them, and then performs an inverse FFT to get the correlation result. This is dramatically faster than the traditional method.

In the Script: The line correlation = cp.fft.ifft2(...) performs this entire frequency-domain operation.

Step 5: Finding the Peak and Displacement
The result of the correlation is a 2D surface where the value at each point represents the "match quality." The highest point, or peak, on this surface corresponds to the best match.

The algorithm finds the (x, y) coordinates of this peak.

The displacement is the difference between the peak's location and the center of the search area. This gives a vector (dx, dy) representing how far that subset moved.

In the Script: cp.argmax(correlation) finds the location of the peak, and the subsequent lines calculate dx and dy.

This process is repeated for every single subset, resulting in a full vector field of displacements across the entire image.

3. Quality Control (QC) Methods 🧐
How do you know if the correlation results are accurate and trustworthy? Quality control is essential.

1. Visual Inspection of the Heatmap
The primary QC tool in this application is the final displacement heatmap. This map visualizes the magnitude of movement (sqrt(dx² + dy²)) for every subset. When you look at it, you should check for:

Smoothness and Physicality: In most real-world scenarios (like material deformation), displacements should be smooth and continuous. The heatmap should show smooth gradients of color. Jagged, noisy, or chaotic-looking areas suggest that the algorithm failed to find a reliable match in those regions.

Outliers: Look for isolated "hot spots" or "cold spots" that don't fit the surrounding pattern. These are likely erroneous vectors where the correlation algorithm latched onto a false peak, perhaps due to repetitive textures, reflections, or significant changes in the surface.

2. The Correlation Coefficient (Implicit QC)
The peak value of the normalized cross-correlation surface itself is the most direct measure of match quality.

A peak value close to 1.0 indicates a very confident, unambiguous match.

A low peak value (e.g., below 0.7) indicates a poor match. The algorithm couldn't find a convincing look-alike for the subset.

While the current script doesn't explicitly record or use this coefficient for QC, a more advanced version could use it to mask out bad vectors. For instance, it could refuse to display any displacement vector where the peak correlation coefficient was below a certain threshold.

3. Parameter Tuning
The quality of the result is highly dependent on the parameters you choose.

Block Size: If the block size is too small, it may not contain enough unique texture to be identified reliably. If it's too large, it will average out fine-scale details in the motion.

Search Area: If the search area is too small, the algorithm might lose track of a subset that moved a large distance. If it's too large, it increases computation time and the risk of finding a false match.

Running tests with different parameters and observing the effect on the final heatmap is a practical form of quality control.
