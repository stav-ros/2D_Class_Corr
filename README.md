# **🔬 Cryo-Correlator: The 2D Class Detective 🕵️**

**Ever** stared at hundreds of fuzzy gray squares **from your cryo-EM run and wondered, "Are... are you two related?" Now you can find out\!**

Cryo-Correlator is a desktop tool with a simple GUI that dives into your 2D class average images and finds the hidden similarities between them. It's like a dating app for your protein views, but with more math and less awkward small talk.

*(You can replace this with a screenshot of your actual GUI\!)*

### **🤔 What's the Big Idea?**

In cryo-electron microscopy, we get thousands of images of individual particles, which are then averaged into "2D classes." These classes represent different views of a molecule. Sometimes, you get very similar-looking classes from different images or even within the same one.

Finding these similarities is crucial for:

* **Validating your structure:** Do different subsets of data produce the same views?  
* **Sorting through heterogeneity:** Are there different conformations or populations of your particle?  
* **Cleaning up your dataset:** Identifying and grouping the best, most consistent views.

Doing this by eye is slow, tedious, and subjective. This tool automates the process\!

### **✨ Key Features**

* **Load Anything:** Don't have just 3 images? Have 5? 10? Go crazy\! Load as many grid images as you want.  
* **Dynamic Square Detection:** No more rigid 10x10 grids. The app uses **OpenCV** to automatically find every single class average square in your images, no matter the layout.  
* **Ignores Text (Like a Boss):** Those pesky resolution and particle count numbers written on the squares? The script automatically detects and masks them out, so it only compares the actual image data.  
* **Rotational Matching:** Is your particle view slightly tilted in one square? No problem. The script rotates each image to find the **best possible match**, giving you a true similarity score.  
* **Interactive Heatmap:** Visualize the entire 300x300 (or NxN) correlation matrix. The brighter the spot, the hotter the match\! Click any pixel to instantly see the two squares being compared.  
* **Top Pairs Gallery:** Instantly see a gallery of the most similar pairs, perfect for a quick overview of your data's consistency.

### **⚙️ How It Works: The Secret Sauce**

The tool follows a sophisticated image processing pipeline to get you the most accurate results.

      \[Image 1\]      \[Image 2\]      \[Image N\]  
           |              |              |  
           V              V              V  
\+---------------------------------------------------+  
|           1\. LOAD & DETECT SQUARES (OpenCV)       |  
|  Finds all the little boxes in your big images.   |  
\+---------------------------------------------------+  
           |  
           V  
\+---------------------------------------------------+  
|          2\. PROCESS EACH SQUARE INDIVIDUALLY      |  
|                                                   |  
|   \[Square\] \-\> \[Mask Text\] \-\> \[Apply Circle Mask\]  |  
|      ^              |                  |          |  
|      |              V                  V          |  
|   Original   (Ignore this\!)   (Prep for rotation) |  
\+---------------------------------------------------+  
           |  
           V  
\+---------------------------------------------------+  
|         3\. ROTATIONAL CORRELATION (Sci-Image)     |  
|                                                   |  
|      FOR each pair of squares (A, B):             |  
|          max\_corr \= \-1                            |  
|          FOR angle in 0..360:                     |  
|              B\_rotated \= rotate(B, angle)         |  
|              corr \= NCC(A, B\_rotated)             |  
|              IF corr \> max\_corr: max\_corr \= corr  |  
|                                                   |  
\+---------------------------------------------------+  
           |  
           V  
\+---------------------------------------------------+  
|               4\. VISUALIZE RESULTS                |  
|                                                   |  
|   \[Interactive Heatmap\] & \[Top Pairs Gallery\]     |  
|                                                   |  
\+---------------------------------------------------+

### **🚀 Getting Started**

Ready to play detective with your own data? It's easy to get started.

#### **1\. Prerequisites**

You'll need Python 3 and a few scientific libraries. You can install them all with pip:

pip install opencv-python scikit-image numpy matplotlib seaborn Pillow

#### **2\. Run the Script**

Save the code as a Python file (e.g., cryo\_correlator.py) and run it from your terminal:

python cryo\_correlator.py

#### **3\. Using the App**

1. Click **"Load Grid Image(s)"** and select one or more of your 2D class average files.  
2. Wait for the status bar to confirm how many squares were found.  
3. Click **"Run Cross-Correlation"**. This may take a moment, especially with many squares, as it's performing thousands of comparisons\!  
4. Explore your results\!  
   * Click on the **Correlation Matrix** to inspect any specific pair.  
   * Browse the **Top Correlations** tab for a quick look at the best matches.  
   * Use the **controls** on the left to filter the matrix by a correlation threshold.

Happy correlating\! May you find all the hidden gems in your datasets. ✨
