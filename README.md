# computer-vision-panorama
# Multi-Image Panorama Stitching Pipeline

An end-to-end computer vision pipeline implemented in Python and OpenCV that aligns and stitches multiple overlapping images into a seamless panoramic view without using OpenCV's built-in `cv2.Stitcher` class.

## Project Overview
This project demonstrates the core geometric and algorithmic principles of image stitching:
1. **Feature Detection & Description:** Extracting distinctive keypoints and binary descriptors across consecutive image frames using ORB (Oriented FAST and Rotated BRIEF).
2. **Feature Matching:** Computing pairwise correspondences with Brute-Force Matcher using Hamming distance and filtering to the top 18% strongest matches.
3. **Perspective Alignment (Homography):** Estimating 3x3 homography transformation matrices via RANSAC outlier rejection (threshold set to 4.0) and warping frames into a unified coordinate frame using `cv2.warpPerspective`.
4. **Automated Boundary Cropping:** Eliminating residual black borders automatically through contour detection and iterative morphological erosion (`cv2.erode`).

## Tech Stack
* **Language:** Python 3.x
* **Libraries:** OpenCV (`cv2`), NumPy, Matplotlib

## Execution
Ensure the required dependencies are installed:
```bash
pip install opencv-python numpy matplotlib


python panorama_stitching.py
