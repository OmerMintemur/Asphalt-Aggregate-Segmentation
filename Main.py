# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:24:46 2024
@author: OMER
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import random 
# ================================
# Utility Functions
# ================================

def random_color():
    """Generate a random color tuple (BGR)."""
    return tuple([random.randint(0, 255) for _ in range(3)])

def label_and_colorize(binary_mask, base_image=None):
    """
    Label connected components in binary mask and color each one.
    Returns a colored overlay image.
    """
    labeled_mask = label(binary_mask)
    if base_image is None:
        # Create a blank color image
        color_image = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    else:
        color_image = base_image.copy()

    for region in regionprops(labeled_mask):
        # Generate a random color for this label
        color = random_color()
        # Draw contour around each labeled region
        coords = region.coords
        for coord in coords:
            color_image[coord[0], coord[1]] = color

    cv2.imwrite("Results\\Colored.png", color_image)

    return labeled_mask, color_image

def distance_between_two_points(points):
    """Compute average distance between all pairs of points."""
    average = 0
    for p in range(len(points)):
        temp = 0
        for c in range(p + 1, len(points)):
            temp += math.hypot(points[p][0] - points[c][0], points[p][1] - points[c][1])
        average += temp / len(points)
    return average

def find_zero_pixels(image):
    """Return a binary mask where zero pixels are 0 and others 1."""
    return np.where(image == 0, 0, 1).astype(np.uint8)

def equalize_and_threshold(channel_diff, threshold_val=180):
    """Equalize histogram and apply binary threshold."""
    eq = cv2.equalizeHist(channel_diff)
    _, binary = cv2.threshold(eq, threshold_val, 255, cv2.THRESH_BINARY)
    binary = cv2.equalizeHist(binary)
    return binary

def apply_mean_filter(image, kernel_size=7):
    """Apply a normalized mean filter to smooth the image."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return cv2.filter2D(image, -1, kernel)

def dilate_image(image, kernel_shape=cv2.MORPH_CROSS, kernel_size=17, iterations=1):
    """Dilate image using specified structuring element."""
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=iterations)

def fill_holes(image, structure_size=5):
    """Fill small holes in binary image."""
    structure = np.ones((structure_size, structure_size), np.float32)
    filled = binary_fill_holes(image, structure=structure)
    return (filled * 255).astype(np.uint8)

def save_image(path, image):
    """Utility to save image."""
    cv2.imwrite(path, image)

def compute_contours(image, binary_mask, threshold=750):
    """Draw contours above threshold on image and return centers."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) > threshold:
            cv2.drawContours(image, [c], 0, (0, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(c)
            centers.append((x + w/2, y + h/2))
    return image, centers

# ================================
# Main Pipeline
# ================================

def process_image(input_path, output_dir="Results"):
    """Full processing pipeline for a single image."""
    # Read images
    gray = cv2.imread(input_path, 0)
    color = cv2.imread(input_path)

    save_image(f"{output_dir}/First.png", gray)

    # Zero-pixel mask
    mask = find_zero_pixels(gray)

    # Convert to HSV and extract S-V channel
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    temp = equalize_and_threshold(s - v)
    save_image(f"{output_dir}/Second.png", temp)

    # Mean filter
    dst = apply_mean_filter(temp)
    save_image(f"{output_dir}/Third.png", dst)

    # Otsu threshold
    _, binary = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    save_image(f"{output_dir}/Fourth.png", binary.astype(np.uint8))

    # Dilate
    temp = dilate_image(binary)
    
    # Invert and smooth
    dst = apply_mean_filter(~temp)
    dst = np.multiply(mask, dst)

    save_image(f"{output_dir}/Fifth.png", dst)

    # Fill holes
    dst = fill_holes(dst)
    save_image(f"{output_dir}/Sixth.png", dst)

    # Contours and centers
    color_with_contours, centers = compute_contours(color.copy(), dst)
    save_image(f"{output_dir}/Seventh.png", color_with_contours)

    return dst, color_with_contours, centers

# ================================
# Run Example
# ================================

if __name__ == "__main__":
    dst_mask, contour_image, centers = process_image("Example.tif")
    labeled_image, color_image = label_and_colorize(dst_mask)
    print(f"Found {len(centers)} large objects.")
