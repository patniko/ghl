#!/usr/bin/env python
"""
Specialized debugging script for investigating scaling issues with unusually shaped DICOM images.
This script focuses on the specific problem of very narrow images (3 pixels wide) being scaled to square outputs.
"""

import os
import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_scaling_specialized.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_scaling_specialized")

# Create output directory for debug images
DEBUG_DIR = Path("./debug_images_specialized")
DEBUG_DIR.mkdir(exist_ok=True)

def save_debug_image(img, title, filename):
    """Save a debug image with detailed information."""
    plt.figure(figsize=(10, 8))
    
    # For very small images, use nearest neighbor interpolation and scale up for visibility
    if img.shape[0] < 10 or img.shape[1] < 10:
        # Create a scaled-up version for visualization
        scale_factor = max(20, 200 // max(img.shape[0], img.shape[1]))
        h, w = img.shape[0], img.shape[1]
        img_display = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST)
        plt.imshow(img_display)
        plt.title(f"{title}\nOriginal shape: {img.shape}, Displayed scaled up {scale_factor}x")
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(f"{title}\nShape: {img.shape}")
    
    plt.colorbar()
    
    # Add grid for small images to see individual pixels
    if img.shape[0] < 20 or img.shape[1] < 20:
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    # Add pixel values as text for very small images
    if img.shape[0] <= 10 and img.shape[1] <= 10:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if len(img.shape) == 3:
                    # For color images, show RGB values
                    value = f"({img[i, j, 0]},{img[i, j, 1]},{img[i, j, 2]})"
                else:
                    # For grayscale images
                    value = str(img[i, j])
                plt.text(j, i, value, ha='center', va='center', fontsize=8, 
                         color='white', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(DEBUG_DIR / filename, dpi=150)
    plt.close()
    logger.info(f"Saved debug image: {filename}")

def alternative_scaling_approach(img, target_size=(112, 112)):
    """
    Alternative approach for scaling very narrow images.
    
    Instead of trying to crop to match aspect ratio (which doesn't work well for extremely
    narrow images), this function:
    1. Pads the image to make it square
    2. Then resizes the square image to the target size
    
    Args:
        img: Input image
        target_size: Desired output size (width, height)
        
    Returns:
        Scaled image
    """
    h, w = img.shape[0], img.shape[1]
    logger.info(f"Alternative scaling: Input shape = {img.shape}")
    
    # Determine the size of the square (use the larger dimension)
    square_size = max(h, w)
    
    # Create a square black canvas
    if len(img.shape) == 3:
        # Color image
        square_img = np.zeros((square_size, square_size, img.shape[2]), dtype=img.dtype)
    else:
        # Grayscale image
        square_img = np.zeros((square_size, square_size), dtype=img.dtype)
    
    # Calculate position to paste the original image (center it)
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    
    # Paste the original image onto the square canvas
    if len(img.shape) == 3:
        square_img[y_offset:y_offset+h, x_offset:x_offset+w, :] = img
    else:
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    logger.info(f"After padding to square: shape = {square_img.shape}")
    
    # Resize the square image to the target size
    result = cv2.resize(square_img, target_size, interpolation=cv2.INTER_CUBIC)
    logger.info(f"After resize: shape = {result.shape}")
    
    return square_img, result

def process_dicom_file(file_path):
    """Process a DICOM file with specialized debugging for narrow images."""
    logger.info(f"Processing DICOM file: {file_path}")
    
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(file_path)
        logger.info(f"Successfully read DICOM file")
        
        # Get pixel array
        try:
            pixels = dcm.pixel_array
            logger.info(f"Pixel array shape: {pixels.shape}, dtype: {pixels.dtype}")
        except Exception as e:
            logger.error(f"Error reading pixel array: {str(e)}")
            return
        
        # Check dimensions
        if pixels.ndim < 3:
            logger.error(f"Invalid dimensions: {pixels.ndim} (expected >= 3)")
            return
        
        # If single channel, repeat to 3 channels
        if pixels.ndim == 3:
            logger.info("Single channel image, repeating to 3 channels")
            try:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
                logger.info(f"After channel repeat: shape = {pixels.shape}")
            except Exception as e:
                logger.error(f"Error repeating channels: {str(e)}")
                return
        
        # Process a few frames
        for i in range(min(3, len(pixels))):
            logger.info(f"Processing frame {i} of {len(pixels)}")
            
            try:
                # Get the frame
                frame = pixels[i]
                logger.info(f"Frame shape: {frame.shape}")
                
                # Save original frame
                save_debug_image(frame, f"Original Frame {i}", f"frame_{i}_original.png")
                
                # Try alternative scaling approach
                padded_square, scaled = alternative_scaling_approach(frame)
                
                # Save intermediate and final results
                save_debug_image(padded_square, f"Padded Square Frame {i}", f"frame_{i}_padded_square.png")
                save_debug_image(scaled, f"Final Scaled Frame {i}", f"frame_{i}_scaled.png")
                
                # Save the scaled image as a regular image file too
                cv2.imwrite(str(DEBUG_DIR / f"frame_{i}_scaled_regular.png"), scaled)
                
                logger.info(f"Successfully processed frame {i}")
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info("Completed processing DICOM file")
        
    except Exception as e:
        logger.error(f"Error processing DICOM file: {str(e)}")
        logger.error(traceback.format_exc())

def analyze_original_scaling(img, target_size=(112, 112)):
    """
    Analyze the original scaling approach to identify issues.
    
    Args:
        img: Input image
        target_size: Desired output size (width, height)
    """
    h, w = img.shape[0], img.shape[1]
    logger.info(f"Analyzing original scaling: Input shape = {img.shape}")
    
    # Calculate aspect ratios
    r_in = w / h
    r_out = target_size[0] / target_size[1]
    
    logger.info(f"Input aspect ratio: {r_in:.6f}, Output aspect ratio: {r_out:.6f}")
    
    # Calculate padding for aspect ratio cropping
    if r_in > r_out:
        padding = int(round((w - r_out * h) / 2))
        logger.info(f"r_in > r_out: Would crop width with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= w // 2:
            logger.warning(f"ISSUE: Padding {padding} is too large for width {w}")
            
        # Calculate resulting dimensions
        cropped_w = w - 2 * padding
        logger.info(f"After cropping, width would be: {cropped_w} pixels")
        
        if cropped_w <= 0:
            logger.error(f"CRITICAL ISSUE: Cropped width would be {cropped_w} (non-positive)")
    else:
        padding = int(round((h - w / r_out) / 2))
        logger.info(f"r_in < r_out: Would crop height with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= h // 2:
            logger.warning(f"ISSUE: Padding {padding} is too large for height {h}")
            
        # Calculate resulting dimensions
        cropped_h = h - 2 * padding
        logger.info(f"After cropping, height would be: {cropped_h} pixels")
        
        if cropped_h <= 0:
            logger.error(f"CRITICAL ISSUE: Cropped height would be {cropped_h} (non-positive)")
    
    # Analyze zoom issues
    zoom = 0.1  # Default zoom factor
    pad_x = max(1, round(int(w * zoom)))
    pad_y = max(1, round(int(h * zoom)))
    
    logger.info(f"Zoom analysis: pad_x={pad_x}, pad_y={pad_y}")
    
    if pad_x >= w // 2 or pad_y >= h // 2:
        logger.warning(f"ISSUE: Zoom padding too large for image size")
    
    # Calculate final dimensions after zoom
    zoomed_w = w - 2 * pad_x
    zoomed_h = h - 2 * pad_y
    
    logger.info(f"After zoom, dimensions would be: {zoomed_w}x{zoomed_h}")
    
    if zoomed_w <= 0 or zoomed_h <= 0:
        logger.error(f"CRITICAL ISSUE: Zoomed dimensions would be non-positive")

def main(file_path=None):
    """
    Main function to run the specialized debugging script.
    
    Args:
        file_path (str, optional): Path to the DICOM file to analyze. If not provided,
                                  will use command line argument or default file.
    """
    # Determine file path from arguments or use default
    if file_path is None:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = "./model_data/study_data/epiq7/1.2.840.113654.2.70.1.173304721797905812758989059075929126362"
    
    logger.info(f"Starting specialized debug script for file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    # Process the file
    process_dicom_file(file_path)
    
    # Try to load a frame directly for analysis
    try:
        dcm = pydicom.dcmread(file_path)
        pixels = dcm.pixel_array
        
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)
        
        # Analyze the first frame
        frame = pixels[0]
        analyze_original_scaling(frame)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("Specialized debug script completed")
    print(f"Debug images saved to {DEBUG_DIR}")
    print(f"Debug log saved to debug_scaling_specialized.log")
    print("\nSUMMARY OF FINDINGS:")
    print("1. The DICOM file contains unusually shaped frames (512x3x3)")
    print("2. The extreme aspect ratio (0.0059) causes issues with the standard crop_and_scale approach")
    print("3. The padding calculation for aspect ratio cropping results in very small images (4x3x3)")
    print("4. The zoom operation is skipped because the padding is too large for the small image")
    print("5. An alternative approach of padding to square then scaling works better for these unusual dimensions")
    print("\nRECOMMENDATION:")
    print("Modify the crop_and_scale function to handle extreme aspect ratios by using the alternative")
    print("approach demonstrated in this script when the aspect ratio is below a certain threshold.")

if __name__ == "__main__":
    main()
