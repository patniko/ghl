#!/usr/bin/env python
"""
Debug script for investigating image scaling issues in DICOM files.
This script focuses on the crop_and_scale function and provides detailed
visualization and logging of each step in the process.
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
        logging.FileHandler("debug_scaling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_scaling")

# Create output directory for debug images
DEBUG_DIR = Path("./results/debug_images")
DEBUG_DIR.mkdir(exist_ok=True)

def save_debug_image(img, step_name, frame_idx, additional_info=""):
    """Save an image for debugging purposes with detailed information."""
    if img is None:
        logger.warning(f"Cannot save None image for {step_name}, frame {frame_idx}")
        return
    
    # Create a figure with two subplots - one for the image and one for text info
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 1]})
    
    # Display the image
    if len(img.shape) == 3 and img.shape[2] == 3:
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(img, cmap='gray')
    
    ax1.set_title(f"Frame {frame_idx}: {step_name}")
    
    # Display image information
    info_text = (
        f"Shape: {img.shape}\n"
        f"Data type: {img.dtype}\n"
        f"Min value: {np.min(img)}\n"
        f"Max value: {np.max(img)}\n"
        f"Mean value: {np.mean(img):.2f}\n"
        f"Is contiguous: {img.flags['C_CONTIGUOUS']}\n"
        f"{additional_info}"
    )
    ax2.text(0.1, 0.5, info_text, fontsize=10, va='center')
    ax2.axis('off')
    
    # Save the figure
    filename = f"{frame_idx:03d}_{step_name.replace(' ', '_')}.png"
    plt.savefig(DEBUG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved debug image: {filename}")

def crop_and_scale_debug(img, frame_idx, res=(112, 112), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """
    Debug version of crop_and_scale function with detailed logging and visualization.
    
    Args:
        img (np.ndarray): Input image
        frame_idx (int): Frame index for logging
        res (tuple): Target resolution (width, height)
        interpolation: OpenCV interpolation method
        zoom (float): Zoom factor
        
    Returns:
        np.ndarray: Cropped and scaled image
    """
    logger.info(f"Processing frame {frame_idx}")
    logger.info(f"Input image shape: {img.shape}, dtype: {img.dtype}")
    
    # Save original image
    save_debug_image(img, "original", frame_idx)
    
    # Check if image is valid
    if img is None:
        logger.error(f"Frame {frame_idx}: Image is None")
        raise ValueError("Invalid image: image is None")
    
    if img.size == 0:
        logger.error(f"Frame {frame_idx}: Image size is 0")
        raise ValueError("Invalid image: empty (size=0)")
    
    if img.shape[0] == 0 or img.shape[1] == 0:
        logger.error(f"Frame {frame_idx}: Image has zero dimension - shape: {img.shape}")
        raise ValueError(f"Invalid image: zero dimension - shape: {img.shape}")
    
    # Get input resolution
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    
    logger.info(f"Input resolution: {in_res}, aspect ratio: {r_in:.4f}")
    logger.info(f"Output resolution: {res}, aspect ratio: {r_out:.4f}")
    
    # Crop to match aspect ratio
    img_after_ratio_crop = img.copy()
    
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        logger.info(f"r_in > r_out: Cropping width with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= in_res[0] // 2:
            old_padding = padding
            padding = max(0, in_res[0] // 2 - 1)
            logger.warning(f"Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
        
        # Check if the resulting slice is valid
        if padding >= in_res[0]:
            logger.error(f"Invalid padding: {padding} >= image width {in_res[0]}")
            raise ValueError(f"Invalid padding: {padding} >= image width {in_res[0]}")
        
        try:
            img_after_ratio_crop = img[:, padding:in_res[0]-padding]
            logger.info(f"After width crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            logger.error(f"Error during width cropping: {str(e)}")
            logger.error(f"Attempted slice: img[:, {padding}:{in_res[0]-padding}]")
            raise
    
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        logger.info(f"r_in < r_out: Cropping height with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= in_res[1] // 2:
            old_padding = padding
            padding = max(0, in_res[1] // 2 - 1)
            logger.warning(f"Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
        
        # Check if the resulting slice is valid
        if padding >= in_res[1]:
            logger.error(f"Invalid padding: {padding} >= image height {in_res[1]}")
            raise ValueError(f"Invalid padding: {padding} >= image height {in_res[1]}")
        
        try:
            img_after_ratio_crop = img[padding:in_res[1]-padding, :]
            logger.info(f"After height crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            logger.error(f"Error during height cropping: {str(e)}")
            logger.error(f"Attempted slice: img[{padding}:{in_res[1]-padding}, :]")
            raise
    
    # Save image after aspect ratio cropping
    save_debug_image(img_after_ratio_crop, "after_ratio_crop", frame_idx, 
                     f"r_in: {r_in:.4f}, r_out: {r_out:.4f}\nPadding: {padding}")
    
    # Apply zoom
    img_after_zoom = img_after_ratio_crop.copy()
    
    if zoom != 0 and img_after_ratio_crop.shape[0] > 2 and img_after_ratio_crop.shape[1] > 2:
        pad_x = max(1, round(int(img_after_ratio_crop.shape[1] * zoom)))
        pad_y = max(1, round(int(img_after_ratio_crop.shape[0] * zoom)))
        
        logger.info(f"Applying zoom with pad_x={pad_x}, pad_y={pad_y}")
        
        # Check if zoom parameters are valid
        if pad_x >= img_after_ratio_crop.shape[1] // 2 or pad_y >= img_after_ratio_crop.shape[0] // 2:
            logger.warning(f"Zoom padding too large for image size, skipping zoom")
        else:
            try:
                img_after_zoom = img_after_ratio_crop[pad_y:-pad_y, pad_x:-pad_x]
                logger.info(f"After zoom: shape = {img_after_zoom.shape}")
            except Exception as e:
                logger.error(f"Error during zoom: {str(e)}")
                logger.error(f"Attempted slice: img[{pad_y}:-{pad_y}, {pad_x}:-{pad_x}]")
                # Continue with unzoomed image
                img_after_zoom = img_after_ratio_crop
    
    # Save image after zoom
    save_debug_image(img_after_zoom, "after_zoom", frame_idx, 
                     f"Zoom factor: {zoom}")
    
    # Resize image
    if img_after_zoom.shape[0] > 0 and img_after_zoom.shape[1] > 0:
        try:
            final_img = cv2.resize(img_after_zoom, res, interpolation=interpolation)
            logger.info(f"After resize: shape = {final_img.shape}")
        except Exception as e:
            logger.error(f"Error during resize: {str(e)}")
            logger.error(f"Attempted to resize image of shape {img_after_zoom.shape} to {res}")
            raise
    else:
        logger.error(f"Invalid image dimensions after cropping: {img_after_zoom.shape}")
        raise ValueError(f"Invalid image dimensions after cropping: {img_after_zoom.shape}")
    
    # Save final image
    save_debug_image(final_img, "final", frame_idx)
    
    return final_img

def process_dicom_file(file_path):
    """Process a single DICOM file with detailed debugging."""
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
        
        # Process each frame
        for i in range(min(5, len(pixels))):  # Process up to 5 frames for brevity
            logger.info(f"Processing frame {i} of {len(pixels)}")
            
            try:
                # Process the frame
                result = crop_and_scale_debug(pixels[i], i)
                logger.info(f"Successfully processed frame {i}")
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info("Completed processing DICOM file")
        
    except Exception as e:
        logger.error(f"Error processing DICOM file: {str(e)}")
        logger.error(traceback.format_exc())

def main(file_path=None):
    """
    Main function to run the debugging script.
    
    Args:
        file_path (str, optional): Path to the DICOM file to analyze. If not provided,
                                  will use command line argument or default file.
    """
    # Determine file path from arguments or use default
    if file_path is None:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = "./data/epiq7/1.2.840.113654.2.70.1.173304721797905812758989059075929126362"
    
    logger.info(f"Starting debug script for file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    process_dicom_file(file_path)
    
    logger.info("Debug script completed")
    print(f"Debug images saved to {DEBUG_DIR}")
    print(f"Debug log saved to debug_scaling.log")

if __name__ == "__main__":
    main()
