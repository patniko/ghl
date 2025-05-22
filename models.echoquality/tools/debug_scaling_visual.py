#!/usr/bin/env python
"""
Visual debugging script for investigating image scaling issues in DICOM files.
This script provides a visual approach to debug the crop_and_scale function,
with side-by-side comparisons and detailed visualizations of problematic areas.
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
        logging.FileHandler("debug_scaling_visual.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_scaling_visual")

# Create output directory for debug images
DEBUG_DIR = Path("./results/debug_images_visual")
DEBUG_DIR.mkdir(exist_ok=True)

def visualize_image_details(img, title, save_path=None):
    """Create a detailed visualization of an image with histograms and edge detection."""
    if img is None:
        logger.warning(f"Cannot visualize None image: {title}")
        return
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Original image
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    if len(img.shape) == 3 and img.shape[2] == 3:
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(img, cmap='gray')
    ax1.set_title(f"{title}\nShape: {img.shape}, Type: {img.dtype}")
    
    # Histogram
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    if len(img.shape) == 3 and img.shape[2] == 3:
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax2.plot(hist, color=color)
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax2.plot(hist)
    ax2.set_title('Histogram')
    ax2.set_xlim([0, 256])
    
    # Edge detection
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    ax3.imshow(edges, cmap='gray')
    ax3.set_title('Edge Detection')
    
    # Image statistics
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax4.axis('off')
    stats_text = (
        f"Shape: {img.shape}\n"
        f"Data type: {img.dtype}\n"
        f"Min value: {np.min(img)}\n"
        f"Max value: {np.max(img)}\n"
        f"Mean value: {np.mean(img):.2f}\n"
        f"Standard deviation: {np.std(img):.2f}\n"
        f"Is contiguous: {img.flags['C_CONTIGUOUS']}\n"
    )
    ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close(fig)

def visualize_crop_process(img, padding, axis, title, save_path=None):
    """Visualize the cropping process with guidelines showing what will be cropped."""
    if img is None:
        logger.warning(f"Cannot visualize None image: {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the image
    if len(img.shape) == 3 and img.shape[2] == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap='gray')
    
    # Draw crop lines
    if axis == 'width':
        # Vertical lines for width cropping
        ax.axvline(x=padding, color='r', linestyle='--', linewidth=2)
        ax.axvline(x=img.shape[1]-padding, color='r', linestyle='--', linewidth=2)
        
        # Shade the areas to be cropped
        ax.axvspan(0, padding, color='r', alpha=0.3)
        ax.axvspan(img.shape[1]-padding, img.shape[1], color='r', alpha=0.3)
    else:
        # Horizontal lines for height cropping
        ax.axhline(y=padding, color='r', linestyle='--', linewidth=2)
        ax.axhline(y=img.shape[0]-padding, color='r', linestyle='--', linewidth=2)
        
        # Shade the areas to be cropped
        ax.axhspan(0, padding, color='r', alpha=0.3)
        ax.axhspan(img.shape[0]-padding, img.shape[0], color='r', alpha=0.3)
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved crop visualization to {save_path}")
    
    plt.show()
    plt.close(fig)

def visualize_comparison(original, processed, title, save_path=None):
    """Create a side-by-side comparison of original and processed images."""
    if original is None or processed is None:
        logger.warning(f"Cannot compare None images: {title}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display original image
    if len(original.shape) == 3 and original.shape[2] == 3:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original, cmap='gray')
    ax1.set_title(f"Original\nShape: {original.shape}")
    
    # Display processed image
    if len(processed.shape) == 3 and processed.shape[2] == 3:
        ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(processed, cmap='gray')
    ax2.set_title(f"Processed\nShape: {processed.shape}")
    
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
    
    plt.show()
    plt.close(fig)

def crop_and_scale_visual(img, frame_idx, res=(112, 112), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """
    Visual debug version of crop_and_scale function.
    
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
    
    # Create frame-specific debug directory
    frame_dir = DEBUG_DIR / f"frame_{frame_idx}"
    frame_dir.mkdir(exist_ok=True)
    
    # Visualize original image
    visualize_image_details(
        img, 
        f"Original Image (Frame {frame_idx})",
        save_path=frame_dir / "01_original_details.png"
    )
    
    # Check if image is valid
    if img is None:
        logger.error(f"Frame {frame_idx}: Image is None")
        return None
    
    if img.size == 0:
        logger.error(f"Frame {frame_idx}: Image size is 0")
        return None
    
    if img.shape[0] == 0 or img.shape[1] == 0:
        logger.error(f"Frame {frame_idx}: Image has zero dimension - shape: {img.shape}")
        return None
    
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
        
        # Visualize the crop
        visualize_crop_process(
            img, 
            padding, 
            'width', 
            f"Width Cropping (Frame {frame_idx})\nPadding: {padding} pixels",
            save_path=frame_dir / "02_width_crop_plan.png"
        )
        
        # Check if padding is valid
        if padding >= in_res[0] // 2:
            old_padding = padding
            padding = max(0, in_res[0] // 2 - 1)
            logger.warning(f"Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
            
            # Visualize the adjusted crop
            visualize_crop_process(
                img, 
                padding, 
                'width', 
                f"Adjusted Width Cropping (Frame {frame_idx})\nPadding: {padding} pixels",
                save_path=frame_dir / "03_adjusted_width_crop_plan.png"
            )
        
        # Check if the resulting slice is valid
        if padding >= in_res[0]:
            logger.error(f"Invalid padding: {padding} >= image width {in_res[0]}")
            return None
        
        try:
            img_after_ratio_crop = img[:, padding:in_res[0]-padding]
            logger.info(f"After width crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            logger.error(f"Error during width cropping: {str(e)}")
            logger.error(f"Attempted slice: img[:, {padding}:{in_res[0]-padding}]")
            return None
    
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        logger.info(f"r_in < r_out: Cropping height with padding {padding} pixels")
        
        # Visualize the crop
        visualize_crop_process(
            img, 
            padding, 
            'height', 
            f"Height Cropping (Frame {frame_idx})\nPadding: {padding} pixels",
            save_path=frame_dir / "02_height_crop_plan.png"
        )
        
        # Check if padding is valid
        if padding >= in_res[1] // 2:
            old_padding = padding
            padding = max(0, in_res[1] // 2 - 1)
            logger.warning(f"Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
            
            # Visualize the adjusted crop
            visualize_crop_process(
                img, 
                padding, 
                'height', 
                f"Adjusted Height Cropping (Frame {frame_idx})\nPadding: {padding} pixels",
                save_path=frame_dir / "03_adjusted_height_crop_plan.png"
            )
        
        # Check if the resulting slice is valid
        if padding >= in_res[1]:
            logger.error(f"Invalid padding: {padding} >= image height {in_res[1]}")
            return None
        
        try:
            img_after_ratio_crop = img[padding:in_res[1]-padding, :]
            logger.info(f"After height crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            logger.error(f"Error during height cropping: {str(e)}")
            logger.error(f"Attempted slice: img[{padding}:{in_res[1]-padding}, :]")
            return None
    
    # Visualize after aspect ratio cropping
    visualize_comparison(
        img, 
        img_after_ratio_crop, 
        f"Before vs After Aspect Ratio Crop (Frame {frame_idx})",
        save_path=frame_dir / "04_after_ratio_crop_comparison.png"
    )
    
    visualize_image_details(
        img_after_ratio_crop, 
        f"After Aspect Ratio Crop (Frame {frame_idx})",
        save_path=frame_dir / "05_after_ratio_crop_details.png"
    )
    
    # Apply zoom
    img_after_zoom = img_after_ratio_crop.copy()
    
    if zoom != 0 and img_after_ratio_crop.shape[0] > 2 and img_after_ratio_crop.shape[1] > 2:
        pad_x = max(1, round(int(img_after_ratio_crop.shape[1] * zoom)))
        pad_y = max(1, round(int(img_after_ratio_crop.shape[0] * zoom)))
        
        logger.info(f"Applying zoom with pad_x={pad_x}, pad_y={pad_y}")
        
        # Visualize the zoom
        visualize_crop_process(
            img_after_ratio_crop, 
            pad_x, 
            'width', 
            f"Zoom Plan - Width (Frame {frame_idx})\nPad X: {pad_x} pixels",
            save_path=frame_dir / "06_zoom_plan_width.png"
        )
        
        visualize_crop_process(
            img_after_ratio_crop, 
            pad_y, 
            'height', 
            f"Zoom Plan - Height (Frame {frame_idx})\nPad Y: {pad_y} pixels",
            save_path=frame_dir / "07_zoom_plan_height.png"
        )
        
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
    
    # Visualize after zoom
    visualize_comparison(
        img_after_ratio_crop, 
        img_after_zoom, 
        f"Before vs After Zoom (Frame {frame_idx})",
        save_path=frame_dir / "08_after_zoom_comparison.png"
    )
    
    visualize_image_details(
        img_after_zoom, 
        f"After Zoom (Frame {frame_idx})",
        save_path=frame_dir / "09_after_zoom_details.png"
    )
    
    # Resize image
    if img_after_zoom.shape[0] > 0 and img_after_zoom.shape[1] > 0:
        try:
            final_img = cv2.resize(img_after_zoom, res, interpolation=interpolation)
            logger.info(f"After resize: shape = {final_img.shape}")
        except Exception as e:
            logger.error(f"Error during resize: {str(e)}")
            logger.error(f"Attempted to resize image of shape {img_after_zoom.shape} to {res}")
            return None
    else:
        logger.error(f"Invalid image dimensions after cropping: {img_after_zoom.shape}")
        return None
    
    # Visualize final image
    visualize_comparison(
        img, 
        final_img, 
        f"Original vs Final (Frame {frame_idx})",
        save_path=frame_dir / "10_original_vs_final_comparison.png"
    )
    
    visualize_image_details(
        final_img, 
        f"Final Image (Frame {frame_idx})",
        save_path=frame_dir / "11_final_details.png"
    )
    
    return final_img

def process_dicom_file(file_path, max_frames=3):
    """Process a DICOM file with visual debugging."""
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
        
        # Process a limited number of frames
        for i in range(min(max_frames, len(pixels))):
            logger.info(f"Processing frame {i} of {len(pixels)}")
            
            try:
                # Process the frame
                result = crop_and_scale_visual(pixels[i], i)
                if result is not None:
                    logger.info(f"Successfully processed frame {i}")
                    # Save the result
                    cv2.imwrite(str(DEBUG_DIR / f"frame_{i}_result.png"), result)
                else:
                    logger.error(f"Failed to process frame {i}")
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info("Completed processing DICOM file")
        
    except Exception as e:
        logger.error(f"Error processing DICOM file: {str(e)}")
        logger.error(traceback.format_exc())

def main(file_path=None):
    """
    Main function to run the visual debugging script.
    
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
    
    logger.info(f"Starting visual debug script for file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    # Process the file
    process_dicom_file(file_path)
    
    logger.info("Visual debug script completed")
    print(f"Debug images saved to {DEBUG_DIR}")
    print(f"Debug log saved to debug_scaling_visual.log")

if __name__ == "__main__":
    main()
