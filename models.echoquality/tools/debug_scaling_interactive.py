#!/usr/bin/env python
"""
Interactive debugging script for investigating image scaling issues in DICOM files.
This script provides a step-by-step interactive approach to debug the crop_and_scale function.
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
import pdb  # Python debugger for interactive debugging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_scaling_interactive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_scaling_interactive")

# Create output directory for debug images
DEBUG_DIR = Path("./results/debug_images_interactive")
DEBUG_DIR.mkdir(exist_ok=True)

def display_image(img, title="Image"):
    """Display an image using matplotlib for interactive debugging."""
    plt.figure(figsize=(10, 8))
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    
    plt.title(f"{title}\nShape: {img.shape}, Type: {img.dtype}")
    plt.colorbar()
    plt.show()

def crop_and_scale_interactive(img, frame_idx, res=(112, 112), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """
    Interactive debug version of crop_and_scale function.
    
    Args:
        img (np.ndarray): Input image
        frame_idx (int): Frame index for logging
        res (tuple): Target resolution (width, height)
        interpolation: OpenCV interpolation method
        zoom (float): Zoom factor
        
    Returns:
        np.ndarray: Cropped and scaled image
    """
    print("\n" + "="*80)
    print(f"INTERACTIVE DEBUGGING - FRAME {frame_idx}")
    print("="*80)
    
    print(f"Original image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Min value: {np.min(img)}, Max value: {np.max(img)}, Mean value: {np.mean(img):.2f}")
    
    # Display original image
    display_image(img, f"Original Image (Frame {frame_idx})")
    
    # Check if image is valid
    if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        print("ERROR: Invalid image dimensions")
        return None
    
    # Get input resolution
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    
    print(f"Input resolution: {in_res}, aspect ratio: {r_in:.4f}")
    print(f"Output resolution: {res}, aspect ratio: {r_out:.4f}")
    
    # Interactive breakpoint - examine the image before cropping
    print("\nExamining image before aspect ratio cropping...")
    print("Type 'c' to continue or 'q' to quit the debugger")
    pdb.set_trace()
    
    # Crop to match aspect ratio
    img_after_ratio_crop = img.copy()
    
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        print(f"r_in > r_out: Cropping width with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= in_res[0] // 2:
            old_padding = padding
            padding = max(0, in_res[0] // 2 - 1)
            print(f"WARNING: Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
        
        # Check if the resulting slice is valid
        if padding >= in_res[0]:
            print(f"ERROR: Invalid padding: {padding} >= image width {in_res[0]}")
            return None
        
        try:
            img_after_ratio_crop = img[:, padding:in_res[0]-padding]
            print(f"After width crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            print(f"ERROR during width cropping: {str(e)}")
            print(f"Attempted slice: img[:, {padding}:{in_res[0]-padding}]")
            return None
    
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        print(f"r_in < r_out: Cropping height with padding {padding} pixels")
        
        # Check if padding is valid
        if padding >= in_res[1] // 2:
            old_padding = padding
            padding = max(0, in_res[1] // 2 - 1)
            print(f"WARNING: Adjusted padding from {old_padding} to {padding} to avoid exceeding image dimensions")
        
        # Check if the resulting slice is valid
        if padding >= in_res[1]:
            print(f"ERROR: Invalid padding: {padding} >= image height {in_res[1]}")
            return None
        
        try:
            img_after_ratio_crop = img[padding:in_res[1]-padding, :]
            print(f"After height crop: shape = {img_after_ratio_crop.shape}")
        except Exception as e:
            print(f"ERROR during height cropping: {str(e)}")
            print(f"Attempted slice: img[{padding}:{in_res[1]-padding}, :]")
            return None
    
    # Display image after aspect ratio cropping
    display_image(img_after_ratio_crop, f"After Aspect Ratio Crop (Frame {frame_idx})")
    
    # Interactive breakpoint - examine the image after cropping
    print("\nExamining image after aspect ratio cropping...")
    print("Type 'c' to continue or 'q' to quit the debugger")
    pdb.set_trace()
    
    # Apply zoom
    img_after_zoom = img_after_ratio_crop.copy()
    
    if zoom != 0 and img_after_ratio_crop.shape[0] > 2 and img_after_ratio_crop.shape[1] > 2:
        pad_x = max(1, round(int(img_after_ratio_crop.shape[1] * zoom)))
        pad_y = max(1, round(int(img_after_ratio_crop.shape[0] * zoom)))
        
        print(f"Applying zoom with pad_x={pad_x}, pad_y={pad_y}")
        
        # Check if zoom parameters are valid
        if pad_x >= img_after_ratio_crop.shape[1] // 2 or pad_y >= img_after_ratio_crop.shape[0] // 2:
            print(f"WARNING: Zoom padding too large for image size, skipping zoom")
        else:
            try:
                img_after_zoom = img_after_ratio_crop[pad_y:-pad_y, pad_x:-pad_x]
                print(f"After zoom: shape = {img_after_zoom.shape}")
            except Exception as e:
                print(f"ERROR during zoom: {str(e)}")
                print(f"Attempted slice: img[{pad_y}:-{pad_y}, {pad_x}:-{pad_x}]")
                # Continue with unzoomed image
                img_after_zoom = img_after_ratio_crop
    
    # Display image after zoom
    display_image(img_after_zoom, f"After Zoom (Frame {frame_idx})")
    
    # Interactive breakpoint - examine the image after zoom
    print("\nExamining image after zoom...")
    print("Type 'c' to continue or 'q' to quit the debugger")
    pdb.set_trace()
    
    # Resize image
    if img_after_zoom.shape[0] > 0 and img_after_zoom.shape[1] > 0:
        try:
            final_img = cv2.resize(img_after_zoom, res, interpolation=interpolation)
            print(f"After resize: shape = {final_img.shape}")
        except Exception as e:
            print(f"ERROR during resize: {str(e)}")
            print(f"Attempted to resize image of shape {img_after_zoom.shape} to {res}")
            return None
    else:
        print(f"ERROR: Invalid image dimensions after cropping: {img_after_zoom.shape}")
        return None
    
    # Display final image
    display_image(final_img, f"Final Image (Frame {frame_idx})")
    
    # Interactive breakpoint - examine the final image
    print("\nExamining final image...")
    print("Type 'c' to continue or 'q' to quit the debugger")
    pdb.set_trace()
    
    return final_img

def process_frame_interactively(pixels, frame_idx):
    """Process a single frame interactively."""
    print(f"\nProcessing frame {frame_idx} of {len(pixels)}")
    
    try:
        # Process the frame
        result = crop_and_scale_interactive(pixels[frame_idx], frame_idx)
        if result is not None:
            print(f"Successfully processed frame {frame_idx}")
            # Save the result
            cv2.imwrite(str(DEBUG_DIR / f"frame_{frame_idx}_result.png"), result)
        else:
            print(f"Failed to process frame {frame_idx}")
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        traceback.print_exc()

def main(file_path=None):
    """
    Main function to run the interactive debugging script.
    
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
    
    print(f"Starting interactive debug script for file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(file_path)
        print(f"Successfully read DICOM file")
        
        # Get pixel array
        try:
            pixels = dcm.pixel_array
            print(f"Pixel array shape: {pixels.shape}, dtype: {pixels.dtype}")
        except Exception as e:
            print(f"Error reading pixel array: {str(e)}")
            return
        
        # Check dimensions
        if pixels.ndim < 3:
            print(f"Invalid dimensions: {pixels.ndim} (expected >= 3)")
            return
        
        # If single channel, repeat to 3 channels
        if pixels.ndim == 3:
            print("Single channel image, repeating to 3 channels")
            try:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
                print(f"After channel repeat: shape = {pixels.shape}")
            except Exception as e:
                print(f"Error repeating channels: {str(e)}")
                return
        
        # Ask which frame to process
        num_frames = len(pixels)
        print(f"\nThe DICOM file contains {num_frames} frames.")
        
        while True:
            try:
                frame_input = input(f"Enter frame number to process (0-{num_frames-1}) or 'q' to quit: ")
                if frame_input.lower() == 'q':
                    break
                
                frame_idx = int(frame_input)
                if 0 <= frame_idx < num_frames:
                    process_frame_interactively(pixels, frame_idx)
                else:
                    print(f"Invalid frame number. Please enter a number between 0 and {num_frames-1}.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
            except Exception as e:
                print(f"Error: {str(e)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error processing DICOM file: {str(e)}")
        traceback.print_exc()
    
    print("\nInteractive debug script completed")
    print(f"Debug images saved to {DEBUG_DIR}")

if __name__ == "__main__":
    main()
