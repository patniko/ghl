import torch
import torch.nn.functional as F
import glob
import numpy as np
from tqdm import tqdm
import cv2
import pydicom
import os
import shutil
import argparse
import json
from torchvision.models.video import r2plus1d_18
from pathlib import Path

# Flags for configuration
SAVE_MASK_IMAGES = True  # Set to False to disable saving before/after masking images

# Default configuration
DEFAULT_DATA_PATH = './data/example_study'
MODEL_WEIGHTS = "./weights/video_quality_model.pt"
FRAMES_TO_TAKE = 32
FRAME_STRIDE = 2
VIDEO_SIZE = 112
QUALITY_THRESHOLD = 0.3  # The threshold used for pass/fail

# Initialize model
device = torch.device("cpu")
video_classification_model = r2plus1d_18(num_classes=1)
weights = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
video_classification_model.load_state_dict(weights)
video_classification_model.eval()

# Normalization parameters
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)


def crop_and_scale(img, res=(112, 112), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """
    Crop and scale an image to the specified resolution.
    
    Args:
        img (np.ndarray): Input image
        res (tuple): Target resolution (width, height)
        interpolation: OpenCV interpolation method
        zoom (float): Zoom factor
        
    Returns:
        np.ndarray: Cropped and scaled image
    """
    # Check if image is valid
    if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Invalid image: empty or zero dimensions")
    
    # Get input resolution
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    
    # Crop to match aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        # Ensure padding doesn't exceed image dimensions
        if padding >= in_res[0] // 2:
            padding = max(0, in_res[0] // 2 - 1)
        img = img[:, padding:in_res[0]-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        # Ensure padding doesn't exceed image dimensions
        if padding >= in_res[1] // 2:
            padding = max(0, in_res[1] // 2 - 1)
        img = img[padding:in_res[1]-padding, :]
    
    # Apply zoom
    if zoom != 0 and img.shape[0] > 2 and img.shape[1] > 2:
        pad_x = max(1, round(int(img.shape[1] * zoom)))
        pad_y = max(1, round(int(img.shape[0] * zoom)))
        if pad_x < img.shape[1] // 2 and pad_y < img.shape[0] // 2:
            img = img[pad_y:-pad_y, pad_x:-pad_x]
    
    # Resize image
    if img.shape[0] > 0 and img.shape[1] > 0:
        img = cv2.resize(img, res, interpolation=interpolation)
    else:
        raise ValueError(f"Invalid image dimensions after cropping: {img.shape}")
    
    return img


def save_frame_image(frame, directory, filename):
    """
    Save a video frame as an image file.
    
    Args:
        frame (np.ndarray): The frame to save
        directory (str): Directory to save the image in
        filename (str): Filename for the image
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save the image
    cv2.imwrite(os.path.join(directory, filename), frame)


def mask_outside_ultrasound(original_pixels: np.array, dicom_filename=None) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC
    dicom_filename (str, optional): Filename of the DICOM file for saving images

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        test_array=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        
        # Save truly original frames without any color conversion
        if SAVE_MASK_IMAGES and dicom_filename:
            # Save first, middle, and last frames of original video without any processing
            frames_to_save = [0, len(original_pixels)//2, -1]
            for i, frame_idx in enumerate(frames_to_save):
                if frame_idx == -1 and len(original_pixels) > 0:
                    frame_idx = len(original_pixels) - 1
                
                if 0 <= frame_idx < len(original_pixels):
                    # Save the raw frame without any color conversion
                    frame_original = original_pixels[frame_idx].astype('uint8')
                    save_frame_image(
                        frame_original, 
                        './results/mask_images/original', 
                        f"{dicom_filename.replace('.dcm', '')}_{i}.png"
                    )
        
        # Save color-converted frames (current "before" images)
        if SAVE_MASK_IMAGES and dicom_filename:
            # Save first, middle, and last frames of original video with YUV to BGR conversion
            frames_to_save = [0, len(original_pixels)//2, -1]
            for i, frame_idx in enumerate(frames_to_save):
                if frame_idx == -1 and len(original_pixels) > 0:
                    frame_idx = len(original_pixels) - 1
                
                if 0 <= frame_idx < len(original_pixels):
                    frame = original_pixels[frame_idx].astype('uint8')
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
                    save_frame_image(
                        frame, 
                        './results/mask_images/before', 
                        f"{dicom_filename.replace('.dcm', '')}_{i}.png"
                    )
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = test_array[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = test_array.shape[0]
        for i in range(frames): # Go through every frame
            frame = test_array[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = test_array[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = test_array[test_array.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        
        # Save masked frames if enabled
        if SAVE_MASK_IMAGES and dicom_filename:
            # Save first, middle, and last frames of masked video
            frames_to_save = [0, len(vid)//2, -1]
            for i, frame_idx in enumerate(frames_to_save):
                if frame_idx == -1 and len(vid) > 0:
                    frame_idx = len(vid) - 1
                
                if 0 <= frame_idx < len(vid):
                    frame = vid[frame_idx].astype('uint8')
                    save_frame_image(
                        frame, 
                        './results/mask_images/after', 
                        f"{dicom_filename.replace('.dcm', '')}_{i}.png"
                    )
        
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid


def process_dicoms(input_path):
    """
    Reads DICOM video data from the specified folder and returns a tensor
    formatted for input into the EchoPrime model.
    Args:
        input_path (str): Path to the folder containing DICOM files.
    Returns:
        tuple: (stack_of_videos, dicom_paths, error_stats) where:
            - stack_of_videos (torch.Tensor): A float tensor representing the video data
            - dicom_paths (list): List of paths to the processed DICOM files
            - error_stats (dict): Statistics about errors encountered during processing
    """
    dicom_paths = glob.glob(f'{input_path}/**/*', recursive=True)
    # Filter to keep only valid files
    valid_dicom_paths = []
    stack_of_videos = []
    
    # Track error statistics
    error_stats = {
        "total_files": len(dicom_paths),
        "processed_files": 0,
        "successful_files": 0,
        "error_counts": {
            "not_dicom": 0,
            "empty_pixel_array": 0,
            "invalid_dimensions": 0,
            "masking_error": 0,
            "scaling_error": 0,
            "other_errors": 0
        },
        "error_files": {
            "not_dicom": [],
            "empty_pixel_array": [],
            "invalid_dimensions": [],
            "masking_error": [],
            "scaling_error": [],
            "other_errors": []
        }
    }
    
    for idx, dicom_path in tqdm(enumerate(dicom_paths), total=len(dicom_paths)):
        try:
            # simple dicom_processing
            try:
                dcm = pydicom.dcmread(dicom_path)
                error_stats["processed_files"] += 1
            except Exception as e:
                error_stats["error_counts"]["not_dicom"] += 1
                error_stats["error_files"]["not_dicom"].append(dicom_path)
                continue
            
            try:
                pixels = dcm.pixel_array
            except Exception as e:
                error_stats["error_counts"]["empty_pixel_array"] += 1
                error_stats["error_files"]["empty_pixel_array"].append(dicom_path)
                print(f"Error reading pixel array: {dicom_path} - {str(e)}")
                continue
            
            # exclude images like (600,800) or (600,800,3)
            if pixels.ndim < 3: # or pixels.shape[2] == 3:
                error_stats["error_counts"]["invalid_dimensions"] += 1
                error_stats["error_files"]["invalid_dimensions"].append(dicom_path)
                print(f"Excluding image with invalid dimensions: {dicom_path}")
                continue
                
            # if single channel repeat to 3 channels
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
                
            # Check if pixel array is valid before processing
            if dcm.pixel_array.size == 0 or dcm.pixel_array.shape[0] == 0:
                error_stats["error_counts"]["empty_pixel_array"] += 1
                error_stats["error_files"]["empty_pixel_array"].append(dicom_path)
                print(f"Skipping file with empty pixel array: {dicom_path}")
                continue
                
            # mask everything outside ultrasound region
            filename = os.path.basename(dicom_path)
            try:
                pixels = mask_outside_ultrasound(dcm.pixel_array, filename)
                
                # Verify pixels are not empty after masking
                if pixels is None or len(pixels) == 0 or pixels.size == 0:
                    error_stats["error_counts"]["masking_error"] += 1
                    error_stats["error_files"]["masking_error"].append(dicom_path)
                    print(f"Skipping file with invalid pixels after masking: {dicom_path}")
                    continue
            except Exception as e:
                error_stats["error_counts"]["masking_error"] += 1
                error_stats["error_files"]["masking_error"].append(dicom_path)
                print(f"Error during masking: {dicom_path} - {str(e)}")
                continue
                
            # model specific preprocessing
            x = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3))
            valid_frames = True
            for i in range(len(x)):
                # Check if the frame is valid before scaling
                if pixels[i].size == 0 or pixels[i].shape[0] == 0 or pixels[i].shape[1] == 0:
                    error_stats["error_counts"]["invalid_dimensions"] += 1
                    error_stats["error_files"]["invalid_dimensions"].append(f"{dicom_path} (frame {i})")
                    print(f"Skipping file with invalid frame at index {i}: {dicom_path}")
                    valid_frames = False
                    break
                try:
                    x[i] = crop_and_scale(pixels[i])
                except Exception as e:
                    error_stats["error_counts"]["scaling_error"] += 1
                    error_stats["error_files"]["scaling_error"].append(f"{dicom_path} (frame {i})")
                    print(f"Error scaling frame {i} in {dicom_path}: {str(e)}")
                    valid_frames = False
                    break
            
            # Skip if any frame processing failed
            if not valid_frames:
                continue
            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            
            # normalize
            x.sub_(mean).div_(std)
            
            ## if not enough frames add padding
            if x.shape[1] < FRAMES_TO_TAKE:
                padding = torch.zeros(
                    (
                        3,
                        FRAMES_TO_TAKE - x.shape[1],
                        VIDEO_SIZE,
                        VIDEO_SIZE,
                    ),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)
            start = 0
            stack_of_videos.append(x[:, start: (start + FRAMES_TO_TAKE): FRAME_STRIDE, :, :])
            valid_dicom_paths.append(dicom_path)
            error_stats["successful_files"] += 1
        except Exception as e:
            error_stats["error_counts"]["other_errors"] += 1
            error_stats["error_files"]["other_errors"].append(dicom_path)
            print(f"Corrupt file or unexpected error: {dicom_path}")
            print(str(e))
    
    if not stack_of_videos:
        print(f"No valid DICOM files found in {input_path}")
        return None, [], error_stats
        
    stack_of_videos = torch.stack(stack_of_videos)
    return stack_of_videos, valid_dicom_paths, error_stats

def get_quality_issues(probability):
    """
    Provides a basic assessment of potential quality issues based on probability score.
    
    Args:
        probability (float): The quality probability score from the model.
        
    Returns:
        str: Description of potential quality issues.
    """
    if probability >= 0.8:
        return "Excellent quality"
    elif probability >= 0.6:
        return "Good quality"
    elif probability >= 0.3:
        return "Acceptable quality, but may have minor issues"
    elif probability >= 0.2:
        return "Poor quality - likely issues with clarity, contrast, or positioning"
    elif probability >= 0.1:
        return "Very poor quality - significant issues with image acquisition"
    else:
        return "Critical issues - may include artifacts, improper view, or technical errors"

def clear_mask_images_directory():
    """
    Clear the mask_images directory to ensure fresh images for each run.
    Creates the directory structure if it doesn't exist.
    """
    if SAVE_MASK_IMAGES:
        # Create or clear the mask_images directory and its subdirectories
        mask_dir = './results/mask_images'
        original_dir = os.path.join(mask_dir, 'original')
        before_dir = os.path.join(mask_dir, 'before')
        after_dir = os.path.join(mask_dir, 'after')
        
        # Remove existing directories if they exist
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)
        
        # Create fresh directories
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        print(f"Cleared and created mask image directories at {mask_dir}")


def process_device_folder(device_folder):
    """
    Process a single device folder and return quality assessment results.
    
    Args:
        device_folder (str): Path to the device folder
        
    Returns:
        dict: Results containing quality metrics for this device
    """
    print(f"\nProcessing device folder: {device_folder}")
    
    # Process the DICOM files in this folder
    stack_of_videos, dicom_paths, error_stats = process_dicoms(device_folder)
    
    if stack_of_videos is None or len(dicom_paths) == 0:
        return {
            "device_name": os.path.basename(device_folder),
            "total_files": error_stats["total_files"],
            "processed_files": error_stats["processed_files"],
            "successful_files": 0,
            "pass_count": 0,
            "pass_rate": 0,
            "average_quality_score": 0,
            "files": [],
            "error_stats": error_stats
        }
    
    # Get the filenames for reference
    filenames = [os.path.basename(path) for path in dicom_paths]
    
    # Run the model
    with torch.no_grad():
        logits = video_classification_model(stack_of_videos)
    
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= QUALITY_THRESHOLD).float()
    
    # Print results for this device
    device_name = os.path.basename(device_folder)
    print(f"\nQuality Assessment Results for {device_name}:")
    print("=" * 80)
    print(f"{'Filename':<60} {'Score':<10} {'Pass/Fail':<10} {'Assessment'}")
    print("-" * 80)
    
    # Create a list to store individual file results
    file_results = []
    
    for i, (filename, prob, pred) in enumerate(zip(filenames, probabilities, predictions)):
        prob_value = prob.item()
        status = "PASS" if pred.item() > 0 else "FAIL"
        assessment = get_quality_issues(prob_value)
        
        # Store results
        file_result = {
            "name": filename,
            "path": dicom_paths[i],
            "score": prob_value,
            "status": status,
            "assessment": assessment
        }
        file_results.append(file_result)
        
        # Truncate filename if too long for display
        short_filename = filename[:57] + "..." if len(filename) > 60 else filename.ljust(60)
        print(f"{short_filename} {prob_value:.4f}    {status:<10} {assessment}")
    
    # Summary statistics for this device
    pass_count = predictions.sum().item()
    total_count = len(predictions)
    pass_rate = (pass_count/total_count*100) if total_count > 0 else 0
    avg_score = probabilities.mean().item() if total_count > 0 else 0
    
    print(f"\nSummary for {device_name}: {pass_count}/{total_count} videos passed quality check ({pass_rate:.1f}%)")
    
    # Return results for this device
    return {
        "device_name": device_name,
        "total_files": error_stats["total_files"],
        "processed_files": error_stats["processed_files"],
        "successful_files": error_stats["successful_files"],
        "pass_count": int(pass_count),
        "pass_rate": pass_rate,
        "average_quality_score": avg_score,
        "files": file_results,
        "error_stats": error_stats
    }

def save_results_to_json(all_results, output_file="quality_results.json"):
    """Save all results to a JSON file"""
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

def save_failed_files_to_json(device_results):
    """
    Save failed files information to a JSON file per device.
    
    Args:
        device_results (dict): Results for a specific device including error statistics
    """
    # Create directory for failed files if it doesn't exist
    failed_files_dir = './results/failed_files'
    os.makedirs(failed_files_dir, exist_ok=True)
    
    device_name = device_results["device_name"]
    error_stats = device_results["error_stats"]
    
    # Create a structured dictionary of failed files with reasons
    failed_files = {}
    for error_type, file_list in error_stats["error_files"].items():
        for file_path in file_list:
            # Use the filename as the key
            filename = os.path.basename(file_path)
            # If the file path contains a frame index (for frame-specific errors)
            if " (frame " in file_path:
                base_path, frame_info = file_path.split(" (frame ", 1)
                filename = os.path.basename(base_path)
                frame_num = frame_info.rstrip(")")
                error_reason = f"{error_type} in frame {frame_num}"
            else:
                error_reason = error_type
            
            # Add to the failed files dictionary
            if filename not in failed_files:
                failed_files[filename] = {
                    "path": file_path.split(" (frame ")[0] if " (frame " in file_path else file_path,
                    "reasons": [error_reason]
                }
            else:
                # If this file already has other errors, add this reason
                if error_reason not in failed_files[filename]["reasons"]:
                    failed_files[filename]["reasons"].append(error_reason)
    
    # Convert to a list format for easier processing
    failed_files_list = [
        {
            "filename": filename,
            "path": info["path"],
            "reasons": info["reasons"]
        }
        for filename, info in failed_files.items()
    ]
    
    # Save to a JSON file named after the device
    output_file = os.path.join(failed_files_dir, f"{device_name}_failed_files.json")
    with open(output_file, "w") as f:
        json.dump({
            "device": device_name,
            "total_failed_files": len(failed_files_list),
            "failed_files": failed_files_list
        }, f, indent=2)
    
    print(f"Failed files for {device_name} saved to {output_file}")

def run_quality_assessment(device_folders):
    """
    Run quality assessment on multiple device folders and generate a summary.
    
    Args:
        device_folders (list): List of paths to device folders
    """
    # Clear mask images directory if saving is enabled
    clear_mask_images_directory()
    
    # Dictionary to store all results
    all_results = {
        "summary": {
            "total_devices": len(device_folders),
            "total_files": 0,
            "processed_files": 0,
            "successful_files": 0,
            "total_pass": 0,
            "overall_pass_rate": 0,
            "average_quality_score": 0,
            "error_summary": {
                "not_dicom": 0,
                "empty_pixel_array": 0,
                "invalid_dimensions": 0,
                "masking_error": 0,
                "scaling_error": 0,
                "other_errors": 0
            }
        },
        "devices": []
    }
    
    # Process each device folder
    for device_folder in device_folders:
        device_results = process_device_folder(device_folder)
        all_results["devices"].append(device_results)
        
        # Update summary statistics
        all_results["summary"]["total_files"] += device_results["total_files"]
        all_results["summary"]["processed_files"] += device_results["processed_files"]
        all_results["summary"]["successful_files"] += device_results["successful_files"]
        all_results["summary"]["total_pass"] += device_results["pass_count"]
        
        # Update error summary
        for error_type, count in device_results["error_stats"]["error_counts"].items():
            all_results["summary"]["error_summary"][error_type] += count
    
    # Calculate overall statistics
    if all_results["summary"]["total_files"] > 0:
        all_results["summary"]["overall_pass_rate"] = (
            all_results["summary"]["total_pass"] / all_results["summary"]["total_files"] * 100
        )
        
        # Calculate average quality score across all devices
        total_score = sum(d["average_quality_score"] * d["total_files"] for d in all_results["devices"])
        all_results["summary"]["average_quality_score"] = (
            total_score / all_results["summary"]["total_files"] if all_results["summary"]["total_files"] > 0 else 0
        )
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY ACROSS ALL DEVICES")
    print("=" * 80)
    print(f"Total devices processed: {all_results['summary']['total_devices']}")
    print(f"Total files found: {all_results['summary']['total_files']}")
    print(f"Total files processed as DICOM: {all_results['summary']['processed_files']}")
    print(f"Total files successfully processed: {all_results['summary']['successful_files']}")
    print(f"Total files passed quality check: {all_results['summary']['total_pass']}")
    print(f"Overall pass rate: {all_results['summary']['overall_pass_rate']:.1f}%")
    print(f"Average quality score: {all_results['summary']['average_quality_score']:.4f}")
    
    # Print error summary
    print("\nERROR SUMMARY:")
    print("-" * 80)
    for error_type, count in all_results['summary']['error_summary'].items():
        print(f"{error_type.replace('_', ' ').title()}: {count}")
    
    print("\nDevice-specific results:")
    
    # Print a table of device results
    if all_results["devices"]:
        print(f"{'Device':<20} {'Files':<10} {'Passed':<10} {'Pass Rate':<15} {'Avg Score':<15}")
        print("-" * 70)
        for device in all_results["devices"]:
            print(f"{device['device_name']:<20} {device['total_files']:<10} {device['pass_count']:<10} "
                  f"{device['pass_rate']:.1f}%{' ':<10} {device['average_quality_score']:.4f}")
    
    # Save all results to JSON
    save_results_to_json(all_results)
    
    # Save failed files for each device to separate JSON files
    for device_results in all_results["devices"]:
        save_failed_files_to_json(device_results)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EchoPrime Quality Control for multiple device folders')
    parser.add_argument('--folders', nargs='+', help='List of device folders to process')
    parser.add_argument('--study_data', action='store_true', help='Process all device folders in data/')
    parser.add_argument('--no-mask-images', action='store_true', help='Disable saving mask images')
    
    args = parser.parse_args()
    
    # If no folders specified and --study_data not used, use the default path
    if not args.folders and not args.study_data:
        return [DEFAULT_DATA_PATH]
    
    # If --study_data flag is used, get all subdirectories in data/
    if args.study_data:
        study_data_path = './data'
        if os.path.exists(study_data_path):
            return [os.path.join(study_data_path, d) for d in os.listdir(study_data_path) 
                   if os.path.isdir(os.path.join(study_data_path, d))]
        else:
            print(f"Error: {study_data_path} directory not found")
            return [DEFAULT_DATA_PATH]
    
    return args.folders

if __name__ == "__main__":
    # Parse command line arguments
    device_folders = parse_arguments()
    
    # Run quality assessment on all device folders
    run_quality_assessment(device_folders)
