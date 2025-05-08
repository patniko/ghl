import torch
import torch.nn.functional as F
import glob
import numpy as np
from tqdm import tqdm
import cv2
import pydicom
import mlflow
import os
import shutil
from torchvision.models.video import r2plus1d_18

# Flags for configuration
USE_MLFLOW = True  # Set to False to disable MLflow tracking
SAVE_MASK_IMAGES = True  # Set to False to disable saving before/after masking images

# Set up MLflow tracking if enabled
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


device = torch.device("cpu")
video_classification_model = r2plus1d_18(num_classes=1)

data_path = './model_data/failures'
model_weights = "./video_quality_model.pt"
weights = torch.load(model_weights, map_location=torch.device('cpu'))
video_classification_model.load_state_dict(weights)

frames_to_take = 32
frame_stride = 2
video_size = 112
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)


def crop_and_scale(img, res=(112, 112), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]
    img = cv2.resize(img, res, interpolation=interpolation)
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
                        './mask_images/original', 
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
                        './mask_images/before', 
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
                        './mask_images/after', 
                        f"{dicom_filename.replace('.dcm', '')}_{i}.png"
                    )
        
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid


def process_dicoms(INPUT):
    """
    Reads DICOM video data from the specified folder and returns a tensor
    formatted for input into the EchoPrime model.
    Args:
        INPUT (str): Path to the folder containing DICOM files.
    Returns:
        stack_of_videos (torch.Tensor): A float tensor of shape  (N, 3, 16, 224, 224)
                                        representing the video data where N is the number of videos,
                                        ready to be fed into EchoPrime.
    """
    dicom_paths = glob.glob(f'{INPUT}/**/*.dcm', recursive=True)

    stack_of_videos = []
    for idx, dicom_path in tqdm(enumerate(dicom_paths), total=len(dicom_paths)):
        try:
            # simple dicom_processing
            dcm = pydicom.dcmread(dicom_path)

            print(dcm)
            pixels = dcm.pixel_array

            print("NDIM: " + str(pixels.ndim))
            print("SHAPE:" + str(pixels.shape[2]))

            # exclude images like (600,800) or (600,800,3)
            if pixels.ndim < 3: # or pixels.shape[2] == 3:
                print("Excluding images!")
                continue
                # if single channel repeat to 3 channels
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
            # mask everything outside ultrasound region
            filename = os.path.basename(dicom_path)
            pixels = mask_outside_ultrasound(dcm.pixel_array, filename)
            # model specific preprocessing
            x = np.zeros((len(pixels), 112, 112, 3))
            for i in range(len(x)):
                x[i] = crop_and_scale(pixels[i])
            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            # normalize
            x.sub_(mean).div_(std)
            ## if not enough frames add padding
            if x.shape[1] < frames_to_take:
                padding = torch.zeros(
                    (
                        3,
                        frames_to_take - x.shape[1],
                        video_size,
                        video_size,
                    ),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)
            start = 0
            stack_of_videos.append(x[:, start: (start + frames_to_take): frame_stride, :, :])
        except Exception as e:
            print("corrupt file")
            print(str(e))
    stack_of_videos = torch.stack(stack_of_videos)
    return stack_of_videos

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
        mask_dir = './mask_images'
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


if __name__ == "__main__":
    # Clear mask images directory if saving is enabled
    clear_mask_images_directory()
    
    # Process with or without MLflow based on the flag
    if USE_MLFLOW:
        # Start an MLflow run for tracking
        with mlflow.start_run(run_name="EchoPrime_QC_Assessment") as run:
            # Log parameters
            mlflow.log_params({
                "data_path": data_path,
                "model_weights": model_weights,
                "frames_to_take": frames_to_take,
                "frame_stride": frame_stride,
                "video_size": video_size,
                "quality_threshold": 0.3  # The threshold used for pass/fail
            })
            
            stack_of_videos = process_dicoms(data_path)
            video_classification_model.eval()
            
            # Get the filenames for reference
            dicom_paths = glob.glob(f'{data_path}/**/*.dcm', recursive=True)
            filenames = [path.split('/')[-1] for path in dicom_paths]
            
            logits = video_classification_model(stack_of_videos)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.3).float()
            
            print("\nQuality Assessment Results:")
            print("=" * 80)
            print(f"{'Filename':<60} {'Score':<10} {'Pass/Fail':<10} {'Assessment'}")
            print("-" * 80)
            
            # Create a dictionary to store individual file results for MLflow
            file_results = {}
            
            for i, (filename, prob, pred) in enumerate(zip(filenames, probabilities, predictions)):
                prob_value = prob.item()
                status = "PASS" if pred.item() > 0 else "FAIL"
                assessment = get_quality_issues(prob_value)
                
                # Store results for MLflow
                file_results[f"file_{i}_name"] = filename
                file_results[f"file_{i}_score"] = prob_value
                file_results[f"file_{i}_status"] = status
                file_results[f"file_{i}_assessment"] = assessment
                
                # Log individual file metrics
                mlflow.log_metric(f"score_{filename}", prob_value)
                mlflow.log_metric(f"pass_{filename}", 1 if status == "PASS" else 0)
                
                # Truncate filename if too long
                short_filename = filename[:57] + "..." if len(filename) > 60 else filename.ljust(60)
                
                print(f"{short_filename} {prob_value:.4f}    {status:<10} {assessment}")
            
            # Summary statistics
            pass_count = predictions.sum().item()
            total_count = len(predictions)
            pass_rate = pass_count/total_count*100
            
            print(f"\nSummary: {pass_count}/{total_count} videos passed quality check ({pass_rate:.1f}%)")
            
            # Log summary metrics
            mlflow.log_metrics({
                "total_files": total_count,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "average_quality_score": probabilities.mean().item()
            })
            
            # Log the detailed results as a JSON artifact
            import json
            with open("quality_results.json", "w") as f:
                json.dump(file_results, f, indent=2)
            mlflow.log_artifact("quality_results.json")
            
            print(f"\nResults logged to MLflow run: {run.info.run_id}")
            print(f"View at: {mlflow.get_tracking_uri()}/#/experiments/0/runs/{run.info.run_id}")
    else:
        # Run without MLflow tracking
        stack_of_videos = process_dicoms(data_path)
        video_classification_model.eval()
        
        # Get the filenames for reference
        dicom_paths = glob.glob(f'{data_path}/**/*.dcm', recursive=True)
        filenames = [path.split('/')[-1] for path in dicom_paths]
        
        logits = video_classification_model(stack_of_videos)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.3).float()
        
        print("\nQuality Assessment Results:")
        print("=" * 80)
        print(f"{'Filename':<60} {'Score':<10} {'Pass/Fail':<10} {'Assessment'}")
        print("-" * 80)
        
        for i, (filename, prob, pred) in enumerate(zip(filenames, probabilities, predictions)):
            prob_value = prob.item()
            status = "PASS" if pred.item() > 0 else "FAIL"
            assessment = get_quality_issues(prob_value)
            
            # Truncate filename if too long
            short_filename = filename[:57] + "..." if len(filename) > 60 else filename.ljust(60)
            
            print(f"{short_filename} {prob_value:.4f}    {status:<10} {assessment}")
        
        # Summary statistics
        pass_count = predictions.sum().item()
        total_count = len(predictions)
        pass_rate = pass_count/total_count*100
        
        print(f"\nSummary: {pass_count}/{total_count} videos passed quality check ({pass_rate:.1f}%)")
        
        # Save results to JSON without MLflow
        import json
        file_results = {}
        for i, (filename, prob, pred) in enumerate(zip(filenames, probabilities, predictions)):
            prob_value = prob.item()
            status = "PASS" if pred.item() > 0 else "FAIL"
            assessment = get_quality_issues(prob_value)
            file_results[f"file_{i}_name"] = filename
            file_results[f"file_{i}_score"] = prob_value
            file_results[f"file_{i}_status"] = status
            file_results[f"file_{i}_assessment"] = assessment
        
        with open("quality_results.json", "w") as f:
            json.dump(file_results, f, indent=2)
        print("\nResults saved to quality_results.json")
