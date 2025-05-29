#!/usr/bin/env python
"""
Ultrasound masking utilities for echocardiogram videos.
This module provides functions for masking non-ultrasound regions in echocardiogram
videos to focus the analysis on the relevant ultrasound data.
"""

import os
import numpy as np
import cv2

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

def mask_outside_ultrasound(original_pixels, dicom_filename=None, save_mask_images=False):
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    original_pixels (np.ndarray): A numpy array representing the video frames. FxHxWxC
    dicom_filename (str, optional): Filename of the DICOM file for saving images
    save_mask_images (bool): Whether to save mask images for debugging

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        test_array = np.copy(original_pixels)
        vid = np.copy(original_pixels)
        
        # Save truly original frames without any color conversion
        if save_mask_images and dicom_filename:
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
        if save_mask_images and dicom_filename:
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
        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        
        # Save masked frames if enabled
        if save_mask_images and dicom_filename:
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
        return vid
