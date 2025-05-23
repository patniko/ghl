#!/usr/bin/env python
"""
Ultrasound masking utilities for echocardiogram videos.
This module provides functions for masking non-ultrasound regions in echocardiogram
videos to focus the analysis on the relevant ultrasound data.
"""

import numpy as np
import cv2


def downsample_and_crop(testarray):
    """
    Downsample and crop video frames to center and square the ultrasound region.
    
    Args:
        testarray (np.ndarray): Video frames array of shape (frames, height, width, channels)
        
    Returns:
        np.ndarray: Processed video frames or None if processing fails
    """
    ##################### CREATE MASK #####################
    # Sum all the frames
    frame_sum = testarray[0] # Start off the frameSum with the first frame<<
    # Convert color profile b/c cv2 messes up colors when it reads it in
    frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_BGR2GRAY)
    original = frame_sum
    frame_sum = np.where(frame_sum>0,1,0) # make all non-zero values 1
    frames = testarray.shape[0]
    for i in range(frames): # Go through every frame
        frame = testarray[i, :, :, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.where(frame > 0, 1, 0) # make all non-zero values 1
        frame_sum = np.add(frame_sum, frame)
    # Dilate
    kernel = np.ones((3,3), np.uint8)
    frame_sum = cv2.dilate(np.uint8(frame_sum), kernel, iterations=10)
    # Make binary
    frame_overlap = np.where(frame_sum>0,1,0)                

    ###### Center and Square both Mask and Video ########        
    # Center image by finding center x of the image
    # Pick first 300 y-values
    center = frame_overlap[0:300, :]
    # compress along y axis
    center = np.mean(center, axis=0)
    try:
        center = np.where(center > 0, 1, 0) # make binary
    except:
        return
    # find index where first goes from 0 to 1 and goes from 1 to 0
    try:
        indexL = np.where(center>0)[0][0]
        indexR = center.shape[0]-np.where(np.flip(center)>0)[0][0]
        center_index = int((indexL + indexR) / 2)
    except:
        return
    # Cut off x on one side so that it's centered on x axis
    left_margin = center_index
    right_margin = center.shape[0] - center_index
    if left_margin > right_margin:
        frame_overlap = frame_overlap[:, (left_margin - right_margin):]
        testarray = testarray[:, :, (left_margin - right_margin):, :]
    else:
        frame_overlap = frame_overlap[: , :(center_index + left_margin)]
        testarray = testarray[:, :, :(center_index + left_margin), :]   

    #Make image square by cutting
    height = frame_overlap.shape[0]
    width = frame_overlap.shape[1]
    #Trim by 1 pixel if a dimension has an odd number of pixels
    if (height % 2) != 0:
        frame_overlap = frame_overlap[0:height - 1, :]
        testarray = testarray[:, 0:height - 1, :, :]
    if (width % 2) != 0:
        frame_overlap = frame_overlap[:, 0:width - 1]
        testarray = testarray[:, :, 0:width - 1, :]
    height = frame_overlap.shape[0]
    width = frame_overlap.shape[1]
    bias = int(abs(height - width) / 2)
    if height > width:
        frame_overlap = frame_overlap[bias:height-bias, :]
        testarray = testarray[:, bias:height-bias, :, :]
    else:
        frame_overlap = frame_overlap[:,bias:width-bias]
        testarray = testarray[:, :, bias:width-bias, :]
    return testarray


def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    original_pixels (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
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
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
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
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid
