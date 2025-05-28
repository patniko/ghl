#!/usr/bin/env python
"""
Image scaling utilities for echocardiogram videos.
This module provides functions for cropping and scaling images to prepare them
for input into the EchoPrime model.
"""

import cv2
import numpy as np
import torch


def apply_zoom(img_batch, zoom=0.1):
    """
    Apply zoom on a batch of images using PyTorch.
    
    Parameters:
        img_batch (torch.Tensor): A batch of images of shape (batch_size, height, width, channels).
        zoom (float): The zoom factor to apply, default is 0.1 (i.e., crop 10% from each side).
        
    Returns:
        torch.Tensor: A batch of zoomed images.
    """
    batch_size, height, width, channels = img_batch.shape

    # Calculate padding for zoom
    pad_x = round(int(width * zoom))  # X-axis (width)
    pad_y = round(int(height * zoom)) # Y-axis (height)

    # Crop the images by the zoom factor
    img_zoomed = img_batch[:, pad_y:-pad_y, pad_x:-pad_x, :]

    return img_zoomed


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
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
