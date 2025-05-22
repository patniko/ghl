"""
Preprocessors for echocardiogram videos.

This package provides utilities for preprocessing echocardiogram videos,
including image scaling, masking, and other transformations needed for
quality assessment and analysis.
"""

from .image_scaling import crop_and_scale
from .ultrasound_masking import mask_outside_ultrasound, save_frame_image

__all__ = [
    'crop_and_scale',
    'mask_outside_ultrasound',
    'save_frame_image'
]
