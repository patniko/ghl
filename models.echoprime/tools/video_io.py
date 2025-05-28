#!/usr/bin/env python
"""
Video I/O utilities for echocardiogram videos.
This module provides functions for reading and writing video files,
including DICOM video processing and color space conversions.
"""

import torchvision
from pathlib import Path
import numpy as np
import cv2
import pydicom as dicom


_ybr_to_rgb_lut = None


def write_video(p: Path, pixels: np.ndarray, fps=30.0, codec='h264'):
    """
    Write video frames to a file using torchvision.
    
    Args:
        p (Path): Output file path
        pixels (np.ndarray): Video frames array
        fps (float): Frames per second
        codec (str): Video codec to use
    """
    torchvision.io.write_video(str(p), pixels, fps, codec)


def write_to_avi(frames: np.ndarray, out_file, fps=30):
    """
    Write video frames to an AVI file.
    
    Args:
        frames (np.ndarray): Video frames array
        out_file: Output file path
        fps (int): Frames per second
    """
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'MJPG'), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()


def write_image(p: Path, pixels: np.ndarray):
    """
    Write image to file.
    
    Args:
        p (Path): Output file path
        pixels (np.ndarray): Image array
    """
    cv2.imwrite(str(p), pixels)


def ybr_to_rgb(pixels: np.array):
    """
    Convert YBR color space to RGB using lookup table.
    
    Args:
        pixels (np.array): Input pixels in YBR format
        
    Returns:
        np.array: Pixels converted to RGB format
    """
    lut = get_ybr_to_rgb_lut()
    return lut[pixels[..., 0], pixels[..., 1], pixels[..., 2]]


def get_ybr_to_rgb_lut(save_lut=True):
    """
    Get or generate YBR to RGB lookup table.
    
    Args:
        save_lut (bool): Whether to save the generated LUT to file
        
    Returns:
        np.array: YBR to RGB lookup table
    """
    global _ybr_to_rgb_lut

    # return lut if already exists
    if _ybr_to_rgb_lut is not None:
        return _ybr_to_rgb_lut
    
    # try loading from file
    lut_path = Path(__file__).parent / 'ybr_to_rgb_lut.npy'
    if lut_path.is_file():
        _ybr_to_rgb_lut = np.load(lut_path)
        return _ybr_to_rgb_lut

    # else generate lut
    a = np.arange(2 ** 8, dtype=np.uint8)
    ybr = np.concatenate(np.broadcast_arrays(a[:, None, None, None], a[None, :, None, None], a[None, None, :, None]), axis=-1)
    _ybr_to_rgb_lut = dicom.pixel_data_handlers.util.convert_color_space(ybr, 'YBR_FULL', 'RGB')
    if save_lut:
        np.save(lut_path, _ybr_to_rgb_lut)
    return _ybr_to_rgb_lut


def read_video(
    path,
    n_frames=None,
    sample_period=1,
    out_fps=None,
    fps=None,
    frame_interpolation=True,
    random_start=False,
    res=None,
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0,
    region=None  # (i_start, i_end, j_start, j_end)
):
    """
    Read video from file with various processing options.
    
    Args:
        path: Path to video file
        n_frames: Number of frames to read
        sample_period: Frame sampling period
        out_fps: Target output FPS
        fps: Input video FPS (if None, will be detected)
        frame_interpolation: Whether to interpolate frames for FPS conversion
        random_start: Whether to start at random position
        res: Target resolution (width, height)
        interpolation: OpenCV interpolation method
        zoom: Zoom factor
        region: Region to crop (i_start, i_end, j_start, j_end)
        
    Returns:
        tuple: (frames, video_size, fps)
    """
    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    
    if res is not None:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)
    else:
        if region is None:
            out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
        else:
            out = np.zeros((n_frames, region[1] - region[0], region[3] - region[2]), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if region is not None:
            frame = frame[region[0]:region[1], region[2]:region[3]]
        if res is not None:
            # Import here to avoid circular imports
            from ..preprocessors.image_scaling import crop_and_scale
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps
