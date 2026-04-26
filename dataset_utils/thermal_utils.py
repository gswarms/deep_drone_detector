import cv2
import numpy as np


def preprocess_thermal_for_cv(raw_14bit, last_frame=None, enable_dilation=True):
    """
    Optimized preprocessing for 14-bit thermal data to detect small, fast targets.

    Args:
        raw_14bit (np.array): Raw uint16 data from the Twin612.
        last_frame (np.array): The previous processed frame (for motion logic).
        enable_dilation (bool): If True, 'beefs up' tiny targets.

    Returns:
        processed_8bit: High-contrast image for the CNN/Detector.
        motion_mask: (Optional) Binary mask showing what moved.
    """

    # 1. ROBUST NORMALIZATION (Percentile Scaling)
    # Instead of absolute min/max (which causes flickering), we use percentiles.
    # This ignores the bottom 0.5% and top 0.5% of pixel values (noise/outliers).
    p_low, p_high = np.percentile(raw_14bit, [0.5, 99.5])
    if p_high <= p_low:
        p_high = np.max(raw_14bit)
        p_low = np.min(raw_14bit)

    if p_high > p_low:
        # Apply the scale: (val - p_low) / (p_high - p_low) * 255
        # We clip to ensure values stay within 0-255
        img_8bit = np.clip((raw_14bit - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        img_8bit = np.zeros_like(raw_14bit).astype(np.uint8)

    # 2. LIGHT DENOISING (Gaussian, NOT Median)
    # We use a tiny Gaussian blur to smooth sensor grain before CLAHE amplifies it.
    # This keeps your small 3x3 targets alive.
    denoised = cv2.GaussianBlur(img_8bit, (3, 3), 0)

    # 3. CLAHE (Local Contrast Enhancement)
    # clipLimit 2.0 is the 'sweet spot' for thermal targets.
    # tileGridSize (8,8) works well for 640x512 resolution.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. TARGET BEEF-UP (Dilation)
    # If your targets are 1-3 pixels, we dilate to make them 'solid' for the CNN.
    if enable_dilation:
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)

    # 5. OPTIONAL: TEMPORAL MOTION MASK
    # If you pass the last_frame, we can return a motion mask to help the detector.
    motion_mask = None
    if last_frame is not None:
        diff = cv2.absdiff(enhanced, last_frame)
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    return enhanced, motion_mask


# Simple way to 'heal' known bad pixels
def heal_bad_pixels(img, mask):
    # mask is a binary image where 255 = bad pixel
    # This replaces bad pixels with the average of their neighbors
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def preprocess_thermal_for_calibration(raw_14bit):
    """
    Optimized preprocessing for 14-bit thermal data for camera calibration

    :param raw_14bit (np.array): Raw uint16 data from the Twin612.

    :return img: 8bit image
    """

    # 1. ROBUST NORMALIZATION (Percentile Scaling)
    # Instead of absolute min/max (which causes flickering), we use percentiles.
    # This ignores the bottom 0.5% and top 0.5% of pixel values (noise/outliers).
    p_low, p_high = np.percentile(raw_14bit, [0.5, 99.5])
    if p_high <= p_low:
        p_high = np.max(raw_14bit)
        p_low = np.min(raw_14bit)

    # Apply the scale: (val - p_low) / (p_high - p_low) * 255
    # We clip to ensure values stay within 0-255
    if p_high > p_low:
        img_8bit = np.clip((raw_14bit - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        img_8bit = np.zeros_like(raw_14bit).astype(np.uint8)

    return img_8bit


def apply_custom_dde(raw_14bit, gain=2.0, sigma_s=15, sigma_r=0.1):
    """
    raw_14bit: numpy array (uint16)
    gain: how much to boost details
    sigma_s: spatial variance (size of the filter)
    sigma_r: range variance (how much temperature difference is 'an edge')
    """
    # 1. Normalize to 0.0 - 1.0 for processing
    BIT_DEPTH = 14
    MAX_VAL = (1 << BIT_DEPTH) - 1  # Result is 16383
    img_float = raw_14bit.astype(np.float32) / MAX_VAL

    # 2. Extract Base Layer (Edge-Preserving Smoothing)
    # Bilateral filter is great for avoiding the 'ringing' halos
    base = cv2.bilateralFilter(img_float, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)

    # 3. Extract Detail Layer
    # Detail = Original - Base
    detail = img_float - base

    # 4. Process the Base Layer (Dynamic Range Compression)
    # Here we just use a simple linear stretch, but you could use CLAHE
    base_8bit = np.clip(base * 255, 0, 255).astype(np.uint8)

    # 5. Recombine: Base + (Detail * Gain)
    # We convert detail back to the 0-255 scale
    enhanced_detail = (detail * gain * 255)

    # Final image recombination
    final_img = cv2.add(base_8bit.astype(np.float32), enhanced_detail)
    return np.clip(final_img, 0, 255).astype(np.uint8)