import os
import pathlib
from pathlib import Path
import cv2
import yaml
import numpy as np
import coco_dataset_manager


def raw14bit_to_cv_frame(raw_14bit, channel3_mode = 'laplacian'):
    """
    transform 14-bit thermal image to a 3-channel 8bit image for deep learning detection.
    channel 1 - rescale gray levels to 5-95 percentiles
    channel 2 - clahe local gray scale rescaling. (better "dynamic range")
    channel 3 - edge enhanced images
                'laplacian' - blob enhancement
                'DDE' - gradient enhancement

    :param raw_14bit (np.array): Raw uint16 data from the Twin612.
    :param channel3_mode (string): what to put in channel 3? 'laplacian' / 'DDE'

    :return: cv image
    """

    channel3_mode = 'laplacian'  # 'laplacian' / 'DDE'

    # --- Channel 1: Robust Percentile Linear Scaling ---
    # Instead of absolute min/max (which causes flickering), we use percentiles.
    # This ignores the bottom 0.5% and top 0.5% of pixel values (noise/outliers).
    p_low, p_high = np.percentile(raw_14bit, [0.5, 99.5])
    if p_high <= p_low:
        p_high = np.max(raw_14bit)
        p_low = np.min(raw_14bit)

    if p_high > p_low:
        # Apply the scale: (val - p_low) / (p_high - p_low) * 255
        # We clip to ensure values stay within 0-255
        ch1 = np.clip((raw_14bit - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        ch1 = np.zeros_like(raw_14bit).astype(np.uint8)

    # --- Channel 2: CLAHE (Local Contrast for Small Targets) ---
    # LIGHT DENOISING (Gaussian, NOT Median)
    # We use a tiny Gaussian blur to smooth sensor grain before CLAHE amplifies it.
    # This keeps your small 3x3 targets alive.
    denoised = cv2.GaussianBlur(ch1, (3, 3), 0)

    # 3. CLAHE (Local Contrast Enhancement)
    # clipLimit 2.0 is the 'sweet spot' for thermal targets.
    # tileGridSize (8,8) works well for 640x512 resolution.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch2 = clahe.apply(denoised)

    # TARGET BEEF-UP (Dilation)
    # If your targets are 1-3 pixels, we dilate to make them 'solid' for the CNN.
    enable_dilation = False
    if enable_dilation:
        kernel = np.ones((3, 3), np.uint8)
        ch2 = cv2.dilate(ch2, kernel, iterations=1)

    # --- Channel 3: edges and small targets enhancement
    if channel3_mode == 'laplacian':
        # --- Channel 3: Laplacian (Point Source & Edge Detection) ---
        # This captures the 'discontinuity' of a 4x4 drone better than unsharp masks.
        laplacian = cv2.Laplacian(ch1, cv2.CV_16S, ksize=5)
        # Convert back to uint8 with absolute values to catch hot and cold edges
        ch3 = cv2.convertScaleAbs(laplacian)

    elif channel3_mode == 'DDE':
        # Highlights the structural silhouette of the drone
        blurred = cv2.GaussianBlur(ch1, (3, 3), 0)
        # Unsharp mask formula: Result = Original + (Original - Blurred) * Amount
        ch3 = cv2.addWeighted(ch1, 1.5, blurred, -0.5, 0)

    else:
        raise Exception('channel3_mode not supported')

    # # 5. OPTIONAL: TEMPORAL MOTION MASK
    # # If you pass the last_frame, we can return a motion mask to help the detector.
    # motion_mask = None
    # if last_frame is not None:
    #     diff = cv2.absdiff(enhanced, last_frame)
    #     _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Stack into BGR (or RGB) for YOLO
    cv_image = cv2.merge([ch1, ch2, ch3])

    return cv_image


def convert_dataset_images(images_folder, save_seperate_channel_images=False, verbose=True):
    """
    copy original images folder to images_raw
    make images folder with the same image names, but converted images.
    
    :param images_folder: input images folder with
    :return: 
    """


    images_folder = pathlib.Path(images_folder)

    # move original folder
    new_images_folder = images_folder.with_name('images_raw_14bit')
    print('in the base folder: {}'.format(images_folder.parent))
    print('moving {} to {}'.format(images_folder.stem , new_images_folder.stem))
    os.rename(images_folder, new_images_folder)

    # make new folder with the original_name
    os.makedirs(images_folder, exist_ok=True)

    if save_seperate_channel_images:
        images_folder_ch1 = images_folder.with_name('images_ch1')
        images_folder_ch2 = images_folder.with_name('images_ch2')
        images_folder_ch3 = images_folder.with_name('images_ch3')
        os.makedirs(images_folder_ch1, exist_ok=True)
        os.makedirs(images_folder_ch2, exist_ok=True)
        os.makedirs(images_folder_ch3, exist_ok=True)

    # convert all images
    print('writing new images to {}'.format(images_folder.stem))
    if save_seperate_channel_images:
        print('writing image ch1 to {}'.format(images_folder_ch1.stem))
        print('writing image ch2 to {}'.format(images_folder_ch2.stem))
        print('writing image ch3 to {}'.format(images_folder_ch3.stem))

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [p for p in new_images_folder.iterdir() if p.suffix.lower() in extensions]
    for img_file in images:
        img_14bit = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        img_cv = raw14bit_to_cv_frame(img_14bit)
        cv2.imwrite(str(images_folder / (img_file.stem+'.png')), img_cv)

        if save_seperate_channel_images:
            cv2.imwrite(str(images_folder_ch1 / (img_file.stem + '.png')), img_cv[:,:,0])
            cv2.imwrite(str(images_folder_ch2 / (img_file.stem+'.png')), img_cv[:,:,0])
            cv2.imwrite(str(images_folder_ch3 / (img_file.stem+'.png')), img_cv[:,:,0])

if __name__ == '__main__':
    images_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260415_reshafim/20260415_1202_12/20260415_1204_46/camera_20260415_1204_extracted/images'

    convert_dataset_images(images_folder, save_seperate_channel_images=True)

    print('Done!')