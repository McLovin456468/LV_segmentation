import cv2
import numpy as np
import pydicom


def preprocess_dicom(file_path, target_size=(256, 256)):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    original_shape = img.shape

    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    resized_img = cv2.resize(norm_img, target_size, interpolation=cv2.INTER_AREA)
    resized_img = np.expand_dims(resized_img, axis=-1)  # (256, 256, 1)

    return resized_img, original_shape, img


def restore_original_size(pred_mask, original_shape):
    pred_mask = pred_mask.squeeze()
    return cv2.resize(pred_mask.astype(np.uint8), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


def calculate_volume(masks, pixel_spacings, slice_thicknesses):
    total_volume_mm3 = 0.0
    for i in range(len(masks)):
        spacing_x, spacing_y = pixel_spacings[i]
        thickness = slice_thicknesses[i]
        voxel_volume = spacing_x * spacing_y * thickness  # мм³
        voxel_count = np.sum(masks[i] > 0.5)
        total_volume_mm3 += voxel_count * voxel_volume
    return total_volume_mm3 / 1000.0  # мл
