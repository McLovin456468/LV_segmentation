import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_dicom

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path("saved_models/light_unet_92.h5")
model = load_model(model_path, compile=False)

def load_model_and_predict(file_path):
    image, original_shape, raw_img = preprocess_dicom(file_path)
    input_tensor = np.expand_dims(image, axis=0)  # (1, 256, 256, 1)
    prediction = model.predict(input_tensor)[0]
    pred_mask = (prediction > 0.5).astype(np.uint8)
    return pred_mask, original_shape, raw_img
