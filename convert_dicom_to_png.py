import os
import pydicom
import numpy as np
from PIL import Image

input = 'data/1/DICOM'
output = 'data/1/PNG'

os.makedirs(output, exist_ok=True)

for filename in os.listdir(input):
    if filename.lower().endswith('.dcm'):
        filepath = os.path.join(input, filename)
        dicom = pydicom.dcmread(filepath)
        pixel_array = dicom.pixel_array
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
        pixel_array = pixel_array.astype(np.uint8)
        image = Image.fromarray(pixel_array)
        output_path = os.path.join(output, filename.replace('.dcm', '.png'))
        image.save(output_path)

print("Изображения конвертированы")
