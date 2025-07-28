import os
import matplotlib.pyplot as plt
from PIL import Image


patient_id = '33'
base_path = 'data'
img_dir = os.path.join(base_path, patient_id, 'PNG')
mask_dir = os.path.join(base_path, patient_id, 'mask_png')

img_files = os.listdir(img_dir)

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)

    if not os.path.exists(mask_path):
        print(f"Маска не найдена для : {img_file}")
        continue

    image = Image.open(img_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Image: {img_file}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Overlay: {img_file}")
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
