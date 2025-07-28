import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

base_dir = 'data'
patient_dirs = sorted(os.listdir(base_dir))

images = []
masks = []

for patient_id in patient_dirs:
    img_dir = os.path.join(base_dir, patient_id, 'PNG')
    msk_dir = os.path.join(base_dir, patient_id, 'mask_png')

    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        continue

    image_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(msk_dir))

    print(f' Обработка: пациент {patient_id}')

    for img_name, msk_name in zip(image_files, mask_files):
        if os.path.splitext(img_name)[0] != os.path.splitext(msk_name)[0]:
            raise ValueError(f"Несовпадение файлов: {img_name} и {msk_name}")

        img_path = os.path.join(img_dir, img_name)
        msk_path = os.path.join(msk_dir, msk_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f" Ошибка чтения файла: {img_path} или {msk_path}")
            continue

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        mask = (mask > 0).astype(np.uint8)

        if not set(np.unique(mask)).issubset({0, 1}):
            raise ValueError("Маска содержит значения, отличные от 0 и 1")

        images.append(image)
        masks.append(mask)

images = np.array(images)
masks = np.array(masks)

print(f' Всего загружено: {images.shape[0]} примеров')

X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

split_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

os.makedirs('dataset', exist_ok=True)

for name, array in split_data.items():
    np.save(os.path.join('dataset', f'{name}.npy'), array)
    print(f' Сохранено: {name} — {array.shape[0]} примеров')
