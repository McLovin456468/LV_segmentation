import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFont
from PyQt5.QtCore import Qt
import cv2
from inference_gui import load_model_and_predict
from utils import restore_original_size
import pydicom
import numpy as np


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LV seg")
        self.setGeometry(100, 100, 800, 870)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        font = QFont("Arial", 10)
        self.setFont(font)

        self.file_paths = []
        self.current_index = 0
        self.segmented_results = {}
        self.per_image_volumes = {}

        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        self.load_button = QPushButton("Загрузить DICOM/IMA-файлы")
        self.prev_button = QPushButton("← Назад")
        self.next_button = QPushButton("Вперед →")
        self.segment_button = QPushButton("Сегментировать")

        self.label = QLabel("Здесь будут отображены МРТ-снимки")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(768, 768)

        self.area_label = QLabel("Снимок: N/A")
        self.area_label.setAlignment(Qt.AlignCenter)

        self.volume_label = QLabel("Объемы ЛЖ: N/A")
        self.volume_label.setAlignment(Qt.AlignCenter)

        self.load_button.clicked.connect(self.load_images)
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.segment_button.clicked.connect(self.segment_all_images)

        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)
        self.button_layout.addWidget(self.segment_button)

        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.area_label)
        self.layout.addWidget(self.volume_label)
        self.setLayout(self.layout)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите DICOM-файлы", "", "DICOM файлы (*.dcm *.ima)"
        )
        if files:
            self.file_paths = files
            self.current_index = 0
            self.segmented_results.clear()
            self.per_image_volumes.clear()
            self.area_label.setText("Снимок: N/A")
            self.volume_label.setText("Объемы ЛЖ: N/A")
            self.show_image()

    def show_image(self):
        if not self.file_paths:
            return

        path = self.file_paths[self.current_index]

        if path in self.segmented_results:
            overlay = self.segmented_results[path]
            pixmap = self._to_pixmap(overlay)
            self.label.setPixmap(pixmap)
        else:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pixmap = self._to_pixmap(img_rgb)
            self.label.setPixmap(pixmap)

        base_text = f"Снимок {self.current_index + 1}/{len(self.file_paths)}"

        if path in self.per_image_volumes:
            volume = self.per_image_volumes[path]
            self.area_label.setText(f"{base_text} | Объем: {volume:.2f} мл")
        else:
            self.area_label.setText(base_text)

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next(self):
        if self.current_index < len(self.file_paths) - 1:
            self.current_index += 1
            self.show_image()

    def segment_all_images(self):
        if not self.file_paths:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите DICOM/IMA-файлы.")
            return

        volumes = []
        self.per_image_volumes.clear()

        for path in self.file_paths:
            try:
                pred_mask, original_shape, raw_img = load_model_and_predict(path)
                restored_mask = restore_original_size(pred_mask, original_shape)

                ds = pydicom.dcmread(path)
                spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
                thickness = float(getattr(ds, 'SpacingBetweenSlices', getattr(ds, 'SliceThickness', 1.0)))

                voxel_volume = spacing[0] * spacing[1] * thickness
                voxel_count = np.sum(restored_mask > 0.5)
                volume = voxel_count * voxel_volume / 1000.0  # мл

                volumes.append(volume)
                self.per_image_volumes[path] = volume

                norm_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
                base_img = (norm_img * 255).astype(np.uint8)
                if len(base_img.shape) == 2:
                    base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

                overlay = self._overlay_mask(base_img, restored_mask)
                self.segmented_results[path] = overlay

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка обработки файла {path}:\n{str(e)}")

        if volumes:
            min_vol = min(volumes)
            max_vol = max(volumes)
            stroke_vol = max_vol - min_vol
            self.volume_label.setText(
                f"Объем ЛЖ:\n в диастолу (КДО): {max_vol:.2f} мл | в систолу (КСО): {min_vol:.2f} мл | УО: {stroke_vol:.2f} мл"
            )
        else:
            self.volume_label.setText("Не удалось вычислить объем.")

        self.show_image()

    def _overlay_mask(self, image, mask):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[..., 0] = 255
        color_mask[..., 1] = 128
        color_mask[..., 2] = 128

        alpha = 0.4
        mask_bool = mask.astype(bool)
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)

        blended = image.copy()
        blended[mask_3ch] = (1 - alpha) * image[mask_3ch] + alpha * color_mask[mask_3ch]
        blended = blended.astype(np.uint8)

        return blended

    def _to_pixmap(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())
