from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import cv2

class SegmentationResultWindow(QWidget):
    def __init__(self, original_image, mask_image, segmented_image):
        super().__init__()

        self.setWindowTitle("Segmentation Result")
        self.layout = QVBoxLayout()

        # Create labels for displaying images
        self.original_label = QLabel("Original Image")
        self.mask_label = QLabel("Mask Image")
        self.segmented_label = QLabel("Segmented Image")

        # Convert images to QPixmap
        original_pixmap = self.convert_cv_image_to_pixmap(original_image)
        mask_pixmap = self.convert_cv_image_to_pixmap(mask_image)
        segmented_pixmap = self.convert_cv_image_to_pixmap(segmented_image)

        # Set the QPixmap to labels
        self.original_label.setPixmap(original_pixmap)
        self.mask_label.setPixmap(mask_pixmap)
        self.segmented_label.setPixmap(segmented_pixmap)

        # Add labels to layout
        self.layout.addWidget(self.original_label)
        self.layout.addWidget(self.mask_label)
        self.layout.addWidget(self.segmented_label)

        self.setLayout(self.layout)
    
    def convert_cv_image_to_pixmap(self, cv_image):
        if len(cv_image.shape) == 2:  # Grayscale image
            height, width = cv_image.shape
            bytes_per_line = width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
        return QPixmap.fromImage(q_image)
