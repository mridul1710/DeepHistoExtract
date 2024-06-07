#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:54:39 2024

@author: sandeshacharya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:10:53 2024

@author: sandeshacharya
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QWidget, QDesktopWidget, QLineEdit, QMessageBox, QCheckBox, QScrollArea
from PyQt5.QtCore import Qt, QRect
import os
import sys
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import segmentation
import pandas as pd
import tf_model
import subprocess
import tensorflow as tf
from tensorflow.keras import layers, models

class ContourLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)
        self.checkbox = QCheckBox("True")
        self.checkbox.hide()
        self.checkbox.stateChanged.connect(self.checkbox_state_changed)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.checkbox)
        self.setLayout(self.layout)
        
        
    def enterEvent(self, event):
        self.checkbox.show()

    def leaveEvent(self, event):
        self.checkbox.hide()

    def checkbox_state_changed(self, state):
        if state == Qt.Checked:
            print("User marked the contour as true (1).")
        else:
            print("User unmarked the contour (0).")

class LabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Labeling Tool")
        
        # Set window size to 70% of the desktop size
        desktop = QDesktopWidget()
        screen_rect = desktop.screenGeometry(desktop.primaryScreen())
        width = int(screen_rect.width() * 0.7)
        height = int(screen_rect.height() * 0.7)
        self.setGeometry(0, 0, width, height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Input fields for RGB coordinates and tolerance
        self.rgb_input = QLineEdit()
        self.rgb_input.setPlaceholderText("Enter RGB coordinates (comma-separated)")
        self.layout.addWidget(self.rgb_input)

        self.tolerance_input = QLineEdit()
        self.tolerance_input.setPlaceholderText("Enter tolerance value")
        self.layout.addWidget(self.tolerance_input)

        # Button for segmenting the image
        self.segment_button = QPushButton("Segment Image")
        self.segment_button.clicked.connect(self.segment_image)
        self.layout.addWidget(self.segment_button)

        self.points = []
        # Initialize images_start_index
        self.images_start_index = 0
        self.image_label.mousePressEvent = self.mouse_click_event
        self.clicked_contour = None
        
            
    def mouse_click_event(self, event):
        x = event.pos().x()
        y = event.pos().y()

        # Iterate through contours to check if mouse click is within bounding box
        for contour in self.contours:
            x_contour, y_contour, w, h = cv2.boundingRect(contour)
            if x_contour <= x <= x_contour + w and y_contour <= y <= y_contour + h:
                self.clicked_contour = contour
                self.show_true_false_dialog()
                break

    def show_true_false_dialog(self):
        dialog = QMessageBox()
        dialog.setWindowTitle("Labeling")
        dialog.setText("Is this contour true or false?")
        dialog.setIcon(QMessageBox.Question)
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.buttonClicked.connect(self.true_false_result)
        dialog.exec_()

    def true_false_result(self, button):
        result = button.text()
        print("User clicked:", result, "for contour:", self.clicked_contour)
        # Perform further actions based on user's response, such as saving the result or updating UI
   
    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.tif)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize the image to fit within the window size
            image_height, image_width, _ = image.shape
            max_width = self.geometry().width() - 20  # Subtract some padding
            max_height = self.geometry().height() - 20  # Subtract some padding
            if image_width > max_width or image_height > max_height:
                scale_factor = min(max_width / image_width, max_height / image_height)
                new_width = int(image_width * scale_factor)
                new_height = int(image_height * scale_factor)
                image = cv2.resize(image, (new_width, new_height))

            q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

            self.image_label.mousePressEvent = self.get_mouse_click

            # Save the loaded image for segmentation
            self.loaded_image = image

    def get_mouse_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.points.append((x, y))
        print("Point:", x, y)

        # Draw points on the image
        if hasattr(self, 'image'):
            image = self.image.copy()
            for point in self.points:
                cv2.circle(image, point, 3, (255, 0, 0), -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def convert_cv_image_to_pixmap(self, cv_image, max_width=None, max_height=None):
        if len(cv_image.shape) == 2:  # Grayscale image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    
        # Get original image dimensions
        height, width, _ = cv_image.shape
    
        # Calculate scaling factors while maintaining aspect ratio
        scale_factor_width = max_width / width if max_width else 1
        scale_factor_height = max_height / height if max_height else 1
    
        # Use the minimum scaling factor to ensure the aspect ratio is maintained
        scale_factor = min(scale_factor_width, scale_factor_height)
    
        # Resize the image
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(cv_image, (new_width, new_height))
    
        # Convert resized image to QPixmap
        bytes_per_line = 3 * resized_image.shape[1]
        q_image = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)
    
    def segment_image(self):
        # Check if an image is already loaded
        if not hasattr(self, 'loaded_image'):
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return
    
        # Get RGB coordinates and tolerance from input fields
        rgb_input_text = self.rgb_input.text()
        tolerance_input_text = self.tolerance_input.text()
    
        # Check if input fields are empty
        if not rgb_input_text or not tolerance_input_text:
            QMessageBox.warning(self, "Error", "Please enter RGB coordinates and tolerance.")
            return
    
        try:
            # Convert RGB coordinates to list of integers
            rgb_coordinates = [int(x) for x in rgb_input_text.split(',')]
            # Convert tolerance to integer
            tolerance = int(tolerance_input_text)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input. RGB coordinates must be comma-separated integers, and tolerance must be an integer.")
            return
    
        # Pass image, coordinates, and tolerance to segmentation function
        image, mask, segmented_image, contours, features = segmentation.segment_image(self.loaded_image, rgb_coordinates, tolerance)
        # Save the contours and features as attributes of the class
        self.contours = contours
        self.features = features
        if not self.features:
            return
    
        # Create column names based on your feature structure (adjust as needed)
        columns = ["Length", "Breadth", "Avg_Red", "Avg_Green", "Avg_Blue"]
    
        # Create DataFrame from features
        df = pd.DataFrame(self.features, columns=columns)
        print(df)
        
        if os.path.exists("classifier.h5"):
            self.model = tf.keras.models.load_model("classifier.h5")
            predictions = self.model.predict(df)          
            print(predictions)
        
            # Draw boxes around contours with predictions = 1
            for contour, prediction in zip(self.contours, predictions):
                if prediction == 1:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        
            segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the segmented image
        segmented_pixmap = self.convert_cv_image_to_pixmap(segmented_image)
        self.image_label.setPixmap(segmented_pixmap)
    
        # Connect mouse click event to record_click method
        self.image_label.mousePressEvent = self.record_click

        # Add the "Done" button
        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.save_features_to_dataframe)
        self.layout.addWidget(self.done_button)

        # Connect mouse click event to show_checkbox method
        self.image_label.mousePressEvent = self.show_checkbox
    
    
    def record_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
    
        # Iterate through features and check if the click is within any contour
        for feature in self.features:
            length, breadth, _, _, _ = feature
            if x >= 0 and x < breadth and y >= 0 and y < length:
                feature_list = list(feature)  # Convert tuple to list
                feature_list.append(1)  # Clicked inside the contour, mark as True (1)
                feature = tuple(feature_list)  # Convert back to tuple
            else:
                feature_list = list(feature)  # Convert tuple to list
                feature_list.append(0)  # Clicked outside the contour, mark as False (0)
                feature = tuple(feature_list)  # Convert back to tuple
    
        # After recording output for all contours, print the features
        self.save_features_to_dataframe()


    def checkbox_state_changed(self, state, widget):
        if state == Qt.Checked:
            print(f"User marked the image as true (1) for widget {widget}.")
        else:
            print(f"User unmarked the image (0) for widget {widget}.")
    
    def show_checkbox(self, event):
        # Check if contours and features are available
        if not hasattr(self, 'contours') or not hasattr(self, 'features'):
            return
    
        # Get the mouse click coordinates
        x = event.pos().x()
        y = event.pos().y()
    
        # Iterate over each contour and its corresponding features
        for contour, feature in zip(self.contours, self.features):
            # Get the bounding box of the contour
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
    
            # Define a range around the bounding box edges
            edge_range = 5
    
            # Check if the mouse click coordinates are within the bounding box or its range
            if (x_contour - edge_range) <= x <= (x_contour + w_contour + edge_range) and \
               (y_contour - edge_range) <= y <= (y_contour + h_contour + edge_range):
                # Display the true/false checkbox
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Result")
                msg_box.setText(f"Length: {feature[0]}, Breadth: {feature[1]}")
                true_button = msg_box.addButton('True', QMessageBox.YesRole)
                false_button = msg_box.addButton('False', QMessageBox.NoRole)
                msg_box.exec_()
                if msg_box.clickedButton() == true_button:
                    updated_feature = feature + (1,)  # Mark as True (1)
                elif msg_box.clickedButton() == false_button:
                    updated_feature = feature + (0,)  # Mark as False (0)
                # Update the features list
                self.features[self.features.index(feature)] = updated_feature
                return
    
        
    def save_features_to_dataframe(self):
        # Check if features exist
        if not self.features:
            return
    
        # Create column names based on your feature structure (adjust as needed)
        columns = ["Length", "Breadth", "Avg_Red", "Avg_Green", "Avg_Blue", "Selection"]
    
        # Create DataFrame from features
        df = pd.DataFrame(self.features, columns=columns)
        print(df)
        test_accuracy = tf_model.tf_model(df.copy())  # Pass a copy to avoid modifying original data
    
        # Print or display the test accuracy as needed
        print(f"Test Accuracy: {test_accuracy:.2f}")
        # Restart the script using subprocess.call
        subprocess.call(['python', 'try7.py'])  # Replace 'try7.py' with your actual script name
        sys.exit()  # Exit the current instance
      
        
def main():
    app = QApplication(sys.argv)
    labeling_app = LabelingApp()
    labeling_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


