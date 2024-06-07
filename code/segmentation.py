import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    color = np.uint8([[[b, g, r]]])  # OpenCV uses BGR format
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv_color[0][0]

def create_color_mask(image, target_rgb, tolerance=40):
    # Convert target RGB to HSV
    target_hsv = rgb_to_hsv(*target_rgb)
    
    # Define the lower and upper bounds for the HSV values
    lower_bound = np.array([max(0, target_hsv[0] - tolerance), 50, 50])
    upper_bound = np.array([min(179, target_hsv[0] + tolerance), 255, 255])

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create the mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def segment_image(image, target_rgb, tolerance=40):
    # Create mask for the specified color
    image1 = image.copy()
    image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    mask = create_color_mask(image, target_rgb, tolerance)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around contours
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle
        if w > 10 or h > 10:  # Define the threshold for larger rectangles
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 5)  # Thick white border
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White Border

    # List to hold the features
    features = []

    # Compute features for each contour
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the region of interest (ROI)
        roi = image[y:y+h, x:x+w]
        
        # Compute the average color (BGR channels) of the ROI
        avg_color = np.mean(roi, axis=(0, 1)).astype(int)
        
        # Extract features
        length = h
        breadth = w
        avg_red, avg_green, avg_blue = avg_color
        features.append((length, breadth, avg_red, avg_green, avg_blue))
    
    # Return original image, mask, segmented image, contours, and features
    return image_rgb, mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), contours, features


'''
def segment_image(image, target_rgb, tolerance=40):
    # Create mask for the specified color
    image1 = image.copy()
    image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    mask = create_color_mask(image, target_rgb, tolerance)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to hold the features
    features = []

    # Draw rectangles around contours and extract features
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Store the length and breadth
        length = h
        breadth = w
        features.append((length, breadth))
        
        # Draw rectangle
        if w > 10 or h > 10:  # Define the threshold for larger rectangles
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 5)  # Thick white border
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Magenta border

    # Return original image, mask, segmented image, and features
    return image_rgb, mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), contours, features
'''