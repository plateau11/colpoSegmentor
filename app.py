from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import os
import cv2

app = Flask(__name__)

# Load the image segmentation model
model = joblib.load('rForestmodel.pkl')

def find_segmentedRegion(original_image_path, segmented_image_path):
    colposcopy_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)

    # Create a copy of the colposcopy image to preserve the original
    image_with_boundaries_and_filled_regions = colposcopy_image.copy()

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a blue color for drawing the boundaries (255, 0, 0) in BGR format
    boundary_color = (255, 0, 0)

    # Iterate through the contours and draw the boundaries in blue
    cv2.drawContours(image_with_boundaries_and_filled_regions, contours, -1, boundary_color, 2)

    # Define a yellow color for filling the regions (0, 255, 255) in BGR format
    yellow_fill_color = (0, 255, 255)

    # Create a mask for the regions of interest
    roi_mask = np.zeros_like(mask_image)

    # Iterate through the contours and fill them with the yellow color
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the screen size of the rectangle (assuming a specific DPI)
        dpi = 96  # Change this value to match your screen's DPI
        screen_width_mm = w / dpi * 25.4  # Convert width to mm
        screen_height_mm = h / dpi * 25.4  # Convert height to mm
        
        # Filter out rectangles with a width or height less than or equal to 1mm
        if screen_width_mm <= 1 or screen_height_mm <= 1:
            continue

        # Fill the region inside the contour with white in the ROI mask
        cv2.fillPoly(roi_mask, [contour], 255)

    # Fill the regions of interest with yellow color in the image
    image_with_boundaries_and_filled_regions[roi_mask > 0] = yellow_fill_color

    # Convert the BGR images to RGB format for Matplotlib
    colposcopy_image_rgb = cv2.cvtColor(colposcopy_image, cv2.COLOR_BGR2RGB)
    image_with_boundaries_and_filled_regions_rgb = cv2.cvtColor(image_with_boundaries_and_filled_regions, cv2.COLOR_BGR2RGB)
    return image_with_boundaries_and_filled_regions_rgb

def detect_abnormalRegion(original_image, segmented_image):
    # Create a copy of the original image to preserve the original
    image_with_rectangles = original_image.copy()

    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a color for drawing rectangles (here is green)
    rectangle_color = (0, 255, 0)  # Green in BGR format

    # Iterate through the contours and draw rectangles around them
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the abnormal region on the copy
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), rectangle_color, 2)

    return image_with_rectangles


def segment_image(img):
    # Convert RGB to BGR
    img = img[:, :, ::-1]
    flattened_image = img.reshape(-1, 3)
    segmented_image = model.predict(flattened_image)
    return segmented_image.reshape((600, 800))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    

    if file:
        try:
            # Read the image file
            img = Image.open(file)

            # Specify the image format explicitly 
            img.format = 'JPEG'  
            # Resize the image to the required shape (600, 800, 3)
            img = img.resize((800, 600))
            img_array = np.array(img)

            # Perform image segmentation
            segmented_image = segment_image(img_array)

            # Save images to the static directory
            static_dir = 'static'
            os.makedirs(static_dir, exist_ok=True)

            original_image_path = os.path.join(static_dir, 'original.jpg')
            segmented_image_path = os.path.join(static_dir, 'segmented.png')

            img.save(original_image_path)
            Image.fromarray(segmented_image).save(segmented_image_path)

            # Add rectangles to the original image and save
            image_with_rectangles = detect_abnormalRegion(img_array, segmented_image)
            image_with_rectangles_path = os.path.join(static_dir, 'image_with_rectangles.jpg')
            Image.fromarray(image_with_rectangles).save(image_with_rectangles_path)

            final_segmentation = find_segmentedRegion(original_image_path, segmented_image_path)
            final_segmentation_path = os.path.join(static_dir, 'finalSegmentation.jpg')
            Image.fromarray(final_segmentation).save(final_segmentation_path)

            return render_template('result.html',
                                   original_image=original_image_path,
                                   segmented_image=segmented_image_path,
                                   image_with_rectangles=image_with_rectangles_path,
                                   final_segmentation = final_segmentation_path)

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
