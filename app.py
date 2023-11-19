from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import os
import cv2

app = Flask(__name__)

# Load the image segmentation model
model = joblib.load('rForestmodel.pkl')

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

            return render_template('result.html',
                                   original_image=original_image_path,
                                   segmented_image=segmented_image_path,
                                   image_with_rectangles=image_with_rectangles_path)

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
