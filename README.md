# Traffic Sign Classification using Deep Learning

## Overview
This project implements a traffic sign classification system using a pre-trained deep learning model. The model identifies and classifies traffic signs from input images by detecting shapes, extracting regions of interest (ROI), and applying a convolutional neural network (CNN) for classification. The system utilizes OpenCV for image processing and TensorFlow/Keras for deep learning inference.

## Features
- **Traffic Sign Detection**: Identifies potential traffic signs using contour detection and shape filtering.
- **Preprocessing**: Converts images to grayscale, normalizes pixel values, and resizes them to match the model's input requirements.
- **Deep Learning Model**: Uses a pre-trained CNN model (`trained_model.h5`) to classify detected traffic signs.
- **Visualization**: Displays processed images with detected signs labeled accordingly.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```sh
pip install tensorflow opencv-python numpy
```

## Usage
1. Place your images inside a directory (e.g., `/content/sample_data/images`).
2. Run the script to process images and classify traffic signs:
```sh
python traffic_sign_classifier.py
```

## Code Structure
- **`load_model()`**: Loads the pre-trained traffic sign classification model.
- **`preprocess_frame(roi)`**: Preprocesses extracted ROIs before classification.
- **`detect_shapes(frame)`**: Detects shapes in an image using contour detection.
- **`process_images(image_dir)`**: Processes all images in the given directory and applies classification.

## Model Details
- The model is trained to classify 43 different traffic signs, including speed limits, cautionary signs, and mandatory directions.
- Uses TensorFlow/Keras for deep learning inference.
- Works with grayscale 32x32 pixel images.

## Example Output
The system will annotate detected traffic signs in the images and display them with bounding boxes and labels.

## Future Improvements
- Improve detection accuracy with advanced image processing techniques.
- Extend support for real-time video processing.
- Train a custom model for better accuracy.
