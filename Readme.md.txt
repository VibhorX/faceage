# Real-time Age Detection using OpenCV and Dlib

This Python script demonstrates real-time age detection using a pre-trained deep learning model. It captures video from the webcam, detects faces in the video frames using the Dlib library, and predicts the age of each detected face using a pre-trained age detection model.

## Requirements

- Python 3.x
- OpenCV
- Dlib
- Numpy

## Usage

1. Make sure you have all the required libraries installed (`pip install opencv-python dlib numpy`).
2. Run the Python script (`python age_detection.py`).
3. Position yourself in front of the webcam.
4. The script will detect your face and predict your age in real-time.

## Description

- The script uses OpenCV to capture video frames from the webcam.
- It converts each frame to grayscale and uses the Dlib library to detect faces in the frames.
- For each detected face, it extracts the face region and preprocesses it for age prediction.
- The pre-trained age detection model is used to predict the age of each face.
- The predicted age is displayed on the video feed along with a bounding box around the detected face.

## Model Details

- The age detection model is loaded using the `cv2.dnn.readNet` function, with pre-trained weights and configuration files.
- The model predicts the age range of each face, which is mapped to predefined age labels.

## Additional Notes

- This script is for educational purposes and may require optimizations for real-world applications.
- The accuracy of age prediction may vary based on factors such as lighting conditions and facial expressions.

