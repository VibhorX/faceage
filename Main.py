import cv2
import dlib
import numpy as np

# Load the age detection model
age_weights = "C:/Users/HP/Desktop/Comp_Viz_Prog/faceage/Models/age_deploy.prototxt"
age_config = "C:/Users/HP/Desktop/Comp_Viz_Prog/faceage/Models/age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights)

# Model requirements for image
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Create a face detector object
face_detector = dlib.get_frontal_face_detector()

# Initialize video capture from your webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(img_gray)

    for face in faces:
        x = face.left()
        y = face.top()
        x2 = face.right()
        y2 = face.bottom()

        box = [x, y, x2, y2]
        face_img = frame[box[1]:box[3], box[0]:box[2]]

        # Image preprocessing
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), model_mean, swapRB=False)

        # Age prediction
        age_Net.setInput(blob)
        age_preds = age_Net.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 200), 2)
        cv2.putText(frame, f'Age: {age}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Age Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
