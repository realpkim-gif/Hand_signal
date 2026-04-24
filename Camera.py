import tensorflow as tf
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
import os

# Determinism
tf.keras.utils.set_random_seed(30)
tf.config.experimental.enable_op_determinism()


# Load your model
model = tf.keras.models.load_model("sign_language_model.keras")

# Labels
labels = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S',
    'T','U','V','W','X','Y'
]

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Rectangle coordinates
    x1, y1 = 200, 200
    x2, y2 = 400, 400

    # Draw rectangle on screen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Extract region of interest
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    digit = cv2.resize(blur, (28, 28))
    digit = digit / 255.0
    digit = digit.reshape(1, 28,28,1)

    prediction = model.predict(digit, verbose=0)
    digit_class = np.argmax(prediction)
    confidence = np.max(prediction)


    print(labels[digit_class])

    cv2.imshow("CAMERA", frame)
    cv2.imshow("ead", blur)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
