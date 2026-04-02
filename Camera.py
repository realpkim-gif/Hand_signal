import tensorflow as tf

tf.keras.utils.set_random_seed(30)
tf.config.experimental.enable_op_determinism()

from tensorflow.keras import datasets
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_save = tf.keras.models.load_model("sign_language_model.keras")

import cv2

"""0 means the default webcam."""

labels = [
'A','B','C','D','E','F','G','H','I',
'K','L','M','N','O','P','Q','R','S',
'T','U','V','W','X','Y'
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video stream or file")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame. Exiting...")
            break

        #preprocess camera data so model can read
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (28, 28))

        img = resized.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        prediction = model_save.predict(img, verbose=0)
        pred_class = np.argmax(prediction)

        letter = labels[pred_class]

        cv2.putText(frame, letter, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Hand Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

