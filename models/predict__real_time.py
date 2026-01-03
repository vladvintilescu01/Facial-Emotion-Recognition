import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPU found. This script requires GPU.")
else:
    print("GPUs detected:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model = load_model("DenseNet121_FER2013.h5")

emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_HEIGHT, IMG_WIDTH = 160, 160

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict emotion (GPU automatically used)
        predictions = model.predict(face, verbose=0)
        emotion_idx = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_idx]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
