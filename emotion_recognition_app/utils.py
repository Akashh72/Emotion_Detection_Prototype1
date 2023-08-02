# utils.py
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
import os

def perform_emotion_recognition(image):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'best_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    mtcnn_detector = MTCNN()

    image_data = image.read()
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    rgb_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    faces = mtcnn_detector.detect_faces(rgb_image)

    emotions_detected = []
    if len(faces) > 0:

        for face in faces:
            x, y, w, h = face['box']
            face_roi = rgb_image[y:y+h, x:x+w]
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
            img_pixels = img_to_array(face_roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            prediction = model.predict(img_pixels)
            max_index = np.argmax(prediction)
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            emotions_detected.append((predicted_emotion, (x, y, w, h)))
    else:
            # No face detected, use OpenCV to detect face
            face_cascade = cv2.CascadeClassifier('emotion_recognition_app\haarcascade_frontalface_alt2.xml')
            faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                face_roi = rgb_image[y:y+h, x:x+w]
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
                img_pixels = img_to_array(face_roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0

                prediction = model.predict(img_pixels)
                max_index = np.argmax(prediction)
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                emotions_detected.append((predicted_emotion, (x, y, w, h)))

    return emotions_detected
