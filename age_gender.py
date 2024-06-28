import cv2
import numpy as np
from utils import get_faces

# Configuration for face detection and prediction models
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Load face detection model
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Load age and gender prediction models
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = get_faces(frame)
        for (start_x, start_y, end_x, end_y) in faces:
            face_img = frame[start_y:end_y, start_x:end_x]
            age_preds = get_age_predictions(face_img)
            gender_preds = get_gender_predictions(face_img)
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence = gender_preds[0][i]
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence = age_preds[0][i]
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0) if gender == "Male" else (147, 20, 255), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if gender == "Male" else (147, 20, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
