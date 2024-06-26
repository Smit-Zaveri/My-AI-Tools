from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

# Global variables for face detection and tools
detect_faces = False
video_window_open = False

# Model files
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Load models (initialize outside of functions)
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
    gender_net.setInput(blob)
    return gender_net.forward()

def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    return age_net.forward()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if detect_faces:
                faces = get_faces(frame)
                for (start_x, start_y, end_x, end_y) in faces:
                    face_img = frame[start_y:end_y, start_x:end_x]
                    age_preds = get_age_predictions(face_img)
                    gender_preds = get_gender_predictions(face_img)
                    gender = GENDER_LIST[gender_preds[0].argmax()]
                    age = AGE_INTERVALS[age_preds[0].argmax()]
                    label = f"{gender}, {age}"
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', video_window_open=video_window_open)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detect_faces
    detect_faces = True
    return redirect(url_for('index'))

@app.route('/toggle_video_window')
def toggle_video_window():
    global video_window_open
    video_window_open = not video_window_open
    return redirect(url_for('index'))

@app.route('/close_program')
def close_program():
    # Perform any cleanup operations here if needed
    return 'Program closed.'

if __name__ == "__main__":
    app.run(debug=True)
