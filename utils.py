import cv2
import numpy as np

# Load face detection model
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                             frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y = max(0, start_x - 10), max(0, start_y - 10)
            end_x, end_y = min(frame.shape[1], end_x + 10), min(frame.shape[0], end_y + 10)
            faces.append((start_x, start_y, end_x, end_y))
    return faces
