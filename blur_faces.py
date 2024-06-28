import cv2
import numpy as np

def process_video(input_path, output_path):
    # Load the pre-trained deep learning model
    prototxt_path = "weights/deploy.prototxt.txt"
    model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Open the input video file
    cap = cv2.VideoCapture(input_path)

    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    _, image = cap.read()
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (image.shape[1], image.shape[0]))

    # Process each frame of the video
    while True:
        captured, image = cap.read()
        if not captured:
            break

        # Detect faces and blur them
        h, w = image.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        output = np.squeeze(model.forward())
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.4:
                box = output[i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype(int)
                face = image[start_y: end_y, start_x: end_x]
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                image[start_y: end_y, start_x: end_x] = face

        # Write the frame to the output video file
        out.write(image)

    # Release the resources
    cap.release()
    out.release()
