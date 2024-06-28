from flask import Flask, render_template, request, send_file, Response, redirect, url_for
import tempfile
import os
from age_gender import generate_frames, get_gender_predictions, get_age_predictions
from blur_faces import process_video

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blurfaces', methods=['GET', 'POST'])
def blur_faces():
    if request.method == 'POST':
        # Save the uploaded file to a temporary location
        input_file = request.files['video']
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(input_path)

        # Process the video file and blur the faces
        output_path = os.path.join(temp_dir, 'output.avi')
        process_video(input_path, output_path)

        # Provide a URL to the processed video
        return render_template('blurface.html', video_path=output_path)
    return render_template('blurface.html')

@app.route('/video/<path:path>')
def serve_video(path):
    return send_file(path)

@app.route('/download/<path:path>')
def download_file(path):
    return send_file(path, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
