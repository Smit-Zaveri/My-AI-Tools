import os
from flask import Flask, render_template, request, send_file, Response, redirect, url_for
import tempfile
from age_gender import generate_frames
from blur_faces import process_video
from pdf_reader import extract_text_from_pdf, answer_question

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

@app.route('/pdf_reader', methods=['GET', 'POST'])
def pdf_reader():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        pdf_file.save(pdf_path)

        text = extract_text_from_pdf(pdf_path)
        question = request.form['question']
        answer = answer_question(text, question)

        return render_template('pdf_reader.html', text=text, question=question, answer=answer)
    return render_template('pdf_reader.html')

if __name__ == '__main__':
    app.run(debug=True)
