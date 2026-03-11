"""
Signature Verification System - Flask Web Application
Author: Sairam Odela
"""

import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, url_for

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_signature

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG/JPG/BMP.'}), 400

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_signature(filepath)
        result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Please train the model first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
