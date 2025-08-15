from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from CNNClassifier.pipeline.prediction import PredictionPipeline

# Environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ClientApp to store filename & classifier
class ClientApp:
    def __init__(self):
        self.filename = None
        self.classifier = None

clApp = ClientApp()

# Home route
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

# About route
@app.route("/about", methods=['GET'])
def about():
    return render_template('about.html')

# Predict route
@app.route("/predict", methods=['GET', 'POST'])
def predictRoute():
    result = None
    uploaded_filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded"
            return render_template('predict.html', result=result, uploaded_filename=uploaded_filename)

        file = request.files['file']
        if file.filename == '':
            result = "No file selected"
            return render_template('predict.html', result=result, uploaded_filename=uploaded_filename)

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        uploaded_filename = filename

        # Run prediction
        clApp.filename = filepath
        clApp.classifier = PredictionPipeline(clApp.filename)
        result = clApp.classifier.predict()

    return render_template('predict.html', result=result, uploaded_filename=uploaded_filename)

# Run app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
