from PIL import Image
from flask import Flask, render_template, request

from predict import Predictor

app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('index.html')

predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device ='cpu')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        # Save the uploaded image to a folder (e.g., 'uploads')
        image = Image.open(image)
        a = str(predictor.predict(image))
        print()
        return a
    return 'No image uploaded.'

if __name__ == '__main__':

    app.run(debug=True)
