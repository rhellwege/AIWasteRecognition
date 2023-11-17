from PIL import Image
from flask import Flask, render_template, request, make_response, session

from predict import Predictor

# initialize globals
predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device ='cpu')

# cache of predictions for live training
# key: ip address of client, value: prediction tensor
prev_predictions = {} 

app = Flask(__name__)

@app.route('/')
def index():
    print(f'Hello {request.remote_addr}')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        # Save the uploaded image to a folder (e.g., 'uploads')
        image = Image.open(image)
        a = str(predictor.predict(image))
        return a
    return 'No image uploaded.'

@app.route('/correct', methods=['PUT'])
def correct_model():
    pass

if __name__ == '__main__':
    app.run(debug=True)
