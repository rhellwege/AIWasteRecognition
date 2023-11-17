from PIL import Image
from flask import Flask, render_template, request, make_response, session

from predict import Predictor

# initialize globals
predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device ='cpu')

# cache of previously uploaded image used for live training.
# key: ip address of client, value: previous image.
prev_images = {} 

app = Flask(__name__)

@app.route('/')
def index():
    print(f'Hello {request.remote_addr}')
    return render_template('index.html')

# expects an image field in the request
@app.route('/predict', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        # Save the uploaded image to a folder (e.g., 'uploads')
        image = Image.open(image)
        prev_images[request.remote_addr] = image
        result = predictor.predict(image)
        return str(result)
    return 'No image uploaded.'

# This endpoint is only valid if the user has uploaded an image and there is an entry in prev_predictions
# expects a binary value: 
@app.route('/train', methods=['PUT'])
def correct_model():
    if not request.remote_addr in prev_images:
        return make_response('No previous image associated with your session', 403)
    data = request.get_json()
    label = data["label"]
    result = predictor.train(prev_images[request.remote_addr], label)
    return result

if __name__ == '__main__':
    app.run(debug=True)
