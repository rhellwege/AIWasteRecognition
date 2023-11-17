from PIL import Image
from flask import Flask, render_template, request, make_response, send_file
import json
import io
import threading

from predict import Predictor

# initialize globals
predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device ='cpu')
# predictor = Predictor("./models/1-48x48-CPUModel-61.pts", device ='cpu')

app = Flask(__name__)
plt_mutex = threading.Lock() # block the main thread whenever matplot lib is generating an image

@app.route('/')
def index():
    print(f'Hello {request.remote_addr}')
    return render_template('index.html')

# expects an image field in the request in formdata where image is the image file
@app.route('/predict', methods=['POST'])
def upload():
    if not 'image' in request.files:
        return make_response("No Image uploaded.", 404)
    image = request.files['image']
    # Save the uploaded image to a folder (e.g., 'uploads')
    image = Image.open(image)
    result = json.dumps(predictor.predict(image))
    return result, 200, {'Content-Type': 'application/json'} # set content type to json

# expects an image and a label in formdata
@app.route('/train', methods=['PUT'])
def train_model():
    if not 'image' in request.files:
        return make_response('no Image uploaded', 404)
    image = request.files['image']
    image = Image.open(image)
    label = request.form.get('label')    
    result = predictor.train(image, label)
    result = json.dumps(result)
    return result, 200, {'Content-Type': 'application/json'} # set content type to json

@app.route('/explore-predictions', methods=['GET'])
def get_explore_image():
    with app.app_context():
        with plt_mutex:
            imgbytes = predictor.explore_predictions(width=5)
    return send_file(imgbytes, mimetype='image/png')
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
