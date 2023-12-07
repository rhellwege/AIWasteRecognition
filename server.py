from PIL import Image
from flask import Flask, render_template, request, make_response, send_file
import json

from predict import Predictor

default_model = "./models/3-64x64-CPUModel-89.pts"

# initialize globals
# predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device ='cpu')
predictor = Predictor(default_model, device ='cpu')

app = Flask(__name__)

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
    print("[SERVER] :: ", result)
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
    print("[SERVER] :: ", result)
    return result, 200, {'Content-Type': 'application/json'} # set content type to json

# returns bytes of a png image
@app.route('/explore-predictions', methods=['GET'])
def get_explore_image():
    width = request.args.get('width')
    imgbytes = predictor.explore_predictions(width=5 if width == None else int(width))
    return send_file(imgbytes, mimetype='image/png')

@app.route('/reload-model', methods=['PUT'])
def reload_model_endpoint():
    global predictor
    predictor = Predictor(default_model, device = 'cpu')
    print("[SERVER] :: ", 'reseting predictor weights')
    return 'Reloaded model.'

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
