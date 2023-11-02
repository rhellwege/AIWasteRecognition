import flask
import json

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def root_endpoint():
    return flask.render_template('index.html') 

# TODO: add endpoints for upload image, and correct model
# TODO: add option for which model to use, or to use no model as a test
# TODO: handle multiple clients at the same time?

if __name__== '__main__':
    app.run(debug=True)
