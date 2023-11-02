from flask import Flask
import flask
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root_endpoint():
    return flask.render_template('index.html') 

if __name__== '__main__':
    app.run(debug=True)
