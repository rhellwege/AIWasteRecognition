from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def root_endpoint():
    return 'hello'

if __name__== '__main__':
    app.run(debug=True)
