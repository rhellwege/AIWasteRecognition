# AI Waste Recognition
The purpose of this project is to test a convolutional neural network with user inputted images.
The server responds with the neural network's decision of whether the image contains trash or recycling,
and the user can then correct the NN live.

## Getting Started
Make sure you have python 3.11 and pip installed, then install the dependencies with this command:
```code
$ pip3 install -r requirements.txt
```

## How to run:
To run the server, use this command:
```code
$ python3 -m server
```
To demo the model, run predict.py like a script
```code
$ python3 -m predict
```

## Adding code
If you import any more python packages, you can update the requirements.txt file by installing the pipreqs package, then running in the project root directory:
```code
$ pipreqs --force .
```
