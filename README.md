# AI Waste Recognition
The purpose of this project is to test a convolutional neural network with user inputted images.
The server responds with the neural network's decision of whether the image contains trash or recycling,
and the user can then correct the NN live.

## Getting Started
First, make sure you have python and pip installed. Preferably python 3.11.
Install pytorch for your system with CUDA enabled (follow this link https://pytorch.org/get-started/locally/).
If you have a CUDA enabled GPU and are using windows, you can run this command:
```code
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Then, you can install dependencies for the server.
```code
$ pip3 install -r requirements.txt
```

## How to run:
To train the model, use this command:
```code
$ python3 -m train
```

To run the server, use this command:
```code
$ python3 -m server
```
Then, you should be able to navigate to localhost:5000 in your browser to see the website.

## Adding code
If you import any more python packages, you can update the requirements.txt file by installing the pipreqs package, then running in the project root directory:
```code
$ pipreqs --force .
```
