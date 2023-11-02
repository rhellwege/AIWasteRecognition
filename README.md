# AI Waste Recognition
The purpose of this project is to test a convolutional neural network with user inputted images.
The server responds with the neural network's decision of whether the image contains trash or recycling,
and the user can then correct the NN live.

## How to Run
To run the server, run
```code
$ python3 -m server
```
You can then navigate to localhost:3000 in your browser to see the website.

To train the model, make sure you have a graphics card, and have pytorch installed, then run 
```code
$ python3 -m train
```

