import torch
import io
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

classes = ["Organic", "Recycle"]

def cuda_info():
    print(f"pytorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA is not enabled on your system. Please enable it to train the model on the GPU.")
        print("This should still work fine if you are just using the model to predict.")
    else:
        print(f"CUDA is enabled on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

def load_model_script(model_path: str) -> nn.Module:
    print(f"Loading model: {model_path}")
    return torch.jit.load(model_path)

class Predictor():
    def __init__(self, model_path: str, device='cpu', optimizer=None, criterion=nn.CrossEntropyLoss(), learning_rate=0.001):
        """
        model_path must be a valid path to a .pts file (pytorch script)
        optimizer is a torch.optim.Optimizer, if nothing is passed, it defaults to torch.Optim.SGD
        learning_rate should be small if you don't want to mess up the model too much
        if the model is small, device should be cpu (it is faster)
        """
        self.device = device
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("CUDA is not enabled on your system. Please enable it to train the model on the GPU.")
                print("falling back to cpu...")
                self.device = 'cpu'
        elif device == 'cpu':
            print("Using CPU as predictor device.")
        try:
            self.model = load_model_script(model_path).to(self.device)
        except Exception as error:
            print("Could not load model.", error)
            exit(1)
        try:
            if optimizer == None:
                self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate)
                print("Successfully set SGD optimizer")
            else:
                self.optimizer = optimizer
                print("Successfully set custom optimizer")
        except Exception as error:
            print("Could not initialize optimizer.", error)
            exit(1)
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.transformer = transforms.Compose([
            transforms.Resize(size=(self.model.image_width, self.model.image_width)),
            transforms.ToTensor(),
        ])

    def predict(self, image) -> dict:
        """
        image must be a PIL image.
        returns a dict full of result statistics
        dur: float amount of seconds the model took to forward the prediction
        probabilities: a list of two floats, represents the probability of each class, adds up to 1.
        prediction: The class name of the most likely prediction 'Organic' or 'Recyclable'
        """
        print('[PREDICTOR] :: Predicting image...')
        self.model.eval()
        result = {}
        image = image.convert("RGB")
        img_tensor = self.transformer(image).unsqueeze(dim=0).to(self.device)
        with torch.inference_mode(): # don't waste time on training parameters
            start = time.time()
            pred = self.model(img_tensor)
            end = time.time()
        result["dur"] = end - start
        result["probabilities"] = pred.softmax(dim=1).squeeze().tolist()
        result["prediction"] = classes[pred.argmax()]
        return result
    
    def train(self, image, label: str, lr: float = 0.001) -> dict:
        """
        image: image to train on
        label is the correct class of the prediction.
        returned dict:
        same as predict
        loss: a number indicating how badly the model predicted based on the label
        """
        if lr != None:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)

        label_tensor = None
        if label == "Organic":
            label_tensor = torch.Tensor([1, 0]).unsqueeze(dim=0).to(self.device)
        elif label == "Recyclable":
            label_tensor = torch.Tensor([0, 1]).unsqueeze(dim=0).to(self.device)

        print(f'[PREDICTOR] :: Live training image as {label} with learning_rate {lr}')
        image = image.convert("RGB")
        img_tensor = self.transformer(image).unsqueeze(dim=0).to(self.device)
        result = {}
        # self.model.train()
        start = time.time()
        pred = self.model(img_tensor) # forward the prediction
        loss = self.criterion(pred, label_tensor) #TODO: NOT LABEL
        self.optimizer.zero_grad() # reset optimizer
        loss.backward()
        self.optimizer.step() # update the weights
        end = time.time()
        result["loss"] = loss.item()
        result["dur"] = end - start
        result["probabilities"] = pred.softmax(dim=1).squeeze().tolist()
        result["prediction"] = classes[pred.argmax()]
        return result
    
    def explore_predictions(self, dataset_dir="./test_data/", width=3):
        """
        dataset_dir must be a path to a folder which has O and R subdirectories
        each with train and test subdirectories with images.
        NOTE: returns a raw bytesIO buffer of a png image
        """
        self.model.eval()
        start = time.time()
        fig, axes = plt.subplots(width, width, figsize=(8,8))
        test_dataset = torchvision.datasets.ImageFolder(dataset_dir, self.transformer)
        num_images = width*width
        test_loader = DataLoader(dataset=test_dataset, batch_size=num_images, shuffle=True, num_workers=0)
        num_correct = 0
        with torch.inference_mode():
            images, labels = next(iter(test_loader))
            preds = self.model(images.to(self.device))
            pred_list = [classes[x.argmax()] for x in preds]
            loss = self.criterion(preds, labels)

            label_list = [classes[x] for x in labels.to('cpu')]
            image_list = [x.squeeze().permute(1,2,0) for x in images.to('cpu')]
            for i in range(width ** 2):
                row = i // width
                col = i % width
                ax = axes[row, col]
                ax.imshow(image_list[i])
                correct = pred_list[i] == label_list[i]
                num_correct += correct
                color = 'green' if correct else 'red'
                ax.set_title(f"{pred_list[i]}", color=color)
                ax.axis('off')
        plt.suptitle(f"{(num_correct/(width*width))*100:.2f}% Accurate | Loss: {loss.item():.4f}")
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf() # make sure we don't interfere with main thread
        plt.close('all')
        end = time.time()
        print(f"[PREDICTOR] :: exploring predictions on {num_images} images. took {end-start} seconds, {(num_correct/(width*width))*100:.2f}% Accurate | Loss: {loss.item():.4f}")
        return buffer

    def extract_last_layer(self):
        self.model.eval()
        with torch.inference_mode():
            start = time.time()
            first_conv_layer = None
            for name, layer in self.model.named_modules():
                if name == "conv_block_3.0":
                    first_conv_layer = layer
                    break

            if first_conv_layer is not None:
                filters = first_conv_layer.weight.data.cpu().numpy()
                num_filters = filters.shape[0]
                num_cols = 12
                num_rows = num_filters // num_cols + 1
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(4,4))
                for i, ax in enumerate(axes.flatten()):
                    if i < num_filters:
                        filter_img = filters[i]
                        for j in range(3):
                            ax.imshow(filter_img[j], cmap='viridis')
                        ax.axis('off')
                    else:
                        ax.axis('off')
                plt.suptitle(f"Last Conv Layer")
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plt.clf() # make sure we don't interfere with main thread
                plt.close('all')
                end = time.time()
                print(f"[PREDICTOR] :: Extracting last layer kernels took {end-start} seconds")
                return buffer
            print(f"[PREDICTOR] :: ERROR: Could not find the last conv layer")
            return None


def explore_dataset(dataset_dir, width=3):
    """
    dataset_dir must be a path to a folder which has O and R subdirectories each with images.
    """
    fig, axes = plt.subplots(width, width, figsize=(8,8))
    explore_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = torchvision.datasets.ImageFolder(dataset_dir, explore_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i in range(width ** 2):
        image, label = next(iter(test_loader))
        row = i // width
        col = i % width
        ax = axes[row, col]
        ax.imshow(image.squeeze().permute(1,2,0))
        ax.set_title(classes[label.item()])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predictor = Predictor("./models/5-64x64-CNN3L-90.pts", device='cuda')
    # image = Image.open("plastic.webp")
    predictor.explore_predictions("./data/DATASET/TEST", 10)
    print(predictor.predict(image))
