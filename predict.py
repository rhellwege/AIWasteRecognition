import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

IMAGE_WIDTH = 64
BATCH_SIZE = 1

MODELS_DIR = "models/"
DATA_DIR = "data/DATASET/"

classes = ["Organic", "Recycle"]

def cuda_info():
    global device
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
    return torch.jit.load(model_path) # no need for the class definition!

class Predictor():
    def __init__(self, model_path: str, device='cpu'):
        """
        model_path must be a valid path to a .pts file (pytorch script)
        if the model is small, device should be cpu (it is faster)
        """
        self.device = device
        try:
            self.model = load_model_script(model_path).to(self.device)
        except:
            print("Could not load model.")
            exit(1)
        self.transformer = transforms.Compose([
            transforms.Resize(size=(self.model.image_width, self.model.image_width)),
            transforms.ToTensor(),
        ])

    def predict(self, image) -> dict:
        """
        image must be a PIL image.
        returns a dict full of result statistics
        """
        result = {}
        img_tensor = self.transformer(image).to(self.device).unsqueeze(dim=0)
        start = time.time()
        pred = self.model(img_tensor)
        end = time.time()
        result["dur"] = end - start
        result["probabilities"] = pred.softmax(dim=1).squeeze().tolist()
        result["prediction"] = classes[pred.argmax()]
        return result

explore_transform = transforms.Compose([
    transforms.ToTensor()
])
def explore_dataset(dataset, width=3):
    fig, axes = plt.subplots(width, width, figsize=(8,8))
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
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

def explore_predictions(model, dataset, width=3):
    model = model.to(device)
    model.eval()
    fig, axes = plt.subplots(width, width, figsize=(8,8))
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    num_correct = 0
    with torch.inference_mode():
        for i in range(width ** 2):
            image, label = next(iter(test_loader))
            label_true = classes[label.item()]
            pred = model(image.to(device))
            pred_label = classes[pred.argmax()]
            row = i // width
            col = i % width
            ax = axes[row, col]
            ax.imshow(image.squeeze().permute(1,2,0))
            correct = pred_label == label_true 
            num_correct += correct
            color = 'green' if correct else 'red'
            ax.set_title(f"{pred_label}", color=color)
            ax.axis('off')
    plt.suptitle(f"{(num_correct/(width*width))*100:.2f}%")
    plt.tight_layout()
    plt.show()

test_transform = transforms.Compose([
    transforms.Resize(size=(IMAGE_WIDTH, IMAGE_WIDTH)),
    # transforms.TrivialAugmentWide(num_magnitude_bins=31), # random rotation, crop, scale, color jittering...
    # transforms.RandomGrayscale(p=1.0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# init_cuda()
#
# test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "TEST"), test_transform)
# train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "TRAIN"), test_transform)
# test_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#
# model = load_model_script("./models/5-64x64-CNN3L-90.pts")
# model = model.to(device)
# model.eval()
# torchinfo.summary(model=model, input_size=[1, 3, IMAGE_WIDTH, IMAGE_WIDTH]) # batch size of 1 (dont overexaggerate)
#
# print(model.image_width)
# explore_predictions(model, test_dataset, 10)
if __name__ == "__main__":
    predictor = Predictor("./models/5-64x64-CNN3L-90.pts")
    image = Image.open("./data/DATASET/TEST/O/O_12568.jpg")
    print(predictor.predict(image))

