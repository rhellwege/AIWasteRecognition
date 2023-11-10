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
    return torch.jit.load(model_path)

class Predictor():
    def __init__(self, model_path: str, device='cpu'):
        """
        model_path must be a valid path to a .pts file (pytorch script)
        if the model is small, device should be cpu (it is faster)
        """
        self.device = device
        if device == 'cuda':
            # cuda_info()
            if not torch.cuda.is_available():
                print("CUDA is not enabled on your system. Please enable it to train the model on the GPU.")
                print("falling back to cpu...")
                self.device = 'cpu'
        try:
            self.model = load_model_script(model_path).to(self.device)
        except Exception as error:
            print("Could not load model.", error)
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
        self.model.eval()
        result = {}
        image = image.convert("RGB")
        img_tensor = self.transformer(image).to(self.device).unsqueeze(dim=0)
        with torch.inference_mode(): # don't waste time on training parameters
            start = time.time()
            pred = self.model(img_tensor)
            end = time.time()
        result["dur"] = end - start
        result["probabilities"] = pred.softmax(dim=1).squeeze().tolist()
        result["prediction"] = classes[pred.argmax()]
        return result
    
    def explore_predictions(self, dataset_dir, width=3):
        """
        dataset_dir must be a path to a folder which has O and R subdirectories each with train and test subdirectories with images.
        """
        self.model.eval()
        fig, axes = plt.subplots(width, width, figsize=(8,8))
        test_dataset = torchvision.datasets.ImageFolder(dataset_dir, self.transformer)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)
        num_correct = 0
        with torch.inference_mode():
            for i in range(width ** 2):
                image, label = next(iter(test_loader))
                label_true = classes[label.item()]
                pred = self.model(image.to(self.device))
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

    def print_arch(self):
        torchinfo.summary(model=self.model, input_size=[1, 3, self.model.image_width, self.model.image_width])

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
