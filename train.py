# note: some of the filenames are too long in the csv, so I just deleted them.
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
# import torchinfo
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
import signal

print("This script is intended to be run on a machine with cuda support to")
print("train models once and use them afterwards.")

# TODO: add loss curve plots and more information about the model in the models directory

MODELS_DIR = "models/"
DATA_DIR = "data/DATASET/"

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

device = ''
def init_cuda():
    global device
    print(f"pytorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA is not enabled on your system. Please enable it to train the model on the GPU.")
        print("Download the right version of pytorch")
        device = 'cpu'
    else:
        print(f"CUDA is enabled on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        device = 'cuda'

def generate_model_filename(model: nn.Module, epochs: int, image_width: int, val_acc: float) -> str:
    """
    expects acc to be a percentage (0-100)
    """
    return f"{epochs}-{image_width}x{image_width}-{model.__class__.__name__}-{int(val_acc)}.pts"

def save_model(model: nn.Module, epochs: int, image_width: int, val_acc: float):
    model=model.to('cpu')
    model_path = os.path.join(MODELS_DIR , generate_model_filename(model, epochs,  image_width, val_acc))
    print(f"Saving model state_dict as {model_path}")
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)

def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct / len(y_pred)) * 100

# https://www.kaggle.com/datasets/techsash/waste-classification-data/download?datasetVersionNumber=1
#
# load datasets

class CPUModel(nn.Module):
    def __init__(self, 
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 dropout_p: float,
                 image_width: int):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_shape = output_shape
        self.dropout_p = dropout_p
        self.image_width = image_width
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units*2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units*4),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=hidden_units*4),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(in_features=(hidden_units*4*int((self.image_width/8)*(self.image_width/8))), out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units//2),
            nn.BatchNorm1d(num_features=hidden_units//2),
            nn.Dropout(dropout_p),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units//2, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(
               self.conv_block_3(
               self.conv_block_2(
               self.conv_block_1(x)
        )))

init_cuda()
device = 'cpu'
print(f"Using {device}")

# Hyper Parameters
IMAGE_WIDTH = 64
AUGMENT_INTENSITY = 3
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_UNITS = 32
# INITIAL_LEARNING_RATE = 0.1
INITIAL_LEARNING_RATE = 0.01
DROPOUT_P = 0.3 # normally 0.5

train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(IMAGE_WIDTH, IMAGE_WIDTH)),
    v2.TrivialAugmentWide(num_magnitude_bins=AUGMENT_INTENSITY), # random rotation, crop, scale, color jittering...
    v2.RandomGrayscale(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(),
    v2.ToDtype(torch.float32, scale=True),
])
train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "TRAIN"), transform=train_transform)

test_transform = v2.Compose([ # this can be used with single predictions as well
    v2.ToImage(),
    v2.Resize(size=(IMAGE_WIDTH, IMAGE_WIDTH)),
    #v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "TEST"), test_transform)

# setup dataloaders
train_loader      = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validation_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = CPUModel(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=2, dropout_p=DROPOUT_P, image_width=IMAGE_WIDTH).to(device)

# torchinfo.summary(model=model, input_size=[1, 3, IMAGE_WIDTH, IMAGE_WIDTH])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=INITIAL_LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
#
# setup save on quit:
def quit_handler(sig, frame):
    save_model(model=model, epochs=EPOCHS, image_width=IMAGE_WIDTH, val_acc=val_acc)
    sys.exit(0)
signal.signal(signal.SIGINT, quit_handler)

train_loss = None
train_acc = 0
val_loss=None
val_acc=0.0
for epoch in range(EPOCHS):
    train_loop = tqdm(train_loader, leave=False) 
    train_loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
    train_loop.set_postfix(lr=optimizer.param_groups[0]['lr'])
    train_loss = 0 
    train_acc = 0
    train_start = time.time()
    model.train()
    for data, labels in train_loop:
        data, labels = data.to(device), labels.to(device)
        pred = model(data)
        train_acc += accuracy_fn(y_pred=pred.argmax(dim=1), y_true=labels)
        loss = criterion(pred, labels)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # update once per batch!
    train_end = time.time()
    train_acc /= len(train_loader)
    train_loss /= len(train_loader)
    val_loss, val_acc = 0, 0
    validation_start = time.time()
    model.eval()
    with torch.inference_mode():
        for data_val, labels_val in validation_loader: # test loop
            data_val, labels_val = data_val.to(device), labels_val.to(device)
            val_pred = model(data_val)
            val_loss += criterion(val_pred, labels_val) # bcewithlogitsloss
            val_acc += accuracy_fn(y_true=labels_val, y_pred=val_pred.argmax(dim=1))
        val_loss /= len(validation_loader)
        val_acc /= len(validation_loader)
        if val_acc >= 80:
            save_model(model=model.to('cpu'), epochs=epoch, image_width=IMAGE_WIDTH, val_acc=val_acc)
    # scheduler.step(val_loss) # update the learning rate
    validation_end = time.time()
    print(f"Epoch {epoch} Complete | Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}% | Test loss: {val_loss:.4f}, Test acc: {val_acc:.2f}% || Training Dur: {train_end-train_start:.2f}s | Validation Dur: {validation_end-validation_start:.2f}s")

save_model(model=model.to('cpu'), epochs=EPOCHS, image_width=IMAGE_WIDTH, val_acc=val_acc)
