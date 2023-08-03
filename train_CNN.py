import torch
import torch.nn as nn
from torch import autograd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from cnn_dataset import *

#from model.CNN import *

from utils import *
import yaml
import argparse

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/confCNN.yaml", help="Path to the config file.")
args = parser.parse_args()
config_file = args.config_file

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

## Basic Training Parameters ##
step_size = config["step_size"]
num_epoch = config["num_epoch"]
batch_size = config["batch_size"]
lr = config["lr"]
weight_decay = config["weight_decay"]
model_name = config["model_name"]
device = config["device"]

## Neural Network Parameters ##
loss_function = config["loss_function"]
out_features = config["out_features"]
color_space = config["color_space"]
input_color_space = config["input_color_space"]
is_classification = config["is_classification"]
input_size = config["input_size"]
normalize_rgb = config["normalize_rgb"]
normalize_cielab = config["normalize_cielab"]


transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.Normalize((0.5,), (0.5,))
    ])

model = model_switch_CNN(model_name, out_features).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

if loss_function == "MSE":
    criterion = nn.MSELoss()
elif loss_function == "Cross-Entropy":
    criterion = nn.CrossEntropyLoss()
elif loss_function == "MAE":
    criterion = nn.L1Loss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

train_dataset = PreviewDataset(transform=transform, 
                               color_space=color_space, 
                               input_color_space=input_color_space,
                               normalize_rgb=normalize_rgb,
                               normalize_cielab=normalize_cielab)

test_dataset = PreviewDataset(transform=transform, 
                              test=True, 
                              color_space=color_space, 
                              input_color_space=input_color_space,
                              normalize_rgb=normalize_rgb,
                              normalize_cielab=normalize_cielab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

save_path = "../CNN_models/" + model_name + "/weights"
loss_path =  "../CNN_models/" + model_name + "/losses"
log_path = "../CNN_models/"+model_name

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(loss_path):
    os.makedirs(loss_path)

# Copy yaml file to the model folder
with open("../CNN_models/"+model_name+"/conf.yaml", "w+") as wf:
    yaml.dump(config, wf)

def train(data, color):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    if out_features == 1:
        loss = criterion(out[:, 0], color[:, 0])
    else:
        if loss_function != "CIELab":
            loss = criterion(out, color)
        else:
            loss = 0
            for i in range(batch_size):
                loss += colormath_CIE2000(out[i], color[i])
            loss /= out.shape[0]

    loss.backward()
    optimizer.step()
    return loss, out

def test(data, color):
    model.eval()
    out = model(data)
    if out_features == 1:
        loss = criterion(out[:, 0], color[:, 0])
    else:
        if loss_function != "CIELab":
            loss = criterion(out, color)
        else:
            loss = colormath_CIE2000(out, color)
    return loss, out

def classification_train(data, color):
    model.train()
    optimizer.zero_grad()
    r, g, b = model(data)
    loss_r = criterion(r, color[0].to(device))
    loss_g = criterion(g, color[1].to(device))
    loss_b = criterion(b, color[2].to(device))

    loss = loss_r + loss_g + loss_b

    loss.backward()
    optimizer.step()
    return  loss

def classification_test(data, color):
    model.eval()
    r, g, b = model(data)
    
    loss_r = criterion(r, color[0].to(device))
    loss_g = criterion(g, color[1].to(device))
    loss_b = criterion(b, color[2].to(device))

    loss = loss_r + loss_g + loss_b

    return  loss

train_losses = []
val_losses = []

best_loss = 100000
for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count = 0

    num_batches = len(train_loader)
    
    for input_data, target_color in train_loader:
        if is_classification:
            loss = classification_train(input_data.to(device), target_color)
        else:
            loss, out = train(input_data.to(device), target_color.to(device))
        total_train_loss += loss.item()

    for input_data, target_color in test_loader:
        if is_classification:
            loss = classification_test(input_data.to(device), target_color)
        else:
            loss, out = test(input_data.to(device), target_color.to(device))
        total_val_loss += loss.item()
        
    val_loss = total_val_loss/len(test_dataset)
    train_loss = total_train_loss/num_batches
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", train_loss)
    print("Test loss: ", val_loss)

    save_plot(train_losses=train_losses, val_losses=val_losses, loss_type=loss_function, loss_path=loss_path)
    if val_loss < best_loss:
        best_loss = val_loss
        model = model.cpu()
        best_model = model.state_dict()
        model = model.to(device)
        save_model(state_dict=best_model, train_losses=train_losses, val_losses=val_losses, epoch=epoch, save_path=save_path)

    lr_scheduler.step()
