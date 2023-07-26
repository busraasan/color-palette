import torch
import torch.nn as nn
from torch import autograd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.model_selection import train_test_split
from model.CNN import Autoencoder

from utils import *

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

class PreviewDataset(Dataset):
    def __init__(self, root="../destijl_dataset/00_preview", transform=None, test=False):

        self.test = test
        self.sample_filenames = os.listdir(root)
        self.transform = transform
        self.img_dir = root

        self.train_filenames, self.test_filenames = train_test_split(self.sample_filenames,
                                                                     test_size=0.2, 
                                                                     random_state=42) 
    def __len__(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)
        
    def __getitem__(self, idx):
        path_idx = "{:04d}".format(idx)
        img_path = os.path.join(self.img_dir, path_idx+".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

batch_size = 16
lr=2e-3
weight_decay=1e-4
step_size=10
device = "cuda:3"
num_epoch = 100
model_name = "CNNAutoencoder"

model = Autoencoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

train_dataset = PreviewDataset(transform=transform)
test_dataset = PreviewDataset(transform=transform, test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_path = "../CNN_models/" + model_name + "/weights"
loss_path =  "../CNN_models/" + model_name + "/losses"
log_path = "../CNN_models/"+model_name

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(loss_path):
    os.makedirs(loss_path)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = 0
    loss = criterion(out, data)
    loss.backward()
    optimizer.step()
    return  loss, out

def test(data):
    model.eval()
    out = model(data)
    loss = criterion(out, data)
    return loss, out

train_losses = []
val_losses = []

best_loss = 100000
for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count = 0

    num_batches = len(train_loader)
    
    for input_data in train_loader:
        loss, out = train(input_data.to(device))
        total_train_loss += loss.item()

    for input_data in test_loader:
        loss, out = test(input_data.to(device))
        total_val_loss += loss.item()

    val_loss = total_val_loss/len(test_dataset)
    train_loss = total_train_loss/num_batches
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", train_loss)
    print("Test loss: ", val_loss)

    save_plot(train_losses=train_losses, val_losses=val_losses, loss_type="CIE2000", loss_path=loss_path)
    if val_loss < best_loss:
        best_loss = val_loss
        model = model.cpu()
        best_model = model.state_dict()
        model = model.to(device)
        save_model(state_dict=best_model, train_losses=train_losses, val_losses=val_losses, epoch=epoch, save_path=save_path)

    lr_scheduler.step()