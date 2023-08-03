import torch
import torch.nn as nn
from torch import autograd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage.color as scicolor

from sklearn.model_selection import train_test_split
#from model.CNN import *

from utils import *
import yaml
import argparse

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")

class PreviewDataset(Dataset):
    def __init__(self, root="../destijl_dataset/rgba_dataset/", transform=None, test=False, color_space="RGB", input_color_space="RGB", is_classification=False):

        self.test = test
        self.sample_filenames = os.listdir(root+"00_preview_cropped")
        self.transform = transform
        self.img_dir = root
        self.color_space = color_space
        self.is_classification = is_classification
        self.input_color_space = input_color_space

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
        img_path = os.path.join(self.img_dir, "00_preview_cropped/" + self.sample_filenames[idx])

        image = Image.open(img_path)
        if self.input_color_space == "CIELab":
            image = scicolor.rgb2lab(image)
           
        bg_path = os.path.join("../destijl_dataset/01_background/" + self.sample_filenames[idx])
        color = self.kmeans_for_bg(bg_path)[0]

        if self.color_space == "CIELab" and self.input_color_space == "RGB":
            target_color = torch.squeeze(torch.tensor(RGB2CIELab(color.astype(np.int32))))
        elif self.color_space == "CIELab" and self.input_color_space == "CIELab":
            # Normalize the cielab space.
            # Add 127 to make the a and b between (0, 255)
            image = image + [0, 127, 127]
            image = image / [100, 255, 255]
            image = torch.from_numpy(image)
            target_color = torch.squeeze(torch.tensor(RGB2CIELab(color.astype(np.int32))))
            target_color += torch.Tensor([0, 127, 127])
            target_color /= torch.Tensor([100, 255, 255])
        else:
            target_color = torch.squeeze(torch.tensor(color))

        if self.is_classification:
            target_color = [torch.zeros(256), torch.zeros(256), torch.zeros(256)]
            target_color[0][color[0]] = 1
            target_color[1][color[1]] = 1
            target_color[2][color[2]] = 1

        if self.transform:
            if image.shape[0] != 3:
                image = image.reshape(-1, image.shape[0], image.shape[1]).type("torch.FloatTensor")
            image = self.transform(image)

        return image, target_color
    
    def kmeans_for_bg(self, bg_path):
        image = cv2.imread(bg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_colors = 1

        # Apply KMeans to the text area
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        palette = np.asarray(palette, dtype=np.int64) # RGB

        return palette

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
map_outputs = config["map_outputs"]
color_space = config["color_space"]
input_color_space = config["input_color_space"]
is_classification = config["is_classification"]
input_size = config["input_size"]

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    #transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

model = model_switch_CNN(model_name, loss_function, map_outputs).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

criterion = nn.MSELoss()
if loss_function == "MSE":
    criterion = nn.MSELoss()
elif loss_function == "Cross-Entropy":
    criterion = nn.CrossEntropyLoss()
elif loss_function == "MAE":
    criterion = nn.L1Loss()


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

train_dataset = PreviewDataset(transform=transform, color_space=color_space, input_color_space=input_color_space, is_classification=is_classification)
test_dataset = PreviewDataset(transform=transform, test=True, color_space=color_space, input_color_space=input_color_space, is_classification=is_classification)

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
    loss = 0
    for i in range(batch_size):
        if loss_function != "CIELab":
            loss += criterion(out[i], color[i]/255)
        elif loss_function == "CIELab" and color_space == "CIELab":
            #loss += colormath_CIE2000(out[i], color[i], normalized=True)
            loss = criterion(out[i], color[i])
    loss /= out.shape[0]
    loss.backward()
    optimizer.step()
    return  loss, out

def test(data, color):
    model.eval()
    out = model(data)

    if loss_function != "CIELab":
        loss = criterion(out[0], color[0]/255)
    else:
        loss = colormath_CIE2000(out[0], color[0])
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
