import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import color, io
from colormath.color_objects import sRGBColor, HSVColor, LabColor, LCHuvColor, XYZColor, LCHabColor, AdobeRGBColor
from colormath.color_conversions import convert_color

from model.GNN import *
from dataset import GraphDestijlDataset

from utils import *
import logging
'''
    Take small test, very small, Overfit, See model is corrext
    nn embedding to 1000 thousand or close to that
    Grey might have Minimum distance to evetutjing else, Penalize grey based on the saturation
'''

model_name = "processed_hsv"
save_path = "../weights/" + model_name

loss_path = f'log/log_'+model_name+'.log'

def save_model(state_dict, train_losses, val_losses, epoch):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best.pth"),
    )

def save_plot(train_losses, val_losses, loss_type):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + loss_type + ".png"), dpi=300)
    plt.cla()

def CIELab2RGB(palette):
    obj_palette = []
    for color in palette:
        color = LabColor(*color)
        color = list(convert_color(color, sRGBColor, through_rgb_type=AdobeRGBColor).get_value_tuple())
        obj_palette.append(color)
    return obj_palette

logging.basicConfig(
    filename=f'log/log_'+model_name+'.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M-%S', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
seed = 42
logger.info("#" * 100)

num_epoch = 1000
batch_size = 16
model = ColorGNNEmbedding(feature_size=1005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

train_dataset = GraphDestijlDataset(root='../destijl_dataset/')
test_dataset = GraphDestijlDataset(root='../destijl_dataset/', test=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def train(data, target_color, node_to_mask):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index.long(), data.edge_attr) 
     # Perform a single forward pass.
    #loss = criterion(out[node_to_mask, :]*255, target_color*255)    
    l, a, b, loss = CIELab_distance2(out[node_to_mask, :][0], target_color[0])
    loss2 = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    # loss = torch.tensor([loss], requires_grad=True)  # Derive gradients.
    # count=0
    # for param in model.parameters():
    #     print(count)
    #     count+=1
    #     print(param)
    loss2.backward()
    optimizer.step()  # Update parameters based on gradients.
    return  l, a, b, loss2, out, out[node_to_mask, :][0], target_color[0]

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)  # Perform a single forward pass.
    l, a, b, loss = CIELab_distance2(out[node_to_mask, :][0], target_color[0])  # Compute the loss solely based on the training nodes.
    loss2 = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    #loss = criterion(out[node_to_mask, :]*255, target_color*255)
    return l, a, b, loss2, out

for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count=0
    best_loss = 1000
    for input_data, target_color, node_to_mask in train_loader:
        l, a, b, loss, out, pred, target = train(input_data, target_color, node_to_mask)
        if count == 0:
            print("IMAGE LOSS")
            print(loss.item())
            print("l: ", l.item(), "a: ", a.item(), "b: ", b.item())
            print("Pred and target")
            print(pred)
            print(target)
        prediction = out[node_to_mask, :]
        other_colors = input_data.y.clone()
        other_colors = torch.cat([other_colors[0:node_to_mask, :], other_colors[node_to_mask+1:, :]])
        # prediction in cielab
        current_palette = torch.cat([other_colors, prediction, target_color]).type(torch.float32).detach().numpy()
        sns.palplot(CIELab2RGB(current_palette))
        plt.savefig("palettes/color_"+str(count)+".jpg")
        plt.close()
        count +=1


        total_train_loss += loss.item()

    for input_data, target_color, node_to_mask in test_loader:
        l, a, b, loss, out = test(input_data, target_color, node_to_mask)
        total_val_loss += loss.item()

    logger.info("------- Epoch " + str(epoch) + " -------")
    logger.info("Train loss: " + str(total_train_loss/len(train_dataset))) 
    logger.info("Test loss: " + str(total_val_loss/len(test_dataset)))

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", total_train_loss/len(train_dataset))
    print("Test loss: ", total_val_loss/len(test_dataset))

    lr_scheduler.step()
