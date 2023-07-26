import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

class Autoencoder(torch.nn.Module):
    def __init__(self, num_channels=3, c_hid=16, latent_dim=256):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(num_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 256x256 => 128x128
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
           )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.linear = nn.Sequential(
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(16 * 16 * c_hid, num_channels*latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                num_channels, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, num_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 32x32 => 64x64
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        linear = self.linear(encoded)
        x = linear.reshape(linear.shape[0], -1, 16, 16)
        decoded = self.decoder(x)
        return decoded
    
class FinetuneResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.fc = nn.Linear(512, 256)
        for param in self.pretrained_model.layer4.parameters():
            param.requires_grad = True