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

class FinetuneResNet18_classify(nn.Module):
    def __init__(self, freeze_resnet=True):
        super().__init__()

        "Classify the color"

        self.pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for param in self.pretrained_model.parameters():
            param.requires_grad_ = False
            
        self.pretrained_model.fc = nn.Linear(in_features=512, out_features=1024)

        self.color_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=768),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.color_head(x)
        r = self.softmax(x[:, :256])
        g = self.softmax(x[:, 256:512])
        b = self.softmax(x[:, 512:])
        return r, g, b
    
class ResNet18(nn.Module):
    def __init__(self, freeze_resnet=True, map_outputs="RGB"):
        super().__init__()

        """
            Just map to interval
        """
        self.pretrained_model = resnet18(weights=None)
        self.map_outputs = map_outputs

        for param in self.pretrained_model.parameters():
            param.requires_grad_ = True
            
        self.pretrained_model.fc = nn.Linear(in_features=512, out_features=256)

        self.color_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=3),
        )

        if self.map_outputs == "CIELab":
            self.l_activation = nn.Sigmoid()
            self.a_activation = nn.Tanh()
            self.b_activation = nn.Tanh()

        elif self.map_outputs == "RGB":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.color_head(x)
        if self.map_outputs == "CIElab":
            x[:, 0] = self.l_activation(x[:, 0]) * 100
            x[:, 1] = self.a_activation(x[:, 1]) * 127
            x[:, 2] = self.b_activation(x[:, 2]) * 127

        elif self.map_outputs == "RGB":
            x = x
        return x
    

class ColorCNN(nn.Module):
    def __init__(self, num_channels=3, c_hid=16, out_feature=3, reverse_normalize_output=True):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(num_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 512x512 -> 256x256
            nn.BatchNorm2d(num_features=c_hid),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2), # 256x256 -> 128x128
            nn.BatchNorm2d(num_features=2*c_hid),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.BatchNorm2d(num_features=2*c_hid),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            nn.BatchNorm2d(num_features=c_hid),
            nn.ReLU(),
            nn.Conv2d(c_hid, 8, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
           )
        
        self.color_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*16*8, out_features=out_feature)
        )

        self.reverse_normalize_output = reverse_normalize_output
        #self.addition = Addition()
        #self.multiplication =  torch.Tensor([100, 255, 255]).to("cuda:1")

    def forward(self, x):
        x = self.encoder(x)
        x = self.color_head(x)
        return x
    
class Addition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input += torch.Tensor([0, 127, 127]).to("cuda:1")
        return input
    
class ColorCNNBigger(nn.Module):
    def __init__(self, num_channels=3, c_hid=16):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(num_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 512x512 -> 256x256
            nn.BatchNorm2d(num_features=c_hid),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2), # 256x256 -> 128x128
            nn.BatchNorm2d(num_features=4*c_hid),
            nn.ReLU(),
            nn.Conv2d(4 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.BatchNorm2d(num_features=c_hid),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid//2, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.BatchNorm2d(num_features=c_hid//2),
           )
        
        self.color_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*32*8, out_features=16*16*4),
            nn.Linear(in_features=16*16*4, out_features=3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.color_head(x)
        return x

