import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super(FullyConvNetwork, self).__init__()
        
        # Encoder (Convolutional Layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 3
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)

        # Decoder forward pass
        x = self.decoder(x)

        return x
    