"""

SRResNet Model Class

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from piq import SSIMLoss, psnr

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output,identity_data)
        return output

class SRResNet(pl.LightningModule):
    def __init__(self, scale_factor=4, batch_size=16, lr=1e-6):
        super(SRResNet,self).__init__()
        self.batch_size = batch_size
        self.learning_rate = lr

        self.save_hyperparameters()

        self.scale_factor = scale_factor

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.PReLU()
        
        self.residual = self.make_layer(ResBlock, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.upscale3x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(3),
        )
        
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_output1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, stride=1, padding=4, bias=False)

        self.conv_output2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.conv_mid(out)
        out = torch.add(out,residual)
        if self.scale_factor == 2:
            out = self.upscale2x(out)
        elif self.scale_factor == 3:
            out = self.upscale3x(out)
        else:
            out = self.upscale4x(out)
        out = F.relu(self.conv_output1(out))
        out = self.conv_output2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': ExpLR}
        return optim_dict

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self(x)    
        loss = F.l1_loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self(x)
        l1loss = F.l1_loss(z, y)
        l2loss = F.mse_loss(z, y)
        self.log('val_L1loss', l1loss)
        self.log('val_L2loss', l2loss)