import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from piq import SSIMLoss, psnr
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn

def QuantConv(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding=1, bias:bool=False, quantize:bool=True):
    if quantize:
        return quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class ResBlock(nn.Module): 
    def __init__(self, quantize:bool=True):
        super(ResBlock, self).__init__()

        self.conv1 = QuantConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuantConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize)
        self.quant = quantize
        if self.quant:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        if self.quant:
            identity_data = self.residual_quantizer(identity_data)
        output = torch.add(output,identity_data)
        return output 

class EDSR(pl.LightningModule):
    def __init__(self, scale_factor=4, batch_size=16, lr=2e-6, quantize=True):
        super(EDSR, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = lr

        self.save_hyperparameters()

        self.scale_factor = scale_factor

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(ResBlock, 32)

        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.upscale3x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(3),
        )

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        if self.scale_factor == 2:
            out = self.upscale2x(out)
        elif self.scale_factor == 3:
            out = self.upscale3x(out)
        else:
            out = self.upscale4x(out)
        out = self.conv_output(out)
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
        PSNR = psnr(z, y,data_range=16383)
        self.log('psnr', PSNR)
        self.log('val_L1loss', l1loss)
        self.log('val_L2loss', l2loss)