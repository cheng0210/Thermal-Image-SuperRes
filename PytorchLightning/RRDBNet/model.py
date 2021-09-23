import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from piq import SSIMLoss, psnr
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True))

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x

class RRDBNet(pl.LightningModule):
    def __init__(self, filters=64, blocks=23, scale_factor=4, gc=32, batch_size=16, lr=1e-5):
        super(RRDBNet, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = lr

        self.save_hyperparameters()

        self.scale_factor = scale_factor

        self.input_conv = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.rrdb = self.make_layer(ResidualInResidualDenseBlock,blocks)
        self.rrdb_conv = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.upscale3x = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(3),
        )

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=filters, out_channels=filters*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1),
        )



    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.rrdb_conv(self.rrdb(out))
        out = torch.add(out,residual)
        if self.scale_factor == 2:
            out = self.upscale2x(out)
        elif self.scale_factor == 3:
            out = self.upscale3x(out)
        else:
            out = self.upscale4x(out)
        out = torch.mean(self.output_conv(out), 1, True)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
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