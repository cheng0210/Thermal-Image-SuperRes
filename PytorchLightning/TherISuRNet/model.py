import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from piq import SSIMLoss, psnr
from torch.nn.functional import interpolate

class ChannelAttentionModule(nn.Module):
    def __init__(self, fea):
        super(ChannelAttentionModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channels=fea, out_channels=fea//4, kernel_size=1, stride=1, padding=1//2)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=fea//4, out_channels=fea, kernel_size=1, stride=1, padding=1//2)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        identity = x
        out = self.avgpool(x)
        out = self.conv1(out)
        out = self.elu(out)
        out = self.sig(self.conv2(out))
        out = torch.mul(identity, out)
        return out

class ConcatModule(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(ConcatModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=3//2)
        self.conv2 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1, padding=1//2)
        self.conv3 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1, padding=1//2)
        self.elu = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.elu(out)
        out = self.conv3(out)
        return out

class ConcatBlock(nn.Module):
    def __init__(self, fea):
        super(ConcatBlock, self).__init__()
        self.cm1 = ConcatModule(in_fea=fea, out_fea=fea)
        self.cm2 = ConcatModule(in_fea=2*fea, out_fea=fea)
        self.cm3 = ConcatModule(in_fea=2*fea, out_fea=fea)
        self.ca1 = ChannelAttentionModule(fea)
        self.ca2 = ChannelAttentionModule(fea)
        self.ca3 = ChannelAttentionModule(fea)
    def forward(self, x):
        x1 = x

        x = self.cm1(x)
        x = self.ca1(x)

        x = torch.cat([x, x1], 1)
        x = self.cm2(x)
        x = self.ca2(x)
        x = torch.cat([x, x1], 1)
        x = self.cm3(x)
        x = self.ca3(x)
        x = torch.cat([x, x1], 1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6*out_fea, out_channels=out_fea, kernel_size=1, stride=1)

        self.concatBlock1 = ConcatBlock(out_fea)
        self.concatBlock2 = ConcatBlock(out_fea)
        self.concatBlock3 = ConcatBlock(out_fea)

    
    def forward(self, x):

        x = self.conv1(x)

        identity = x
        x1 = self.concatBlock1(x)
        x2 = self.concatBlock1(x)
        x3 = self.concatBlock1(x)

        x = torch.cat([x1,x2], 1)
        x = torch.cat([x, x3], 1)

        x = self.conv2(x)
        x = torch.add(x, identity)

        return x

class GlobalResLearning(nn.Module):
    def __init__(self, scale, fea):
        super(GlobalResLearning, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=fea, kernel_size=1, stride=1)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=fea, out_channels=fea//4, kernel_size=1, stride=1)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=fea//4, out_channels=1, kernel_size=1, stride=1)
        self.elu3 = nn.ELU()
    def forward(self, x):
        out = interpolate(x, scale_factor=self.scale, mode='bicubic')
        out = self.conv1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = self.conv3(out)
        out = self.elu3(out)
        return out

class LFE(nn.Module):
    def __init__(self, fea):
        super(LFE, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=fea, kernel_size=7, stride=1, padding=7//2)
        self.elu = nn.ELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.elu(x)
        return x

class HFE1(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(HFE1, self).__init__()
        self.res1 = ResBlock(in_fea=in_fea, out_fea=out_fea)
        self.res2 = ResBlock(in_fea=2*in_fea, out_fea=out_fea)
        self.res3 = ResBlock(in_fea=2*in_fea, out_fea=out_fea)
        self.res4 = ResBlock(in_fea=2*in_fea, out_fea=out_fea)
        self.conv1 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=2*out_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=3//2)

        self.upsample_conv = nn.Conv2d(in_channels=out_fea, out_channels=4*out_fea, kernel_size=3, stride=1, padding=3//2)
        self.upsample_elu = nn.ELU()
        self.upsample = nn.PixelShuffle(2)
    
    def forward(self, x):
        identity = x
        x = self.res1(x)
        x = self.conv1(x)
        x = torch.cat([x, identity], 1)
        x = self.res2(x)
        x = self.conv2(x)
        x = torch.cat([x, identity], 1)
        x = self.res3(x)
        x = self.conv3(x)
        x = torch.cat([x, identity], 1)
        x = self.res4(x)
        x = self.conv4(x)
        x = torch.cat([x, identity], 1)
        x = self.conv5(x)
        x = torch.add(x, identity)

        x = self.upsample_conv(x)
        x = self.upsample_elu(x)
        x = self.upsample(x)

        return x

class HFE2(nn.Module):
    def __init__(self, scale, in_fea, out_fea):
        super(HFE2, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=1, stride=1)

        self.res1 = ResBlock(in_fea=in_fea, out_fea=out_fea//2)
        self.res2 = ResBlock(in_fea=3*in_fea//2, out_fea=out_fea//2)
        self.conv1 = nn.Conv2d(in_channels=out_fea//2, out_channels=out_fea//2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_fea//2, out_channels=out_fea//2, kernel_size=1, stride=1)

        self.conv3 = nn.Conv2d(in_channels=3*out_fea//2, out_channels=out_fea, kernel_size=3, stride=1, padding=3//2)

        self.upsample_conv = nn.Conv2d(in_channels=out_fea, out_channels=4*out_fea, kernel_size=3, stride=1, padding=3//2)
        self.upsample_elu = nn.ELU()
        self.upsample = nn.PixelShuffle(2)
    
    def forward(self, x):
        x = self.conv(x)
        identity = x
        x = self.res1(x)
        x = self.conv1(x)
        x = torch.cat([x, identity], 1)

        x = self.res2(x)
        x = self.conv2(x)
        x = torch.cat([x, identity], 1)

        x = self.conv3(x)
        x = torch.add(x, identity)

        if self.scale is 4:
            x = self.upsample_conv(x)
            x = self.upsample_elu(x)
            x = self.upsample(x)

        return x


class TherISuRNet(nn.Module):
    def __init__(self, scale=4, batch_size=16, lr=1e-6):
        super(TherISuRNet, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = lr
        self.save_hyperparameters()
        
        self.lfe = LFE(fea=64)
        self.hfe1 = HFE1(in_fea=64, out_fea=64)
        self.hfe2 = HFE2(scale=scale, in_fea=64, out_fea=64)
        self.grl = GlobalResLearning(scale=scale, fea=64)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=3//2)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1,padding=3//2)
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.lfe(x)
        x = self.hfe1(x)
        x = self.hfe2(x)
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        grl = self.grl(identity)
        x = torch.add(x, grl)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
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