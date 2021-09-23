import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
from EDSR.model import EDSR
from RRDBNet.model import RRDBNet
from SRResNet.model import SRResNet
from WDSR.model import WDSR
from TherISuRNet.model import TherISuRNet
from torch.nn.functional import interpolate
from piq import psnr, ssim

parser = argparse.ArgumentParser(description="SuperRes Pytorch EVAL")
parser.add_argument("--model", type=str, help="model name")
opt = parser.parse_args()

#change dir here
TEST_LR_DIR = "D:/thermal-images-datasets/FLIR_ADAS_1_3/merged/lr"
TEST_HR_DIR = "D:/thermal-images-datasets/FLIR_ADAS_1_3/merged/"

if opt.model == "SRResNet":
    model = SRResNet()
    print("Model SRResNet is loaded")
elif opt.model == "EDSR":
    model = EDSR(4)
    print("Model EDSR is loaded")
elif opt.model == "WDSR":
    model = WDSR(4)
    print("Model WDSR is loaded")
elif opt.model == "RRDBNet":
    model = RRDBNet(64, 23)
    print("Model RRDBNet is loaded")
elif opt.model == "TherISuRNet":
    model = TherISuRNet(scale=opt.scale)
    print("Model TherISuRNet is loaded")


torch.cuda.empty_cache()

# change dir here
model = model.load_from_checkpoint("EDSR-epoch=88-val_L2loss=248.69.ckpt")
model.eval()
model.cuda()


psnr1 = 0.0
psnr2 = 0.0
ssim1 = 0.0
ssim2 = 0.0
count = 0.0
for name in tqdm(os.listdir(TEST_LR_DIR)):
    lr_image = cv2.imread(TEST_LR_DIR + '/' + name, cv2.IMREAD_UNCHANGED)
    lr_image = np.expand_dims(lr_image, axis=0)
    lr_image = torch.from_numpy(lr_image.astype('float32'))
    lr_image = torch.unsqueeze(lr_image, 0)
    lr_image =  lr_image.cuda()
    
    bicubic = interpolate(lr_image, scale_factor=4, mode='bicubic')
    output = torch.clamp(model(lr_image), max=16383)

    hr_image = cv2.imread(TEST_HR_DIR + '/' + name, cv2.IMREAD_UNCHANGED)
    hr_image = np.expand_dims(hr_image, axis=0)
    hr_image = torch.from_numpy(hr_image.astype('float32'))
    hr_image = torch.unsqueeze(hr_image, 0)
    hr_image = hr_image.cuda()

    psnr1 += psnr(output, hr_image, data_range=16383).detach().cpu().numpy()
    psnr2 += psnr(bicubic, hr_image, data_range=16383).detach().cpu().numpy()
    ssim1 += ssim(output, hr_image, data_range=16383).detach().cpu().numpy()
    ssim2 += ssim(bicubic, hr_image, data_range=16383).detach().cpu().numpy()
    count = count + 1

print("PSNR SR image:", psnr1 / count, "    ", ssim1/count)
print("PSNR BIC image:", psnr2/ count, "    ", ssim2/count)
