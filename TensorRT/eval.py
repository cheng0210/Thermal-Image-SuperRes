import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
from piq import psnr, ssim
import tensorrt as trt
from torch2trt import torch2trt, TRTModule

parser = argparse.ArgumentParser(description="SuperRes Pytorch")
parser.add_argument("--trt", type=str, help="trt engine file path")

opt = parser.parse_args()

logger = trt.Logger(trt.Logger.INFO)
with open(opt.trt, "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())

model_all_names = []
for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    model_all_names.append(name)
    shape = engine.get_binding_shape(idx)
    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)

trt_model = TRTModule(engine, ["input"], ["output"])


TEST_LR_DIR = "/media/jacob/Elements/thermal-images-datasets/FLIR_ADAS_1_3/merged/lr"
TEST_HR_DIR = "/media/jacob/Elements/thermal-images-datasets/FLIR_ADAS_1_3/merged/"
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
    
    output = torch.clamp(trt_model(lr_image), max=16383)
    output = torch.squeeze(output, 0)

    hr_image = cv2.imread(TEST_HR_DIR + '/' + name, cv2.IMREAD_UNCHANGED)
    hr_image = np.expand_dims(hr_image, axis=0)
    hr_image = torch.from_numpy(hr_image.astype('float32'))
    hr_image = torch.unsqueeze(hr_image, 0)
    hr_image = hr_image.cuda()

    psnr1 += psnr(output, hr_image, data_range=16383).detach().cpu().numpy()
    ssim1 += ssim(output, hr_image, data_range=16383).detach().cpu().numpy()
    count = count + 1

print("PSNR SR image:", psnr1 / count, "    ", ssim1/count)
print("PSNR BIC image:", psnr2/ count, "    ", ssim2/count)