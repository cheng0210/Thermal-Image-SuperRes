import torch
import cv2
import argparse
from flirpy.camera.lepton import Lepton
import numpy as np
from EDSR_Quant.model import EDSR
from RRDBNet_Quant.model import RRDBNet
import onnx
import onnxruntime
import tensorrt as trt
from torch2trt import torch2trt, TRTModule
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def thermal_to_color(img):
    # Rescale to 8 bit
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return img_col


parser = argparse.ArgumentParser(description="SuperRes Pytorch")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--ckpt", type=str, help="checkpoint path")
opt = parser.parse_args()
if opt.model is None:
        print("model name?")
elif not torch.cuda.is_available():
    print("cuda is not available")
else:
    print("===> Loading model")
    if opt.model == "EDSR":
        model = EDSR(opt.scale, lr=opt.lr, batch_size=opt.batchSize)
        print("Model EDSR is loaded")
    elif opt.model == "RRDBNet":
        model = RRDBNet(64, 23, scale_factor=opt.scale, lr=opt.lr, batch_size=opt.batchSize)
        print("Model RRDBNet is loaded")

model = model.load_from_checkpoint(opt.ckpt)
model.cuda()
model.eval()

# MUST ADD THIS LINE OF CODE HERE !!!!
quant_nn.TensorQuantizer.use_fb_fake_quant = True

# can be replaced by torch.rand but must have same image size with lepton 160 x 120
with Lepton() as camera:
    x = camera.grab().astype(np.float32)
    x = np.expand_dims(x, axis=0)

x = torch.from_numpy(x)
x = torch.unsqueeze(x, 0).cuda()

y = model(x)


torch.onnx.export(model, x, opt.model+".onnx", opset_version=12, export_params=True,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
onnx_model = onnx.load(opt.model+".onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(opt.model+".onnx")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=0, atol=0)

torch_pred = to_numpy(y)
onnx_pred = ort_outs[0]
torch_pred = torch_pred.reshape((480, 640))
onnx_pred = onnx_pred.reshape((480, 640))
cv2.imshow("torch", thermal_to_color(torch_pred))
cv2.imshow("onnx", thermal_to_color(onnx_pred))
cv2.waitKey()
