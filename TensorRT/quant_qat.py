import datetime
import os
import sys
import time
import collections
import argparse
import torch
import torch.utils.data
from torch import nn
import h5py
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from utils import customDataset
from EDSR_Quant.model import EDSR
from RRDBNet_Quant.model import RRDBNet
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#model = EDSR(4)
#model = model.load_from_checkpoint("EDSR_Quant.ckpt", lr=5e-6)

model = RRDBNet(4)
model = model.load_from_checkpoint("RRDBNet_Quant.ckpt", lr=5e-6)

print("===> Loading datasets")
hf = h5py.File(os.environ["TMPDIR"]+"/FLIR16_x"+str(4)+".hdf5", 'r')
inputs = hf.get("input")
labels = hf.get("label")

kf = KFold(n_splits=10)
kf_index = 0

for trainFold_indexes, testFold_indexes in kf.split(inputs):

    if kf_index < 6:
        kf_index+=1
        continue

    print("===> Start K-FOLD Index " + str(kf_index))

    train_dataset = customDataset(inputs[trainFold_indexes], labels[trainFold_indexes])
    test_dataset = customDataset(inputs[testFold_indexes], labels[testFold_indexes])

    training_dataloader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=12)
    testing_dataloader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=12)

    break

#save_path = "EDSR_Quant" + "/x"+str(4)+"_checkpoint-kfoldIndex-" + str(kf_index) + '/'

save_path = "RRDBNet_Quant" + "/x"+str(4)+"_checkpoint-kfoldIndex-" + str(kf_index) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

checkpoint_callback = ModelCheckpoint(
    monitor="val_L2loss",
    dirpath=save_path,
    filename="RRDBNet_Quant" + "-{epoch:02d}-{val_L2loss:.2f}",
    save_top_k=3,
    mode="min",
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(gpus=-1, accelerator="ddp", default_root_dir=save_path, max_epochs=100, callbacks=[checkpoint_callback, lr_monitor])

trainer.fit(model, training_dataloader, testing_dataloader)