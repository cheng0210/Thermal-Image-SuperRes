import argparse
import os
import torch
import torch.profiler
from torch.utils.data import DataLoader
from EDSR.model import EDSR
from RRDBNet.model import RRDBNet
from SRResNet.model import SRResNet
from WDSR.model import WDSR
from TherISuRNet.model import TherISuRNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import customDataset
from sklearn.model_selection import KFold
import h5py

parser = argparse.ArgumentParser(description="SuperRes Pytorch")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-5")
#parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--kfoldIndex", default=0, type=int, help="start kfold valid index 0 ~ 9")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--scale", type=int, default=4, help="scale factor, Default: 4")

opt = parser.parse_args()

print("===> Loading datasets")
hf = h5py.File(os.environ["TMPDIR"]+"/FLIR16_x"+str(opt.scale)+".hdf5", 'r')
inputs = hf.get("input")
labels = hf.get("label")

kf = KFold(n_splits=10)
kf_index = 0

for trainFold_indexes, testFold_indexes in kf.split(inputs):

    if kf_index < opt.kfoldIndex:
        kf_index+=1
        continue

    print("===> Start K-FOLD Index " + str(kf_index))

    train_dataset = customDataset(inputs[trainFold_indexes], labels[trainFold_indexes])
    test_dataset = customDataset(inputs[testFold_indexes], labels[testFold_indexes])

    training_dataloader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize)
    testing_dataloader = DataLoader(dataset=test_dataset, num_workers=opt.threads, batch_size=opt.batchSize)

    if opt.model is None:
        print("model name?")
    elif not torch.cuda.is_available():
        print("cuda is not available")
    else:
        print("===> Loading model")

        if opt.model == "SRResNet":
            model = SRResNet(scale_factor=opt.scale, lr=opt.lr, batch_size=opt.batchSize)
            print("Model SRResNet is loaded")
        elif opt.model == "EDSR":
            model = EDSR(opt.scale, lr=opt.lr, batch_size=opt.batchSize)
            print("Model EDSR is loaded")
        elif opt.model == "WDSR":
            model = WDSR(scale_factor=opt.scale, lr=opt.lr, batch_size=opt.batchSize)
            print("Model WDSR is loaded")
        elif opt.model == "RRDBNet":
            model = RRDBNet(64, 23, scale_factor=opt.scale, lr=opt.lr, batch_size=opt.batchSize)
            print("Model RRDBNet is loaded")
        elif opt.model == "TherISuRNet":
            model = TherISuRNet(scale=opt.scale, lr=opt.lr, batch_size=opt.batchSize)
            print("Model TherISuRNet is loaded")
        
        save_path = opt.model + "/grayscale_x"+str(opt.scale)+"_checkpoint-kfoldIndex-" + str(kf_index) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_L2loss",
            dirpath=save_path,
            filename=opt.model + "-{epoch:02d}-{val_L2loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        print("===> Training KFLOD index " + str(kf_index))
        # training
        trainer = pl.Trainer(gpus=-1, accelerator="ddp", default_root_dir=save_path, max_epochs=opt.nEpochs, callbacks=[checkpoint_callback, lr_monitor])
        trainer.fit(model, training_dataloader, testing_dataloader)
        
        kf_index += 1
