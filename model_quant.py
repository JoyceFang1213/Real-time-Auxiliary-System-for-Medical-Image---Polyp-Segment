import os
import re
import sys
import argparse
import time
import pdb
import random
import numpy as np
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
from PIL import Image

sys.path.append("HarDNet-MSEG/")
from lib.HarDMSEG import HarDMSEG

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="data",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="best.pt",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode',
                    default='calib',
                    choices=['float', 'calib', 'test'],
                    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune',
                    dest='fast_finetune',
                    action='store_true',
                    help='fast finetune model before calibration')
parser.add_argument('--deploy',
                    dest='deploy',
                    action='store_true',
                    help='export xmodel for deployment')
args, _ = parser.parse_known_args()


class KvasirDataset(data.Dataset):
    def __init__(self, image_root, img_size):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images.sort()
        self.size = len(self.images)

        normalize = transforms.Normalize(
            mean=[0.5572, 0.3216, 0.2357], std=[0.3060, 0.2145, 0.1773])

        self.img_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
            #transforms.ToPILImage()
        ])

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        #print(type(image))
        #image = cv2.imread(image_path)
        image = self.img_transform(image)
        #print(type(image))
        return image

    def __len__(self):
        return self.size


def get_load(train=True,
               data_dir="data",
               batchsize=128,
               subset_len=200,
               sample_method='random',
               **kwargs):
    traindir = data_dir + '/images/'
    valdir = data_dir + '/images/'
    img_size = 352
    print(valdir)
    if train:
        dataset = KvasirDataset(traindir, img_size=img_size)
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))

        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=True,
                                      sampler=None)

    else:
        dataset = KvasirDataset(valdir, img_size=img_size)
        print(len(dataset))
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=False)
    return data_loader


def evaluate(model, val_loader) -> object:
    model.eval()
    model = model.to(device)

    for iteration, (images) in tqdm(
            enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        model(images)
    return


def quantization(title='optimize',
                 model_name='',
                 file_path=''):
    data_dir = args.data_dir
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    model = HarDMSEG().to(device)
    check = torch.load(file_path)
    model.load_state_dict(check, strict=False)

    rand_input = torch.randn([batch_size, 3, 352, 352])
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
            quant_mode, model, rand_input, device=device)

        quant_model = quantizer.quant_model

    val_loader = get_load(
        subset_len=subset_len,
        train=False,
        batch_size=batch_size,
        sample_method='random',
        data_dir=data_dir)

    evaluate(quant_model, val_loader)

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)


if __name__ == '__main__':

    model_name = 'HarDNet-MSEG'
    file_path = 'best_old.pt'

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    quantization(
        title=title,
        model_name=model_name,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))
