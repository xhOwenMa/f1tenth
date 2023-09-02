import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score

import numpy as np
import pandas as pd
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def train():
    args = parse_args()
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    resize_height = args.height
    resize_width = args.width
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.GaussianBlur(kernel_size=(1, 9), sigma=(0.1, 0.2))
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])
    """
    data_transforms = {
        'train': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(resize_height, resize_width, scale=(0.6, 1.0)),
            A.FancyPCA(always_apply=False, p=0.75),
            A.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3, always_apply=False, p=0.95),
            A.AdvancedBlur(always_apply=False, p=0.9),
            A.Resize(resize_height, resize_width),
            A.Normalize(),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(resize_height, resize_width),
            A.Normalize(),
            ToTensorV2()
        ]),
    }
    
    target_transforms = A.Compose([
        A.Resize(resize_width, resize_height),
    ])

    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

    model = LaneNet(arch=args.model_type)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")
    
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=args.lr_decay)

    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE,
                            loss_type=args.loss_type, num_epochs=args.epochs, pretrained=args.pretrained, ckpt=args.ckpt, save_path=save_path)
    df=pd.DataFrame({'epoch':[],'training_loss':[],'training_miou':[],'val_loss':[],'val_miou':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['training_miou'] = log['training_miou']
    df['val_loss'] = log['val_loss']
    df['val_miou'] = log['val_miou']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','training_miou','val_loss','val_miou'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
