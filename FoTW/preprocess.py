import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

make_tensor = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.ImageFolder("fotw_sims/smiles_trset", transform=make_tensor)
validset = torchvision.datasets.ImageFolder("fotw_sims/smiles_valset", transform=make_tensor)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, num_workers=2)

def labs_to_sens(labs):
    return (labs == trainset.class_to_idx['msmile']) | (labs == trainset.class_to_idx['mno'])

def labs_to_target(labs):
    return (labs == trainset.class_to_idx['msmile']) | (labs == trainset.class_to_idx['fsmile'])

df = pd.read_csv("fotw_sims/smiles_trset/gender_fex_trset.csv")

to_image = transforms.ToPILImage()
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataiter = iter(trainloader)
processed = torch.zeros(df.shape[0], 3, 224, 224)
targets = torch.zeros(df.shape[0])
for i in range(df.shape[0]):
    ref = trainset.imgs[i][0]
    row = df.loc[df['image_name'].str.slice(start=-9) == ref[len(ref)-9:]]
    assert row.shape == (1, 7)
    row = row.iloc[0]
    (top, left, height, width) = (row[' bbox_y'], row[' bbox_x'], row[' bbox_height'], row[' bbox_width'])
    images, labels = dataiter.next()
    processed[i] = preprocess(transforms.functional.crop(to_image(images[0]),
                                                         top=top,left=left,height=height,width=width))
    targets[i] = trainset.imgs[i][1]

torch.save(processed, "fotw_sims/train_data")
torch.save(targets, "fotw_sims/train_targets")

dataiter = iter(validloader)
processed = torch.zeros(df.shape[0], 3, 224, 224)
targets = torch.zeros(df.shape[0])
for i in range(df.shape[0]):
    ref = validset.imgs[i][0]
    row = df.loc[df['image_name'].str.slice(start=-9) == ref[len(ref)-9:]]
    assert row.shape == (1, 7)
    row = row.iloc[0]
    (top, left, height, width) = (row[' bbox_y'], row[' bbox_x'], row[' bbox_height'], row[' bbox_width'])
    images, labels = dataiter.next()
    processed[i] = preprocess(transforms.functional.crop(to_image(images[0]),
                                                         top=top,left=left,height=height,width=width))
    targets[i] = validset.imgs[i][1]

torch.save(processed, "fotw_sims/valid_data")
torch.save(targets, "fotw_sims/valid_targets")