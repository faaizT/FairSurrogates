import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import pandas as pd

from PIL import Image

class SimpleCelebA(Dataset):
    def __init__(self, root, split='train', target_type='attr', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        
        # Read the data files
        attr_data = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        partition_data = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        
        # Use only the first n-10000 data points
        attr_data = attr_data.tail(10000)
        partition_data = partition_data.tail(10000)
        
        # Filter based on the provided split
        # split_dict = {'train': 0, 'valid': 1, 'test': 2}
        # partition_data = partition_data[partition_data['partition'] == split_dict[split]]
        
        # Merge datasets on image_id and filter attributes
        self.data = pd.merge(partition_data, attr_data, on='image_id')
        
        # Convert attributes from -1 to 0
        self.data[target_type] = (self.data[target_type] == 1).astype(int)
        self.attr = torch.FloatTensor(self.data[target_type].values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root, 'img_align_celeba/img_align_celeba', self.data.iloc[idx, 0])
        image = Image.open(img_name)
        
        attrs = self.attr[idx, :]
        if self.transform:
            image = self.transform(image)
            
        return image, attrs

attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()

# Usage
root_dir = '/root/celeba'
testset  = SimpleCelebA(root=root_dir, split='test',  target_type=attrs, transform=preprocess)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)

attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()

model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=False)
model.fc = nn.Linear(2048, 1, bias=True)

model.to(device)

ti = attrs.index("Smiling")
si = attrs.index("Male")

(Pmale, Pfem) = ((testset.attr[:,si].bool() & testset.attr[:,ti].bool()).float().mean(),
                (~testset.attr[:,si].bool() & testset.attr[:,ti].bool()).float().mean())


ploss = nn.BCEWithLogitsLoss()
lam_fair = 0
form = "linear"
if form == "logistic":
    def floss(outputs, sens_attr):
        return -lam_fair/32 * (F.logsigmoid(outputs[sens_attr]).sum()/Pmale + F.logsigmoid(-outputs[~sens_attr]).sum()/Pfem)
elif form == "linear":
    def floss(outputs, sens_attr):
        return lam_fair/32 * (-outputs[sens_attr].sum()/Pmale + outputs[~sens_attr].sum()/Pfem)
elif form == "weighting":
    def floss(outputs, sens_attr):
        return -lam_fair/32 * (F.logsigmoid(outputs[sens_attr]).sum()/Pmale - F.logsigmoid(outputs[~sens_attr]).sum()/Pfem)

def calc_loss(data):
    inputs, labels = data
    inputs, labels, sens_attr = inputs.to(device), labels[:,ti].float().to(device), labels[:,si].bool().to(device)
    labels_bool = labels.bool()
#     optimizer.zero_grad()
    outputs = model(inputs).reshape(-1)
    pred_loss = ploss(outputs, labels)
    loss = pred_loss + floss(outputs[labels_bool], sens_attr[labels_bool])
#     loss.backward()
    preds_acc = (outputs >= 0).float()
    preds = torch.sigmoid(outputs)
    unfairness = torch.tensor([preds[ sens_attr & labels_bool].sum(), preds[ sens_attr & labels_bool].shape[0],
                               preds[~sens_attr & labels_bool].sum(), preds[~sens_attr & labels_bool].shape[0]]) #msmiling, m, fsmiling, f
    runfairness = torch.tensor([preds[ sens_attr & ~labels_bool].sum(), preds[ sens_attr & ~labels_bool].shape[0],
                                preds[~sens_attr & ~labels_bool].sum(), preds[~sens_attr & ~labels_bool].shape[0]]) #msmiling, m, fsmiling, f
    return ((labels == preds_acc).float().mean(), loss, unfairness, pred_loss, runfairness)

with torch.cuda.device('cuda:0'):
    lfs = list(np.linspace(0, 0.5, num=11).round(2))
    seeds = [2, 3, 4]
    df = pd.DataFrame(columns = ['Lam_fair', 'Accuracy', 'unfairness', 'Loss',"LL"])
    for lam_fair in lfs:
        for seed in seeds:
            if os.path.exists(f"../model_results_celeba-fairsurrogates_{form}_abs/model_{form}_{lam_fair}_{seed}_eop.pth"):
                model.load_state_dict(torch.load(f"../model_results_celeba-fairsurrogates_{form}_abs/model_{form}_{lam_fair}_{seed}_eop.pth"))
                torch.cuda.empty_cache()
                iterator = iter(testloader)
                N = iterator.__len__()
                running_loss = 0.0
                running_acc = 0.0
                running_unfair = 0.0
                running_runfair = 0.0
                running_predloss = 0.0
                with torch.no_grad():
                    for i in trange(N):
                        # get the inputs; data is a list of [inputs, labels]
                        (acc, loss, unfair, pred_loss, runfair) = calc_loss(next(iterator))

                        # print statistics
                        running_loss += loss.item()
                        running_acc += acc.item()
                        running_unfair += unfair
                        running_runfair += runfair
                        running_predloss += pred_loss.item()

                    d = {"Lam_fair": lam_fair,
                        "Accuracy": (running_acc / N), 
                        "unfairness": (running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1]).item(),
                        "Runfairness": (running_runfair[2]/running_runfair[3] - running_runfair[0]/running_runfair[1]).item(),
                        "Loss": (running_loss / N),
                        "LL": -(running_predloss / N)}
                    for k in d:
                        d[k] = list([d[k]])
                    d = pd.DataFrame(data = d)
                df = pd.concat([df, d], ignore_index=True) 
    df.to_csv(f"./celeba_{form}_eop_test.csv")