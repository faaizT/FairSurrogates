import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CelebA(root='./data', split='train', target_type='attr', transform=preprocess)
validset = torchvision.datasets.CelebA(root='./data', split='valid', target_type='attr', transform=preprocess)
testset  = torchvision.datasets.CelebA(root='./data', split='test',  target_type='attr', transform=preprocess)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)

attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()

model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=False)
model.fc = nn.Linear(2048, 1, bias=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

ti = attrs.index("Smiling")
si = attrs.index("Male")

(Pmale, Pfem) = (testset.attr[:,si].float().mean(), 1 - testset.attr[:,si].float().mean())
ploss = nn.BCEWithLogitsLoss()
lam_fair = 0
form = "linear" 
if form == "logistic":
    def floss(outputs, sens_attr):
        return -lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr]).sum()/Pmale + F.logsigmoid(-outputs[~sens_attr]).sum()/Pfem)
elif form == "hinge":
    baseline = torch.tensor(0.).to(device)
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr]).sum()/Pmale + torch.max(baseline,1+outputs[~sens_attr]).sum()/Pfem)
else:
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (-outputs[sens_attr].sum()/Pmale + outputs[~sens_attr].sum()/Pfem)

def calc_loss(data):
    inputs, labels = data
    inputs, labels, sens_attr = inputs.to(device), labels[:,ti].float().to(device), labels[:,si].bool().to(device)
#     optimizer.zero_grad()
    outputs = model(inputs).reshape(-1)
    pred_loss = ploss(outputs, labels)
    loss = pred_loss + floss(outputs, sens_attr)
#     loss.backward()
    preds = (outputs >= 0).float()
    unfairness = torch.tensor([preds[ sens_attr].sum(), preds[ sens_attr].shape[0],
                               preds[~sens_attr].sum(), preds[~sens_attr].shape[0]]) #msmiling, m, fsmiling, f
    return ((labels == preds).float().mean(), loss, unfairness, pred_loss)

with torch.cuda.device('cuda:1'):
    lfs = np.array([0.   , 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024,
        0.027, 0.03 , 0.033, 0.036, 0.039, 0.042, 0.045, 0.048, 0.051,
        0.054, 0.057, 0.06 ])
    df = pd.DataFrame(columns = ['Lam_fair', 'Accuracy', 'Unfairness', 'Loss',"LL"])
    for lam_fair in lfs:
        model.load_state_dict(torch.load("sims/linear" + str(lam_fair)))
        torch.cuda.empty_cache()
        iterator = testloader.__iter__()
        N = iterator.__len__()
        running_loss = 0.0
        running_acc = 0.0
        running_unfair = 0.0
        running_predloss = 0.0
        with torch.no_grad():
            for i in trange(N):
                # get the inputs; data is a list of [inputs, labels]
                (acc, loss, unfair, pred_loss) = calc_loss(iterator.next())

                # print statistics
                running_loss += loss.item()
                running_acc += acc.item()
                running_unfair += unfair
                running_predloss += pred_loss.item()
            d = {"Lam_fair": lam_fair,
                 "Accuracy": (running_acc / N), 
                 "Unfairness": (running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1]).item(),
                 "Loss": (running_loss / N),
                 "LL": -(running_predloss / N)}
        df = df.append(d,ignore_index=True)
    df.to_csv("linear.csv")