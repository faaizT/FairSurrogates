import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms

if len(sys.argv) != 4:
    print("Usage: python simulate.py lambda_fair formulation GPU_index")
    exit(1)
lam_fair = float(sys.argv[1])
form = sys.argv[2]
gpui = int(sys.argv[3])


traindata = torch.load("train_data") 
trainlabs = torch.load("train_targets")
validdata = torch.load("valid_data")
validlabs = torch.load("valid_targets")


trainset = torch.utils.data.TensorDataset(traindata, trainlabs)
validset = torch.utils.data.TensorDataset(validdata, validlabs)
lab_ref = torchvision.datasets.ImageFolder("smiles_trset").class_to_idx

assert lab_ref == torchvision.datasets.ImageFolder("smiles_valset").class_to_idx

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, num_workers=2)

model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=True) #pretrained = True if you want
model.fc = nn.Linear(2048, 1, bias=True)
model.load_state_dict(torch.load("baseline"))

for p in model.conv1.parameters():
    p.requires_grad = False
for p in model.bn1.parameters():
    p.requiers_grad = False
for p in model.layer1.parameters():
    p.requires_grad = False


optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=4)

device = torch.device("cuda:" + str(gpui) if torch.cuda.is_available() else "cpu")
model.to(device)
torch.cuda.empty_cache()

def labs_to_unknown(labs):
    return (labs == lab_ref['usmile']) | (labs == lab_ref['uno'])

def labs_to_sens(labs):
    return (labs == lab_ref['fsmile']) | (labs == lab_ref['fno'])

def labs_to_target(labs):
    return (labs == lab_ref['msmile']) | (labs == lab_ref['fsmile'])

Pmale = labs_to_sens(trainset.tensors[1]).float().mean()
Pfem =  1 - Pmale - labs_to_unknown(trainset.tensors[1]).float().mean()

ploss = nn.BCEWithLogitsLoss()
if form == "logistic":
    def floss(outputs, male, female):
        return -lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[male]).sum()/Pmale + F.logsigmoid(-outputs[female]).sum()/Pfem)
elif form == "hinge":
    baseline = torch.tensor(0.).to(device)
    def floss(outputs, male, female):
        return lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[male]).sum()/Pmale + torch.max(baseline,1+outputs[female]).sum()/Pfem)
else:
    def floss(outputs, male, female):
        return lam_fair/outputs.shape[0] * (-outputs[male].sum()/Pmale + outputs[female].sum()/Pfem)

def calc_loss(data): 
    inputs, labels = data
    sens_attr, unknown = labs_to_sens(labels).bool().to(device), labs_to_unknown(labels).bool().to(device)
    female = sens_attr & ~unknown
    male = ~sens_attr & ~unknown
    inputs, labels = inputs.to(device), labs_to_target(labels).float().to(device)
    optimizer.zero_grad()
    outputs = model(inputs).reshape(-1)
    loss = ploss(outputs, labels) + floss(outputs, male, female)
    loss.backward()
    preds = (outputs >= 0).float()
    unfairness = torch.tensor([preds[  male].sum(), preds[  male].shape[0],
                               preds[female].sum(), preds[female].shape[0]]) #msmiling, m, fsmiling, f
    return ((labels == preds).float().mean(), loss, unfairness)


print_every = 50
valid_batches = 16
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    running_acc = 0.0
    running_unfair = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        (acc, loss, unfair) = calc_loss(data)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_acc += acc.item()
        running_unfair += unfair

        if i % print_every == (print_every - 1):    # print every 200 mini-batches
            print('[%d, %5d]' % (epoch + 1, i + 1))
            valid_loss = 0.0
            valid_acc = 0.0
            valid_iter = iter(validloader)
            for vi in range(valid_batches):
                (new_valid_acc, new_valid_loss, _) = calc_loss(next(valid_iter))
                valid_loss += new_valid_loss
                valid_acc += new_valid_acc.item()
            scheduler.step(valid_loss)
            print('Training Accuracy: %.3f, Validation Accuracy: %.3f, Unfairness: %.3f' % (running_acc / print_every,
                                                        valid_acc/valid_batches,
                                                        running_unfair[2]/running_unfair[3] -
                                                        running_unfair[0]/running_unfair[1]))
            print('Training Loss: %.3f, Validation Loss: %.3f' % (running_loss / print_every, valid_loss/valid_batches))
            print('')
            running_loss = 0.0
            running_acc = 0.0
            running_unfair = 0.0
            sleep(1)

print('Finished Training')

torch.save(model.state_dict(), form + str(lam_fair))
