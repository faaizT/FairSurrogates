import pandas as pd
import os
import torchvision
from torchvision import transforms
import torch

df = pd.read_csv("smiles_valset/gender_fex_valset.csv")
j = [0,0,0,0]
for i, row in df.iterrows():
    if row[' Gender'] == 0 and row[' Smile'] == 1:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/msmile/" + row['image_name'])
        j[0] += 1
    elif row[' Gender'] == 1 and row[' Smile'] == 1:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/fsmile/" + row['image_name'])
        j[1] += 1
    elif row[' Gender'] == 0 and row[' Smile'] == 0:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/mno/" + row['image_name'])
        j[2] += 1
    elif row[' Gender'] == 1 and row[' Smile'] == 0:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/fno/" + row['image_name'])
        j[3] += 1
    elif row[' Gender'] == 2 and row[' Smile'] == 0:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/uno/" + row['image_name'])
    elif row[' Gender'] == 2 and row[' Smile'] == 1:
        os.rename("smiles_valset/" + row['image_name'], "smiles_valset/usmile/" + row['image_name'])
    else:
        print("Exception!")
print(j)
print(j[0]/(j[0]+j[2]), j[1]/(j[1]+j[3]))
