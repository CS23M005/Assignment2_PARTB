# -*- coding: utf-8 -*-
"""dl-assn2-partb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rcTF55hVpNrLozvpb6CbyNBdZ4-owodY

<a href="https://colab.research.google.com/github/CS23M005/Assignment2_PartA/blob/main/CS23M005_A2_PartA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device

!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip
!unzip -q nature_12K.zip

!rm nature_12K.zip

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torchvision
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models

# Below function takes the optimizer string as input and outputs the model optimizer
def getOptim(model,optim_name, learning_rate):
  if(optim_name == 'sgd'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  elif(optim_name == 'adam'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  else:
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
  return optimizer

# this function takes the data and do forward propagation and generates the accuracy and loss
def check_accuracy(loader,model,criterion,batchSize):
    num_correct = 0
    num_loss = 0
    total = 0
    num_samples = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x) # forward propagation
            loss = criterion(scores, y)
            total_loss += loss.item()*batchSize
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
    model.train()
    return (num_correct / num_samples)*100 , total_loss

#Below code reads the dataset and transforms (2 types - with augmentation and without augmentation)

    #without augmentation
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])

    train_dataset = datasets.ImageFolder(root='inaturalist_12K/train',transform=transform)

    train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[8000,1999])

    #with augmentation
    transform2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])

    train_dataset2 = datasets.ImageFolder(root='inaturalist_12K/train',transform=transform2)

    train_dataset2,val_dataset2 = torch.utils.data.random_split(train_dataset2,[8000,1999])

    #function takes input augmentation string and produces required transformed data loader
    def getData(data_aug, batchSize):
        if(data_aug == "no"):
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset2,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset2,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        return train_loader, val_loader

# resnet50 model importing and removing the base model last layer and adding the required sized last layer
def resnet50_ud(output_size):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, output_size)
    for p in model.parameters():
        p.requires_grad = False #freezing
    for p in model.fc.parameters():
        p.requires_grad = True #unfreezing
    return model

#training the model constructed above
#get the data loader, model and train for each epoch
#log the necessary data into wandb
def train_cnn_ud(output_size,optim_name,batchSize,num_epochs,learning_rate, data_aug):

    train_loader, val_loader = getData(data_aug, batchSize)
    model = resnet50_ud(output_size).to(device)
    optimizer = getOptim(model,optim_name, learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data) #forward propagation
            loss = criterion(scores,targets)
            optimizer.zero_grad()
            loss.backward() #backaward propagation for weights update
            optimizer.step()
        train_accuracy,train_loss = check_accuracy(train_loader, model,criterion,batchSize)
        validation_accuracy,validation_loss = check_accuracy(val_loader, model,criterion,batchSize)
        print(f"train_accuracy:{train_accuracy:.4f},train_loss:{train_loss:.4f}")
        print(f"validation_accuracy:{validation_accuracy:.4f},validation_loss:{validation_loss:.4f}")
        wandb.log({'train_accuracy':train_accuracy})
        wandb.log({'train_loss':train_loss})
        wandb.log({'val_accuracy':validation_accuracy})
        wandb.log({'val_loss':validation_loss})

    #wandb.log({'train_accuracy':train_accuracy})

optim_name = 'adam'
batchSize=32
dropOut = 0.1
num_epochs = 5
learning_rate = 1e-3
input_channel=3
output_size=10
num_filters=16
filter_size=3
activation_fun = "relu"
filter_config = "same"
stride = 1
poolstride = 2
poolsize = 2
data_aug = "no"
train_cnn_ud(output_size,optim_name,batchSize,num_epochs,learning_rate, data_aug)

!pip install wandb
import wandb
wandb.login()

# def main_fun():
#     wandb.init(project ='Assignment2_PartB')
#     params = wandb.config
#     with wandb.init(project = 'Assignment2_PartB', name='optim_'+str(params.optim_name)
#                     +'epochs'+str(params.num_epochs) + 'batch_size_'+str(params.batchSize)
#                     +'lear_rate_'+str(params.learning_rate) + 'data_aug_'+ str(params.data_aug)) as run:
#         train_cnn_ud(output_size,params.optim_name,params.batchSize,params.num_epochs,params.learning_rate, params.data_aug)

# sweep_params = {
#     'method' : 'bayes',
#     'name'   : 'cs23m005',
#     'metric' : {
#         'goal' : 'maximize',
#         'name' : 'val_accuracy',
#     },
#     'parameters' : {
#             'optim_name' :{'values':['sgd','adam','nadam']},
#             'batchSize' : {'values':[32,64]},
#             'data_aug' :{'values':['yes','no']},
#             'num_epochs':{'values':[5,10]},
#             'learning_rate' :{'values':[1e-3,1e-4]}
#     }
# }
# sweepId = wandb.sweep(sweep_params,project = 'Assignment2_PartB')
# wandb.agent(sweepId,function =main_fun,count = 10)
# wandb.finish()

import argparse


def parse_args():
    p = argparse.ArgumentParser(description = "provide optinal parameters for training")
    p.add_argument('-wp', '--wandb_project', type=str, default="Assignment2_PartB", help="wandb project name")
    p.add_argument('-opt', '--optim_name', type=str, default="nadam", choices = ['sgd','adam','nadam'], help="optimizer for backprop")
    p.add_argument('-bS', '--batchSize', type=int, default=32, choices = [32, 64], help="batch size")
    p.add_argument('-ag', '--data_aug', type=str, default="no", choices = ['yes', 'no'], help="data augmentation")
    p.add_argument('-nE', '--num_epochs', type=int, default=5, choices = [5, 10], help="number of epochs")
    p.add_argument('-lR', '--learning_rate', type=float, default=1e-3, choices = [1e-3, 1e-4], help="learning rate")

args = parse_args()
wandb.init(project = args.wadb_project)
wandb.run.name=f'optimizer {str(args.optim_name)} epochs {str(args.num_epochs)} learning rate {args.learning_rate}'

train_cnn_ud(output_size,args.optim_name,args.batchSize,args.num_epochs,args.learning_rate, args.data_aug)