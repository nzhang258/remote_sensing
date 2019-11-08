# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import resnet
#import matplotlib.pyplot as plt
import time
import os
import copy
# import matplotlib.pyplot as plt
# import numpy as np
import h5py
import pickle as pkl
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224, scale=(0.8,1.0), ratio=(0.95, 1.05)),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = './bing_data'
#data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
relu = torch.nn.ReLU(inplace=True)
def cluster_loss(labels, x1, centroids_gpu):
    # print('preds', preds)
    # print('x1', x1)
    # print('centroids_gpu', centroids_gpu)
    n = labels.shape[0]
    loss = 0
    for i in range(n):
        # print('x1', x1[i])
        # print('center', centroids_gpu[labels[i]])
        # diff = x1[i]-centroids_gpu[preds[i]]
        # diff = diff*diff
        # diff = torch.sum(diff)
        # diff = torch.sqrt(diff)
        # print(diff)
        # print('loss', torch.norm(x1[i]-centroids_gpu[labels[i]]))
        loss += relu(torch.norm(x1[i]-centroids_gpu[labels[i]])-0.2)
    # print('                              loss', loss)
    return loss

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    centroids = None

    centroids = np.random.rand(num_classes, 2)

    # centroids = np.array([[0,1],[1,1]])
    centroids_gpu = torch.from_numpy(centroids).float().to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        data = []
        for i in range(num_classes):
            data.append([])
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_closs = 0.0
            running_loss = 0.0
            running_corrects = 0
            
            sums = [np.zeros(2)]*num_classes #torch.Tensor(num_classes, 2, device=torch.device("cuda"))
            counts = [0]*num_classes

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('labels', labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, x1 = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    closs = cluster_loss(labels, x1, centroids_gpu)
                    # loss = loss + closs
                    # backward + optimize only if in training phase
                    
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        for i in range(len(preds)):
                            pred = preds[i].cpu().data.numpy()
                            label = labels[i].cpu().data.numpy()
                            v1 = x1[i].cpu().data.numpy()
                            counts[label] += 1                    
                            sums[label] += v1
                            data[label].append(v1)
                        # print('centroids', centroids[pred])
                        # centroids[i] += x1[preds[i],:]

                # statistics
                running_closs += closs.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            print('counts', counts)
            print('centroids', centroids)
            
            epoch_closs = running_closs / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} loss: {:.4f}, closs: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_closs, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        # fp = open('../data/t0/epoch_{}.pkl'.format(epoch),'wb')
        # pkl.dump({'data': data, 'centroids': centroids}, fp)
        # # fp['data'] = data
        # fp.close()
        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = resnet.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2500)
