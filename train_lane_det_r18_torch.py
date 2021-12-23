#!/usr/bin/python3

import torch
import torchvision

import threading
import time
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

from lane_dataset import cs_dataset 

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset = cs_dataset('/home/nano/workspace/imgs',transform=TRANSFORMS, random_hflip=True)

device = torch.device('cuda')

output_dim = 1   # angle coordinate for each category

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, output_dim)

model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())


def train_eval(dataset,epochs, batch_no, is_training=True):

    global model, optimizer 
        
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_no,
        shuffle=True
    )

    if is_training:
        model = model.train()
    else:
        model = model.eval()

    while epochs > 0:
        i = 0
        sum_loss = 0.0
        error_count = 0.0
        for img_name, images, angles in iter(train_loader):
            # send data to device
            images = images.to(device)
            angles = angles.to(device)

            if is_training:
                # zero gradients of parameters
                optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)
            
            loss = F.mse_loss(outputs, angles)
        
            if is_training:
                # run backpropogation to accumulate gradients
                loss.backward()
                # step optimizer to adjust parameters
                optimizer.step()

            # increment progress
            count = len(outputs.flatten())
            i += count
            
            progress = i / len(dataset)
            print(f'epoch ={epochs}, progress : {progress:.4f}, {i:04d} / {len(dataset):04d} batch_loss = {loss:.4f}')
        
        if is_training:
            epochs = epochs - 1
        else:
            break


if __name__ == "__main__" :

    import sys

    dir = '/home/nano/workspace/models/'
    file = 'lane_follow_r18.pth'

    epochs = 150
    if not sys.argv[1] == None :
        epochs = int(sys.argv[1])
 
    print('start training ')
    train_eval( dataset, epochs, batch_no=20, is_training=True)
    torch.save(model.state_dict(), dir + file)