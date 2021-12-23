#!/usr/bin/python3
import cv2

import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import torchvision
import time
import atexit

device = torch.device('cuda')
working_dir = '/home/nano/workspace/models/lane-det-r18/'

output_dim = 1   # angle coordinate for each category

class lane_det_cnn_torch():
    
    def __init__(self):
        # RESNET 18
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, output_dim)

    def preprocess(self,image):

        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
       
        #image size change
        image = cv2.resize(image,(224,224))

        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def load_model(self, file):
        
        self.model.load_state_dict(torch.load( working_dir + file))
        #self.model.load_state_dict(torch.load(file))

        self.model = self.model.to(device)
        self.model.eval()

    def img_to_angle(self,image):

        image = self.preprocess(image)

        output = self.model(image)
        x = float(output[0])
        angle = int(x * 900)
        return angle

if __name__ == '__main__' :
    
    img = cv2.imread('000.jpg')
    det = lane_det_cnn_torch()

    det.load_model('lane_follow_r18.pth')
    print('load model')

    angle = det.img_to_angle(img)

    print(f' angle = {angle} ')

