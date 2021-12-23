
import torch
import cv2
import PIL.Image

from torch2trt import TRTModule
import torchvision.transforms as transforms


dir = '/home/nano/workspace/models/lane-det-r18/'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load( dir + 'lane_follow_r18.trt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(image):

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
 
    #image size change
    image = cv2.resize(image,(224,224))

    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

image = cv2.imread('000.jpg')
image = preprocess(image).half()
output = model_trt(image).detach().cpu().numpy().flatten()
x = float(output[0])
print(x * 900)
