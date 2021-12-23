
import torch
import torchvision

from torch2trt import torch2trt

dir = '/home/nano/workspace/models/lane-det-r18/'

device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 1)
model = model.cuda().eval().half()

model.load_state_dict(torch.load( dir + 'lane_follow_r18.pth'))
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), dir + 'lane_follow_r18.trt')
