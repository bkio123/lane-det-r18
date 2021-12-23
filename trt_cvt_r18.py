
import torch
import torchvision

from torch2trt import torch2trt

device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 1)
model = model.cuda().eval().half()

model.load_state_dict(torch.load('/home/nano/workspace/models/lane-follow/lane_follow_r18.pth'))
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), 'lane_follow_r18.trt')


