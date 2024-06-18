import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

def dice_coefficient(input, target, smooth=1):
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.sigmoid(input)

    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    target = target.float()

    intersect = (input * target).sum(dim=1)
    dice = (2.*intersect)/(input.sum(dim=1) + target.sum(dim=1))

    return torch.mean(dice)

def mse(input, target):
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    shape = input.shape
    n = shape[1]*shape[2]*shape[3]*shape[4]

    input = torch.sigmoid(input)

    input = input.view(input.shape[0], -1)
    target = target.view(input.shape[0], -1)
    target = target.float()

    mse = ((input - target)**2).sum(dim=1) / n

    return torch.mean(mse)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice