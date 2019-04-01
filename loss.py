from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import copy
import math

mse = nn.MSELoss()

def loss_calculation(prediction, target):

    loss = mse(prediction, target)

    return loss

class Loss(_Loss):

    def __init__(self):
        super(Loss, self).__init__(True)
    
    def forward(self, pos_pred, point, cos_pred, cos, sin_pred, sin, width_pred, width):
        pos_loss = loss_calculation(pos_pred, point)
        cos_loss = loss_calculation(cos_pred, cos)
        sin_loss = loss_calculation(sin_pred, sin)
        width_loss = loss_calculation(width_pred, width)
        total_loss = pos_loss + cos_loss + sin_loss + width_loss
        return pos_loss, cos_loss, sin_loss, width_loss, total_loss