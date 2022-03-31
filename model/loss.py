import torch.nn as nn
import torch.functional as F


def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss

def reconstruction_loss():
    loss = nn.MSELoss()
    return loss