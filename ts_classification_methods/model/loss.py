import torch.nn as nn


def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss


def reconstruction_loss():
    loss = nn.MSELoss()
    return loss
