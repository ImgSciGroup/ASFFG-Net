import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.BCEWithLogitsLoss()
criterion1 = nn.MSELoss()

def DistanceLoss(pred,label):

    pred = torch.sigmoid(pred)
    label = torch.sigmoid(label)
    pred = np.array(pred)
    label = np.array(label)

    loss = np.mean((pred - label)**2)

    return loss

def Wight(pred,label):
    pred = torch.sigmoid(pred)
    label = torch.sigmoid(label)
    w = torch.mean(abs(label-pred))/4
    return w

def dice_loss(pred,label):
    pred=torch.sigmoid(pred)
    loss=torch.sum(1-2*pred*label/(pred+label)) / ((128 ** 2)*8)
    return loss

def loss_sum(pred,label):


    loss1 = DistanceLoss(pred,label)

    loss2 = dice_loss(pred,label)

    loss3 = criterion(pred,label)


    total_loss = 0.2*loss1 + 0*loss2 + 0.5*loss3

    return total_loss

if __name__ == '__main__':

    image1 = torch.randn(3, 1, 128, 128)
    image2 = torch.randn(3, 1, 128, 128)
    #criterion =focal_loss()
    out1 = DistanceLoss(image1,image2)
    print(out1)


