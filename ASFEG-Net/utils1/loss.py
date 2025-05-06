import torch
import torch.nn as nn
import torch.nn.functional as F
criterion = nn.BCEWithLogitsLoss()
def focal_loss(pred, label):
    alpha, gamma = 0.25, 2
    p = torch.sigmoid(pred)
    loss = -1*torch.sum( alpha*((1-p)**gamma)*(torch.log(p) * label + (1-alpha)*(p**gamma)*torch.log(1 - p) * (1 - label)))/((128**2)*6)
    return loss
def focal_loss1(y_true, y_pred):
    alpha, gamma = 0.25, 2
    p = torch.sigmoid(y_pred)
    return torch.sum(- alpha*y_true * torch.log(p) - (1 - y_true) * torch.log(1 - p))/((128** 2)*8)

def dice_loss(pred,label):
    pred=torch.sigmoid(pred)
    loss=torch.sum(1-2*pred*label/(pred+label)) / ((128 ** 2)*8)
    return loss
def Wight(pred,label):
    pred = torch.sigmoid(pred)
    label = torch.sigmoid(label)
    w = torch.mean(abs(label-pred))/4
    return w



def loss_sum(pred,label):
    dice = criterion(pred, label)
    focal = focal_loss(label, pred)
    loss_s =dice+0.5*focal
    return loss_s

class ContrastiveLoss1(nn.Module):
    def __init__(self, margin1=0.1, margin2=1.2, eps=1e-6):
        super(ContrastiveLoss1, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x, y):
        diff = torch.abs(x - y)
        dist_sq = torch.pow(diff+self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)

        # print(y.data.type(), mdist_pos.data.type(), mdist_neg.data.type())
        loss_pos =(1- y)*(mdist_pos.pow(2))
        loss_neg = y*(mdist_neg.pow(2))

        loss = torch.mean(loss_pos + loss_neg)
        return loss

if __name__ == '__main__':
    criterion1 = ContrastiveLoss1()
    image1 = torch.randn(3, 1, 128, 128)
    image2 = torch.randn(3, 1, 128, 128)
    #criterion =focal_loss()
    out1 = Wight(image1,image2)
    print(out1)