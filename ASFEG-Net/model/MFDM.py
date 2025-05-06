import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LSTM


class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()
        outfeats = torch.mul(feats, input_)

        return outfeats

class conv_block(nn.Module):
    def __init__(self, input, output):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class MFDM(nn.Module):
    def __init__(self,inplane,patch_size):
        super(MFDM, self).__init__()



        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ca1 = CAM(in_channels=inplane)
        self.ca2 = CAM(in_channels=inplane)

        self.patch_size = patch_size
        self.Conv1 = conv_block(input=inplane,output=inplane)
        self.lstm1 = LSTM.ConvLSTM(input_channels=inplane, hidden_channels=[inplane], kernel_size=3, step=2,
                                   effective_step=[1], height=self.patch_size, width=self.patch_size)

        self.Conv4 = nn.Sequential(
            nn.Conv2d(inplane,inplane,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(inplane),
            nn.ReLU(True)
        )



    def forward(self,img1,img2):
        img1_1 = img1.unsqueeze(0)
        img2_1 = img2.unsqueeze(0)
        x = torch.cat([img1_1,img2_1],axis=0)
        weight,_= self.lstm1(self.Conv1,x)

        x1 = self.ca1(img1)
        x2 = self.ca2(img2)
        out = torch.abs(x1-x2)

        weight = weight[0]          #LSTM计算两期影像相似度权重

        out = torch.mul(weight,out)

        out = self.Conv4(out)
        #print(weight.shape)
        #out = self.Conv4(out)

        return out


if __name__ == '__main__':
    img1 = torch.randn(1,16,256,256)
    img2 = torch.randn(1, 16, 256, 256)

    net = MFDM(16,256)
    out = net(img1,img2)
    print(out.shape)