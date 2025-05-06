import torch
import torch.nn as nn
import torch.nn.functional as F
class BEDE(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(BEDE, self).__init__()

        self.k1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes)
            #nn.ReLU(True)
        )

        self.k2 = nn.Sequential(            #先尺度减一半，在卷积通道数减一半
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),      #kernel_size:池化核的尺寸大小，stride:窗口的移动步幅，默认与kernel_size大小一致     ##3*3
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,        #输入的四维张量[N,C,H,W]中的c，输入张量的channels数
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),     #norm_layer:归一化层，Layer Norm在通道方向上，对CHW归一化，就是对每个深度上的输入进行归一化，主要对RNN作用明显
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,        #经过卷积后，通道减一半                ##5*5
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k5 = nn.Sequential(
            nn.Conv2d(2*inplanes, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.edgeconv = nn.Sequential(
                    nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,padding=0),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*inplanes,planes,kernel_size=1,stride=1,padding=0)
        )



    def forward(self, x):
        identity = x        #[N,C,H,W]表示图片，先取W方向的一行数据，从左向右；然后H方向，从上向下；之后C方向；最后N方向，从batch中的一张图片(n=0）跳转到n-1

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)   #sigmoid是一种激活函数，它会将样本值映射到0到1之间
        #interpolate:利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整
        out1 = torch.sigmoid(torch.add(identity, F.interpolate(self.k3(x), identity.size()[2:])))
        out = torch.mul(out1, out) # k3 * sigmoid(identity + k2)
        #torch.mul:输入两个张量矩阵，输出：他们的点乘运算结果；  在计算机视觉领域中，torch.mul常用于特征图与注意力的相乘
        out = self.k4(out) # k4

        #print(out.shape)
        out2 = self.k1(x)
        outedge = self.edgeconv(x)
        #print(out2.shape)
        out2 = torch.cat((out2,outedge),dim=1)
        #print(outedge.shape)
        out2 = self.conv1(out2)
        out2 = torch.softmax(out2, 1)
        #print(out2.shape)
        out = torch.cat((out2,out),dim=1)

        out = self.k5(out)
        out = torch.softmax(out, 1)

        return out