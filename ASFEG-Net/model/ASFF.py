import torch
import torch.nn as nn
import torch.nn.functional as F

class ASFF(nn.Module):
    def __init__(self,input_nbr,output_nbr):
        super(ASFF, self).__init__()
        self.conv1 = nn.Conv2d(input_nbr,input_nbr,kernel_size=3,stride=1,padding=1)
        self.Upconv = nn.ConvTranspose2d(output_nbr,input_nbr,kernel_size=3,padding=1,stride=2,output_padding=1)

    def forward(self,img1,img2):
        x1 = self.conv1(img1)
        x2 = self.conv1(self.Upconv(img2))

        out = F.relu(torch.add(x1,x2))

        out = F.sigmoid(self.conv1(out))

        out = torch.mul(out,img1)

        return out


if __name__ == '__main__':
    net = ASFF(3,16)
    im1 = torch.randn(1, 3, 256, 256)
    im2 = torch.randn(1, 16, 128, 128)
    out = net(im1,im2)
    print(out.shape)