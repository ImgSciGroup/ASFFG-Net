

"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
from model.MFDM import MFDM
from model.ASFF import ASFF
from model.BEDE import BEDE


class ASFEGNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(ASFEGNet, self).__init__()
        self.input_nbr = input_nbr
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#********************************************************************
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# ********************************************************************
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
# ********************************************************************
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.eg5 = nn.Sequential(
            BEDE(512, 512, stride=1, padding=1, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        self.eg4 = nn.Sequential(
            BEDE(384,384,stride=1,padding=1,dilation=1,groups=1,pooling_r=3,norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        self.eg3 = nn.Sequential(
            BEDE(192, 192, stride=1, padding=1, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        self.eg2 = nn.Sequential(
            BEDE(96, 96, stride=1, padding=1, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        self.eg1 = nn.Sequential(
            BEDE(48, 48, stride=1, padding=1, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        #self.maxpoolgb = nn.MaxPool2d(kernel_size=2,padding=0)

        self.af1 = ASFF(16,32)
        self.af2 = ASFF(32,64)
        self.af3 = ASFF(64,128)
        self.af4 = ASFF(128,256)

        self.df1 = MFDM(16,128)
        self.df2 = MFDM(32,64)
        self.df3 = MFDM(64,32)
        self.df4 = MFDM(128,16)



##################反卷积#################################################

        self.Dconv5 = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )

        self.Dconv4 = nn.Sequential(


            nn.ConvTranspose2d(384, 256, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )

        self.Dconv3 = nn.Sequential(

            nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )

        self.Dconv2 = nn.Sequential(

            nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True)


        )
        self.Dconv1 = nn.Sequential(

            nn.ConvTranspose2d(48, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(32,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(16,label_nbr,kernel_size=3,padding=1)
        )

    def forward(self,x1,x2):

        #stage 1
        img11 = self.conv1(x1)
        img12 = self.conv1(x2)

        img11 = self.maxpool1(img11)
        img12 = self.maxpool1(img12)

        #stage_2
        img21 = self.conv2(img11)
        img22 = self.conv2(img12)

        img21 = self.maxpool2(img21)
        img22 = self.maxpool2(img22)

        # stage 3
        img31 = self.conv3(img21)
        img32 = self.conv3(img22)

        img31 = self.maxpool1(img31)
        img32 = self.maxpool1(img32)

        # stage 4
        img41 = self.conv4(img31)
        img42 = self.conv4(img32)

        img41 = self.maxpool1(img41)
        img42 = self.maxpool1(img42)

        #stage5
        img51 = self.conv5(img41)
        img52 = self.conv5(img42)

        img51 = self.maxpool5(img51)
        img52 = self.maxpool5(img52)

#############difference feature pinjie##############
        img41 = self.af4(img41, img51)
        img42 = self.af4(img42, img52)
        # print(img42.shape)
        img4 = self.df4(img41, img42)

        img31 = self.af3(img31, img41)
        img32 = self.af3(img32, img42)
        # print(img32.shape)
        img3 = self.df3(img31, img32)

        img21 = self.af2(img21, img31)
        img22 = self.af2(img22, img32)
        # print(img21.shape)
        img2_df = self.df2(img21, img22)

        img11 = self.af1(img11,img21)
        img12 = self.af1(img12,img22)

        img1 = self.df1(img11,img12)

##################Deconder#############################
        img5 = torch.cat((img51,img52),1)

        img5_edge = self.eg5(img5)
        img5 = self.Dconv5(img5_edge)
        #print(img5.shape)


        img4 = torch.cat((img5,img4),1)

        img4_edge = self.eg4(img4)
        img4 = self.Dconv4(img4_edge)

        #print(img4.shape)

        img3 = torch.cat((img3,img4),1)
        img3_edge = self.eg3(img3)
        img3 = self.Dconv3(img3_edge)
        #print(img3.shape)

        img2 = torch.cat((img3,img2_df),1)
        img2_edge = self.eg2(img2)
        img2 = self.Dconv2(img2_edge)
        #print(img2.shape)

        img2 = torch.cat((img2,img1),1)
        img1_edge = self.eg1(img2)
        out = self.Dconv1(img1_edge)

        return out



if __name__ == '__main__':
    net = ASFEGNet(3,1)
    im1 = torch.randn(1, 3, 256, 256)
    im2 = torch.randn(1, 3, 256, 256)
    out = net(im1, im2)
    print(out.shape)