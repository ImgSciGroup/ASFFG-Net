from utils1.dataset import ISBI_Loader
from utils1.loss import focal_loss,dice_loss,ContrastiveLoss1,loss_sum,focal_loss,Wight
from torch import optim
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
from model.ASFEGNet import ASFEGNet

import torch
import torch.nn as nn
import logging
import datetime



ModelName = 'ASFEGNet'


def train_net(net, device, data_path, epochs=150, batch_size= 8, lr=0.0001, is_Transfer=False):
    print('Conrently, Traning Model is :::::' + ModelName + ':::::')
    # 加载数据集

    #定义优化器
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.INIT_LEARNING_RATE, weight_decay=cfg.DECAY)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    #milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
                                                                    ,gamma=0.9)
    #scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    #scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9, last_epoch=-1)
    # 定义loss
    criterion = nn.BCEWithLogitsLoss()

    f_loss = open('train_Base_loss.txt', 'w')
    f_time = open('train_Base_time.txt', 'w')
    startime = datetime.datetime.now().strftime('%m-%d')
    log_dir = 'logger/' + startime + '-1'
    # 训练epochs次
    best_loss = float('inf')
    epochbest = 1
    for epoch in range(1, epochs + 1):
        isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
        train_loader = data.DataLoader(dataset=isbi_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        print(epoch)
        sums = 0
        logging.info('validation set:%d patches' % epoch)
        #开始训练
        net.train()
        num = int(0)
        best_mIoU = float(0)

        starttime = time.time()#记录时间
        #print('==========================epoch = ' + str(epoch) + '==========================')
        for image1, image2, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image1 = image1.to(device=device)
            image2 = image2.to(device=device)
            label = label.to(device=device)
            # 使用网络参数，输出预测结果
            pred1 = net(image1, image2)
            # 计算loss
    #***********************重新计算损失***********************************
            loss = criterion(pred1, label)
            total_loss = loss

            print("************",total_loss)
            sums += float(total_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(net.state_dict(),'best' + ModelName  + '_model_final.pth')
            ##############################保存损失##################################
            total_loss.backward()
            optimizer.step()
            num += 1
        # learning rate delay
        # f_loss.write('***********************sum****************************')
        # f_loss.write('\n')
        f_loss.write(str(float('%5f' % (sums/20))) + '\n')
        # f_loss.write('***********************sum****************************')
        # f_loss.write('\n')
        scheduler1.step()
        endtime = time.time()
        #######################保存时间#########################

        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
            float('%2f' % (starttime - endtime))) + '\n')

        print(epoch)


    f_loss.close()
    f_time.close()

if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道3，分类1(目标)3
    net = ASFEGNet(input_nbr=3,label_nbr=1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 指定训练集地址，开始训练A
    data_path = "data/ablation_experiment/train/"

    train_net(net, device, data_path)