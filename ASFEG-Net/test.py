import torch
import torchvision.transforms as Transforms
import glob
import cv2

import numpy as np

from model.ASFEGNet import ASFEGNet

if __name__ == "__main__":
    print('Starting test...')
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # 加载网络，图片3通道，分类为1。

    net = ASFEGNet(input_nbr=3, label_nbr=1)
    # 将网络拷贝到device
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('bestASFEGNet_model_final.pth', map_location=device))
    # 测试模式
    net.eval()
    trans = Transforms.Compose([Transforms.ToTensor()])
    # 读取所有图片路径
    tests1_path = glob.glob('data/ablation_experiment/test/image1/*.png')
    tests2_path = glob.glob('data/ablation_experiment/test/image2/*.png')
    #label_path = glob.glob('\*.png')
    t=1
    for tests1_path, tests2_path in zip(tests1_path, tests2_path):
        save_res_path = tests1_path.split('.')[0] + '_res_test_base.png'
        save_cmi_path = tests1_path.split('.')[0] + '_cmi_test_base.png'

        save_cmi_path = save_cmi_path.replace('image1', 'cmi_result')
        save_res_path = save_res_path.replace('image1', 'result')

        # name = tests1_path.split('/')[3].split('\\')[1].split('.')[0]
        # 读取图片
        test1_img = cv2.imread(tests1_path)
        test2_img = cv2.imread(tests2_path)
        print(test2_img.shape)
        #label_img = cv2.imread(label_path)
        #label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        test1_img = trans(test1_img)
        test2_img = trans(test2_img)
        test1_img = test1_img.unsqueeze(0)
        test2_img = test2_img.unsqueeze(0)
        test1_img = test1_img.to(device=device, dtype=torch.float32)
        test2_img = test2_img.to(device=device, dtype=torch.float32)
        # 将tensor拷贝到device中：有gpu就拷贝到gpu，否则就拷贝到cpu
        # 预测
        # 使用网络参数，输出预测结果
        pred1 = net(test1_img, test2_img)
        #print(list(pred1))

        pred_Img1 = torch.sigmoid(pred1)
        #print(list(pred_Img1))
        # 提取结果
        pred1 = torch.sigmoid(pred1)
        pred1 = np.array(pred1.data.cpu()[0])[0]
        pred_Img1 = np.array(pred_Img1.data.cpu()[0])[0]




        #保存变化幅度图
        # k = 255/(np.max(pred1)-np.min(pred1))
        # pred1_cmi = [k*(x-np.min(pred1)) for x in pred1]
        # #print(pred1_s)
        # cv2.imwrite(save_cmi_path,np.transpose(pred1_cmi))
        pred_cmi1 = np.uint8(pred_Img1 * 255.0)
        cv2.imwrite(save_cmi_path, pred_cmi1)

        # 处理结果
        pred1[pred1 >= 0.5] = 255
        pred1[pred1 < 0.5] = 0

        # 保存图片
        #x1,x2,x3,x4 = pixel_sum(pred1, pred2, pred3, pred4)
        #cv2.imwrite(save_res_path, x1/(x1+x2+x3+x4)*pred1+x2/(x1+x2+x3+x4)*pred2+x3/(x1+x2+x3+x4)*pred3+x4/(x1+x2+x3+x4)*pred4)
        cv2.imwrite(save_res_path,pred1)