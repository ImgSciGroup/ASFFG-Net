import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image1/*.png'))
        #print(self.imgs_path)
        self.transform = transform
        #print(self.transform)
    def __getitem__(self, index):
        # 根据index读取图像
        image1_path = self.imgs_path[index]
        # 根据image_path生成image2_path
        image2_path = image1_path.replace('image1', 'image2')
        # 根据image_path生成label_path
        label_path = image1_path.replace('image1', 'label')
        # labelgray_path = image1_path.replace('image1', 'label_gray')
        # 读取训练图像和标签
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        label = cv2.imread(label_path)
        # 将图像转为单通道图片
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(labelgray_path, label)
        label = label.reshape(label.shape[0], label.shape[1], 1)
        #image 转化为ToTensor()
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        label = self.transform(label)
        #image = torch.cat([image1, image2], dim=0)
        return image1, image2, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

if __name__ == "__main__":

    print("*********")
    isbi_dataset = ISBI_Loader(data_path="D:/ImageProcess/3_Multi-scale house change detection/code/29_SCNet-master/SCNet-master/data/WHU/train/",
                               transform=Transforms.ToTensor())
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=False)
    print(len(train_loader))
    i = 0
    for image1, image2, label in train_loader:
        print(label.shape)
        i += 1

