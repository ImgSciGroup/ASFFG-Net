# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image


def ImageRotate(imagepath):
    image = cv2.imread(imagepath)
    # 要有中心坐标、旋转角度、缩放系数
    h, w = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (w // 2, h // 2)  # 绕图片中心进行旋转
    angle = 90  # 旋转角度
    scale = 0.8  # 将图像缩放为80%

    # 1. 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=scale)  # 当angle为负值时，则表示为顺时针

    # 2. 进行仿射变换，borderValue:缺失背景填充色彩，默认是黑色（0, 0 , 0），这里指定填充白色
    # 注意，这里的dsize=(w, h)顺序不要搞反了
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(w, h), borderValue=(255, 255, 255))

    return image_rotation


def ImageRotate_2(imagepath):
    image = cv2.imread(imagepath)
    # 要有中心坐标、旋转角度、缩放系数
    h, w = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (w // 2, h // 2)  # 绕图片中心进行旋转
    angle = 180  # 旋转角度
    scale = 0.8  # 将图像缩放为80%

    # 1. 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=scale)  # 当angle为负值时，则表示为顺时针

    # -----------------------计算图像的新边界尺寸、调整旋转矩阵以考虑平移-------------------- #
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    # ------------------------------------------------------------------------------ #

    # 2. 进行仿射变换，边界填充为255，即白色，borderValue 缺省，默认是黑色（0, 0 , 0）
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(nW, nH), borderValue=(0, 0, 0))

    return image_rotation


if __name__ == '__main__':
    imagepath = r"D:\ImageProcess\8_Experiment_SCEGM-UNet\code\8_SCEGM-UNet\gabor.jpg"
    img = cv2.imread(imagepath)
    imgRotation = ImageRotate(imagepath)  # 待旋转的原始图像，旋转角度45
    imgRotation_2 = ImageRotate_2(imagepath)  # 待旋转的原始图像，旋转角度45

    # 显示并保存旋转结果
    cv2.imshow("img", img)
    cv2.imshow("imgRotation", imgRotation)  # 图像高宽无变化
    cv2.imshow("imgRotation_2", imgRotation_2)  # 图像高宽有变化，图像大小无太大变化
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('imgRotation_crack.png', imgRotation)
    cv2.imwrite('imgRotation_crack_2.png', imgRotation_2)

    # 查看图像模式是否有变化，查看现在图像高宽
    img = Image.open(imagepath)
    imgRotation_crack = Image.open('imgRotation_crack.png')
    imgRotation_crack_2 = Image.open('imgRotation_crack_2.png')

    print(f'img:                 mode={img.mode}, size={img.size}')
    print(f'imgRotation_crack:   mode={imgRotation_crack.mode}, size={imgRotation_crack.size}')
    print(f'imgRotation_crack_2: mode={img.mode}, size={imgRotation_crack_2.size}')
