import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_random_noise(image, noise_factor=0.5):
    """
    在图像上添加随机噪声。
    :param image: 输入图像
    :param noise_factor: 噪声因子，控制噪声的强度
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0.0, 1.0)  # 将像素值限制在 [0, 1] 范围内
    return noisy_image

# 示例使用
original_image = cv2.imread("D:\\ImageProcess\\8_Experiment_SCEGM-UNet\\code\\8_SCEGM-UNet\\data\\levir-cd\\train\\test_14_B.png")
noisy_image = add_random_noise(original_image)

# 可视化
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.show()
