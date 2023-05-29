import socket
import random
from PIL import Image
import numpy as np
from math import sqrt


private_key = 0.3

image = Image.open('00035.jpg')
img_fft = np.fft.fft2(image, axes=(0,1))
img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)
# 이미지 변환을 위한 작업
h, w = image.size
h_crop = int(h * sqrt(1.0))
w_crop = int(w * sqrt(1.0))
h_start = h // 2 - h_crop // 2
w_start = w // 2 - w_crop // 2


    
image = np.asarray(image)
image = image.reshape((512,512,3))
print(image[10][10])
    
# 보낼 이미지 불러오기
sending_image = Image.open('sending_img.jpg')
F_transform = np.fft.fft2(image, axes=(0,1))
SI_abs, SI_pha = np.abs(F_transform), np.angle(F_transform)

# 공유하고 있는 이미지와 private key 사용하여 보낼 이미지 변환
SI_abs_ = np.copy(SI_abs)
SI_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
    private_key * img_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - private_key) * SI_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

    

SI_abs = np.fft.ifftshift(SI_abs, axes=(0, 1))    
new_img = SI_abs * (np.e ** (1j * SI_pha))
new_img = np.real(np.fft.ifft2(new_img, axes=(0, 1)))
new_img = np.uint8(np.clip(new_img, 0, 255))
print(new_img[10][10])



F_transform = np.fft.fft2(new_img, axes=(0,1))
abs , pha = np.abs(F_transform), np.angle(F_transform)

abs_ = np.copy(abs)
abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
    private_key * img_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - private_key) * abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

abs = np.fft.ifftshift(abs, axes=(0,1))
nnn = abs * (np.e ** (1j * pha))
nnn = np.real(np.fft.ifft2(nnn, axes=(0,1)))
nnn = np.uint8(np.clip(nnn, 0, 255))
print(nnn[10][10])