
import numpy as np
from math import sqrt

def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    # assert는 주어진 조건이 'Ture'가 아니면 'AssertionError'예외를 발생시킴.
    # 이 코드에서 'assert img1.shape = img2.shape 는 img1과 img2의 크기가 같은지 확인하는 것.
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

#   입력 이미지들의 2차원 Fourier 변환을 계산하고, 절댓값과 위상값을 저장
#   fft2() 함수는 2차원 이미지에 대한 2차원 FFT를 계산 -> axes=(0, 1) 인자는 이미지의 세 개의 차원 중에서 첫 번째와 두 번째 차원에 대한 FFT를 계산한다는 것을 의미
    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1)) # 주파수가 0인 부분을 정중앙에 위치시키고 재배열해주는 함수.
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))
    """
    푸리에 변환을 수행하면 주파수 영역에서의 저주파 성분은 영상의 중앙에 위치하고, 고주파 성분은 영상의 주변에 위치
    """

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1)) # 이를 이용해 재배열된 주파수 값들의 위치를 본래대로 되돌림.
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha)) # 
    img12 = img2_abs * (np.e ** (1j * img2_pha)) 
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255)) # np.clip 함수를 사용하여 이미지 값의 범위를 0과 255 사이로 제한
    img12 = np.uint8(np.clip(img12, 0, 255)) # np.unit8로 데이터 타입을 변환하여 반환.

    return img21, img12