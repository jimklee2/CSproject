import socket
import random
from PIL import Image
import numpy as np
from math import sqrt


# 공유하고 있는 이미지 키 정보
image = Image.open('00035.jpg')
img_fft = np.fft.fft2(image, axes=(0,1))
img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)


# 이미지 변환을 위한 작업
h, w = image.size
h_crop = int(h * sqrt(1.0))
w_crop = int(w * sqrt(1.0))
h_start = h // 2 - h_crop // 2
w_start = w // 2 - w_crop // 2

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# 서버 정보
HOST = '127.0.0.1'  # 서버의 IP 주소
PORT = int(input(print('PORT번호를 입력하세요 : ')))

# 서버에 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print('서버에 연결되었습니다.')


while True:
    
    # Cloud로부터 p와 g 값을 공유받음
    p = int(client_socket.recv(1024).decode())
    g = int(client_socket.recv(1024).decode())
    

    
    # 임의의 정수 a를 선택하고 A값 계산
    a = random.randint(1,100)
    A = (g ** a) % p
    
    # Camera로 B값 수신 및 A값 전송
    B = client_socket.recv(1024).decode()
    client_socket.send(str(A).encode())
    
    # Private key 값 계산
    private_key = int(B) ** a % p
    private_key = 1/private_key

    print('private_key 값은 : ', private_key)
    
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
    

    # 변환한 이미지 전송
    
    new_img = new_img.tobytes()    
    client_socket.send((str(len(new_img)).encode()))
    client_socket.send(new_img)
    
    sending_image = np.asarray(sending_image)
    sending_image = sending_image.reshape((512,512,3))
    print(sending_image[3][3])
    

    

    

# 소켓 종료
client_socket.close()