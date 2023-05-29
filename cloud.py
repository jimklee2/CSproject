import socket
import random 
from PIL import Image
import numpy as np
from math import sqrt
import struct
import io
# import cv2 

# 공유하고 있는 이미지 키 정보
image = Image.open('00035.jpg')
img_fft = np.fft.fft2(image, axes=(0,1))
img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)

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

# 소켓 생성, 바인딩, 연결
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()
print('서버가 시작되었습니다.')


# 클라이언트로부터 연결을 받음
client_socket, addr = server_socket.accept()
print('클라이언트가 연결되었습니다:', addr)


while True:
    
    # Cloud에서 소수 p와 p이하의 자연수 g를 선택하여 Camera와 공유
    p = int(input('소수 p를 입력하시오 : '))
    g = int(input('p 이하의 자연수 g를 입력하시오 :'))
    client_socket.send(str(p).encode())
    client_socket.send(str(g).encode())
    
    # 임의의 정수 b를 선택하고 B값 계산
    b = random.randint(1,100)
    B = (g ** b) % p
       
    # Camera로 B값 전송 및 A값 수신
    client_socket.send(str(B).encode())
    A = int(client_socket.recv(1024).decode())
    
    # Private key 값 계산
    private_key = A ** b % p
    private_key = 1/private_key

    print('private_key 값은 : ', private_key)
    
    # 변환된 이미지 수신
    length = client_socket.recv(1024).decode()
    new_img = recvall(client_socket, int(length))    
    new_img = np.frombuffer(new_img, dtype=np.uint8)
    new_img = Image.fromarray(new_img)
    new_img = np.asarray(new_img)
    new_img = new_img.reshape((512,512,3))
    

    # 변환된 이미지를 원래 이미지로 복구
    F_transform = np.fft.fft2(new_img, axes=(0,1))
    ori_abs, ori_pha = np.abs(F_transform), np.angle(F_transform)
    
    ori_abs_ = np.copy(ori_abs)
    ori_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        private_key * img_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1-private_key) * ori_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ori_abs = np.fft.ifftshift(ori_abs, axes=(0, 1))
    new_img = ori_abs * (np.e ** (1j * ori_pha))
    new_img = np.real(np.fft.ifft2(new_img, axes=(0, 1)))
    new_img = np.uint8(np.clip(new_img, 0, 255))
    
    print(new_img[3][3])
    
    
    
    
    
    
    
    
    
  

# 소켓 종료
client_socket.close()
server_socket.close()