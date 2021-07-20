import cv2
from multiprocessing import Process
import numpy as np


im_src = cv2.imread('./ch_b1.png')


point_list = []
def func1():

    a, b = int(753.7), int(153.7)
    
    
    

def func2():
    # im_src = cv2.imread('./ch_b1.png')
    # dst2 = im_src.copy()
    # dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    a, b = int(1400.7), int(153.7)
    
    
    
def p_for(a, b, i):
    print(i)
    car_width = 4  # 지도 차량 크기 변수 (2.5m : 64px)
    car_length = 9 # 6m

    a = a + 50*i
    cv2.rectangle(dst2, (a - car_length, b - car_width//2), (a , b + car_width//2), (0, 0, -255), car_width)
    
    cv2.imshow('image', dst2)
    cv2.waitKey(97)
        

if __name__ == '__main__':

    # 프로세스를 생성합니다
    now = []
    now.append((1,2))
    now.append((3,4))
    print(len(now))
    
    for i in range(10):
        dst2 = im_src.copy()
        dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        a, b = int(753.7), int(153.7)
        p1 = Process(target=p_for(a=a, b=b, i=i)) #함수 1을 위한 프로세스
        a, b = int(1400.7), int(153.7)
        p2 = Process(target=p_for(a=a, b=b, i=i)) #함수 2을 위한 프로세스

        # start로 각 프로세스를 시작합니다. func1이 끝나지 않아도 func2가 실행됩니다.
        p1.start()
        p2.start()

        # join으로 각 프로세스가 종료되길 기다립니다 p1.join()이 끝난 후 p2.join()을 수행합니다
        p1.join()
        p2.join()

    