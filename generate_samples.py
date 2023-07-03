import cv2
import numpy as np
import os
def divide_img(img):
    h = img.shape[0]
    w = img.shape[1]
    lower_region=np.zeros((h,w,3),np.uint8)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    save_full_path = os.path.join('huakuai.mp4')
    video_write = cv2.VideoWriter(save_full_path, fourcc, 50, (w, h))

    n=40
    m=40
    print('h={},w={},n={},m={}'.format(h,w,n,m))
    dis_h=int(np.floor(h/n))
    dis_w=int(np.floor(w/m))
    num=0
    for i in range(n):
        if i%2==0:
            start=0
            stop=m+1
            step=1
        else:
            start=m
            stop=-1
            step=-1
        for j in range(start,stop,step):
            num+=1
            print('i,j={}{}'.format(i,j))
            sub=img[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1),:]
            if np.mean(sub)>220:continue
            lower_region[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1),:]=sub
            video_write.write(lower_region)
    video_write.release()
img=cv2.imread('A110 PD-L1_pdl1doc_20191004102721_5445784.png')
ori_h,ori_w=img.shape[0],img.shape[1]
img=cv2.resize(img,(ori_h//10,ori_w//10))
divide_img(img)

