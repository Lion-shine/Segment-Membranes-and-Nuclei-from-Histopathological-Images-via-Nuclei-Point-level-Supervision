import cv2
import math
import cmath
import numpy as np
import os
def draw_harmonic(m=-2,beta=0):
    real = []
    for x_ind, x in enumerate(np.arange(-1, 1, 0.01)):  # real
        x_real = []
        for y_ind, y in enumerate(np.arange(-1, 1, 0.01)):  # complex
            cn = complex(y, x)
            r, hudu = cmath.polar(cn)
            theta = math.degrees(hudu)
            # print(math.cos(theta))
            if x==0 and y==0:
                print(theta)
            phi = 1
            R_r = math.exp(-r * r)
            W = R_r * cmath.exp(complex(0, m*hudu+beta))
            W_real = W.real
            x_real.append(W_real)
        real.append(x_real)
    real_arr = np.array(real)
    real_arr_nor = (real_arr - np.min(real_arr)) / (np.max(real_arr) - np.min(real_arr)) * 255
    real_arr_nor = real_arr_nor.astype(np.uint8)[:, :, np.newaxis]
    heat_img = cv2.applyColorMap(real_arr_nor, cv2.COLORMAP_BONE)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    return heat_img

def change_m(m=30,out_width=800):

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    save_full_path = os.path.join('change_m.mp4')
    video_write = cv2.VideoWriter(save_full_path, fourcc, 5, (out_width, out_width+out_width//4))
    for m in range(-m,m):
        copyright = np.zeros((out_width // 4, out_width, 3), np.uint8)
        cv2.putText(copyright, 'Author:Hansheng Li, Northwest University', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 3)
        cv2.putText(copyright,'m=%d'%m,(out_width//2,out_width//8),cv2.FONT_HERSHEY_PLAIN,6,(255,255,255),3)
        heat_img=draw_harmonic(m)
        heat_img=cv2.resize(heat_img,(out_width,out_width))
        draw=np.concatenate([copyright,heat_img],0)
        video_write.write(draw)
    video_write.release()


change_m(out_width=1600)