import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2
import numpy as np
from libtiff import TIFF
from scipy import misc
import 2_txt2loc

b = scipy.misc.imread('p.png')
print(b.shape)
b0 = np.transpose(b, [2, 0, 1])
print(b0.shape)
c, h, w = b0.shape
sub1 = b0[1]


top = 0
tmp = []
for i in range(1000):
    a = np.argmin(sub1[i])
    # print(i,a)
    tmp.append(a)
    if a != 0 and tmp[-1] != 0:
        break
top = len(tmp)
print(top)

down = 0
tmp = []
for i in range(1000):
    a = np.argmin(sub1[h-1-i])
    tmp.append(a)
    if a != 0 and tmp[-1] != 0:
        break
down = len(tmp)
print(down)

left = 0
tmp = []
for i in range(1000):
    a = np.argmin(sub1[:, i])
    tmp.append(a)
    if a != 0 and tmp[-1] != 0:
        break
left = len(tmp)
print(left)

right = 0
tmp = []
for i in range(1000):
    a = np.argmin(sub1[:, w-1-i])
    tmp.append(a)
    if a != 0 and tmp[-1] != 0:
        break
right = len(tmp)
print(right)


bing = 2_txt2loc.bingc("mws.txt")
bing1 = sorted(bing, key=lambda x: (x[0], x[1]))
bing2 = [bing1[0]]

for i in bing1:
    if i == bing2[-1]:
        continue
    bing2.append(i)
print(len(bing2))

nn = 0
mm = 0
k_size = 50
thresh = 10
for i in range(k_size,h,k_size):
    for j in range(k_size,w,2*k_size):
        cx = i
        cy = j
        tu = cx-thresh
        td = cx+thresh
        tl = cy-thresh
        tr = cy+thresh

        u = cx-k_size
        d = cx+k_size
        l = cy-k_size
        r = cy+k_size
        if u<0 or d>=h or l<0 or r>=w or (sub1[u][l]==255 and sub1[u][l+1]==255) \
        or (sub1[u][r-1]==255 and sub1[u][r-2]==255) \
        or (sub1[d][l]==255 and sub1[d][l+1]==255) \
        or (sub1[d][r-1]==255 and sub1[d][r-2]==255):
            continue
        flag = True
        for it in bing2:
            if it[0]<tu:
                continue
            if it[0]>td:
                break
            if it[1]<tl:
                continue
            if it[1]>tr:
                continue
            flag = False
            break
        if flag:
            
            kernel = b0[:,u:d,l:r]
            k_img = np.transpose(kernel,[1,2,0])
            scipy.misc.imsave('./b_n/'+str(nn)+'.jpg', k_img)
            nn+=1
            print(nn)
        
num = 0
for i in bing2:
    
    u = i[0]-k_size
    d = i[0]+k_size
    l = i[1]-k_size
    r = i[1]+k_size
    if u<0 or d>=h or l<0 or r>=w or (sub1[u][l]==255 and sub1[u][l+1]==255) \
    or (sub1[u][r-1]==255 and sub1[u][r-2]==255) \
    or (sub1[d][l]==255 and sub1[d][l+1]==255) \
    or (sub1[d][r-1]==255 and sub1[d][r-2]==255):
        # print('sub1',u,l,sub1[u][0])
        continue
    kernel = b0[:,u:d,l:r]
    k_img = np.transpose(kernel,[1,2,0])
    scipy.misc.imsave('./b_y/'+str(num)+'.jpg', k_img)
    num+=1
    print(num)
