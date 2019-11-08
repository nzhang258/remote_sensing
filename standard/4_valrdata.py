import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2
import numpy as np
from libtiff import TIFF
from scipy import misc
import crop
import random


b = cv2.imread('zn0428.png')
# b = scipy.misc.imread('zn0428.png')

b0 = np.transpose(b, [2, 0, 1])
c, h, w = b0.shape
sub1 = b0[1]




nn = 0
mm = 0
k_size = 50

for i in range(1000000):
    cx = random.randint(50,30288)
    cy = random.randint(50,26963)
    
    u = cx-k_size
    d = cx+k_size
    l = cy-k_size
    r = cy+k_size
    if u<0 or d>=h or l<0 or r>=w or (sub1[u][l]==0 and sub1[u][l+1]==0) \
    or (sub1[u][r-1]==0 and sub1[u][r-2]==0) \
    or (sub1[d][l]==0 and sub1[d][l+1]==0) \
    or (sub1[d][r-1]==0 and sub1[d][r-2]==0):
        continue
    kernel = b0[:,u:d,l:r]
    k_img = np.transpose(kernel,[1,2,0])
    scipy.misc.imsave('./val/neg/'+str(cx)+'_'+str(cy)+'.jpg', k_img)
    nn+=1
    if nn>50000:
        break
    print(nn)
    

################################################################################################################3
# ccc = b0
# # for i in range(h):
# #     for j in range(w):
# #         ccc[0,i,j] = 65#[65,105,225]
# #         ccc[1,i,j] = 105
# #         ccc[2,i,j] = 225
# # print(ccc)
# # img = np.transpose(ccc,[1,2,0])
# # scipy.misc.imsave('hhh.jpg', img)

# f = open('./mydata/my1_7.txt','r')
# a = f.readlines()
# f.close()

# a1 = []
# for i in a:
#     a1.append(i.split('/')[-1].split('.')[0])
# print(len(a1))
# cell = 15
# numm = 0
# for i in a1:
#     print(numm)
#     numm+=1
#     cx = int(i.split('_')[0])
#     cy = int(i.split('_')[1])
#     uu = cx - cell
#     dd = cx + cell
#     ll = cy - cell
#     rr = cy + cell
#     for ii in range(uu,dd):
#         for jj in range(ll,rr):
#             ccc[0,ii,jj] = 255
#             ccc[1,ii,jj] = 0
#             ccc[2,ii,jj] = 0

# img = np.transpose(ccc,[1,2,0])
# scipy.misc.imsave('result1.jpg', img)


#################################################################################################





# num = 0
# for i in bing2:
    
#     u = i[0]-k_size
#     d = i[0]+k_size
#     l = i[1]-k_size
#     r = i[1]+k_size
#     if u<0 or d>=h or l<0 or r>=w or (sub1[u][l]==255 and sub1[u][l+1]==255) \
#     or (sub1[u][r-1]==255 and sub1[u][r-2]==255) \
#     or (sub1[d][l]==255 and sub1[d][l+1]==255) \
#     or (sub1[d][r-1]==255 and sub1[d][r-2]==255):
#         # print('sub1',u,l,sub1[u][0])
#         continue
#     kernel = b0[:,u:d,l:r]
#     k_img = np.transpose(kernel,[1,2,0])
#     scipy.misc.imsave('./b_y/'+str(num)+'.jpg', k_img)
#     num+=1
#     print(num)
