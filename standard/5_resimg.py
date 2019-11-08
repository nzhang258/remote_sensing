import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2
import numpy as np
from libtiff import TIFF
from scipy import misc
import crop

b = cv2.imread('zn0428.png')
# b = scipy.misc.imread('p.png')
print(b.shape)
b0 = np.transpose(b, [2, 0, 1])
print(b0.shape)
c, h, w = b0.shape
sub1 = b0[1]
print('bbbb',b0[:,500,2100])



nn = 0
mm = 0
k_size = 50



# ###############################################################################################################3
ccc = b0
# for i in range(h):
#     for j in range(w):
#         print(i,j)
#         ccc[0,i,j] = min(255,ccc[0,i,j])
#         ccc[1,i,j] = min(255,ccc[0,i,j])
#         ccc[2,i,j] = min(255,ccc[0,i,j])
# for i in range(h):
#     for j in range(w):
#         ccc[0,i,j] = 65#[65,105,225]
#         ccc[1,i,j] = 105
#         ccc[2,i,j] = 225
# print(ccc)
# img = np.transpose(ccc,[1,2,0])
# scipy.misc.imsave('hhh.jpg', img)

f = open('./res0429/my1_10.txt','r')
a = f.readlines()
f.close()

a1 = []
for i in a:
    a1.append(i.split('/')[-1].split('.')[0])
print(len(a1))
cell = 10
cell1 = 25
numm = 0


# bing = crop.bingc("mws.txt")
# bing1 = sorted(bing,key=lambda x: (x[0],x[1]))
# bing2 =[bing1[0]]
# for i in bing1:
#     if i==bing2[-1]:
#         continue
#     bing2.append(i)
# numm0 = 0
# for i in bing2:
#     print(numm0)
#     numm0+=1
#     cx = int(i[0])
#     cy = int(i[1])
#     uu = cx - cell1
#     dd = cx + cell1
#     ll = cy - cell1
#     rr = cy + cell1
#     for ii in range(uu,dd):
#         for jj in range(ll,rr):
#             ccc[0,ii,jj] = 255
#             ccc[1,ii,jj] = 0
#             ccc[2,ii,jj] = 0

for i in a1:
    print(numm)
    numm+=1
    cx = int(i.split('_')[0])
    cy = int(i.split('_')[1])
    uu = cx - cell
    dd = cx + cell
    ll = cy - cell
    rr = cy + cell
    for ii in range(uu,dd):
        for jj in range(ll,rr):
            ccc[0,ii,jj] = 255
            ccc[1,ii,jj] = 255
            ccc[2,ii,jj] = 0



img = np.transpose(ccc,[1,2,0])
cv2.imwrite('./res0429/res01.png', img)



# #################################################################################################

# bing = crop.bingc("mws.txt")
# bing1 = sorted(bing,key=lambda x: (x[0],x[1]))
# bing2 =[bing1[0]]
# for i in bing1:
#     if i==bing2[-1]:
#         continue
#     bing2.append(i)

# num = 0
# for i in bing2:
#     cx = i[0]
#     cy = i[1]
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
#     scipy.misc.imsave('./val/pos/'+str(cx)+'_'+str(cy)+'.jpg', k_img)
#     num+=1
#     print(num)
