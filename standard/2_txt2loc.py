# -*- coding: utf-8 -*-
import cv2
import numpy as np
# img = cv2.imread("demo_b.jpg")
# print(img[0].shape)
# print(np.max(img))

def bingc(path):
    f = open(path,"r",encoding='gb18030')
    data = f.readlines() 
    f.close() 
    dd=[]
    for i in data[1:]:
        i=i.strip('\n')
        a=i.split('\t')
        dd.append(a)
    out = []
    for i in dd:
        out.append([float(i[-2]),float(i[-1])])
    #print(out)
    print(len(out))
    out.sort(key=lambda x:(x[0],x[1]))    
    f1=open('../data1106/bing.txt','w')
    for i in out:
        f1.write(str(i[0])+" "+str(i[1])+'\n')
    f1.close()
    return out
    a=[]
    b=[]
    for i in out:
        a.append(i[0])
        b.append(i[1])

    x0 = 111.7462203704
    y0 = 30.99877963#9987592596  # 30.8446194696+7567*2.037e-5
    # print(int((30.9987592596-30.8446194696)/(2.037*0.00001)))
    bing = []
    for i in out:
        bing.append([int((y0-i[1])/(2.037*0.00001)),int((i[0]-x0)/(2.037*0.00001))])
    # print(bing)
    return bing

def main():
    p='../data1106/lzz.txt'
    bing=bingc(p)
    print(bing)
    return 0


if __name__ == '__main__':
    main()




