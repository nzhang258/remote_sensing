# coding=utf-8
from osgeo import gdal
from gdalconst import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2
import numpy as np
import crop

def readTifImage(img_path):
    data = []
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return data
    else:
        print("Open image file success.")
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")
        for i in range(bands_num):
            # 获取影像的第i+1个波段
            band_i = dataset.GetRasterBand(i + 1)
            # 读取第i+1个波段数据
            band_data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
            data.append(band_data)
            print("band " + (i + 1).__str__() + " read success.")
        return data

def writeTif(bands, path):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

img = readTifImage('bch3_GeoTiff.tif')
img_array = np.array(img)
ii = img_array[:3,:,:]
print(ii.shape)
i0 = np.transpose(ii,[1,2,0])
# scipy.misc.imsave('bbbb.png',i0)
cv2.imwrite('b111.png',i0)



