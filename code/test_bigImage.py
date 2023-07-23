from osgeo import gdal
import numpy as np
import datetime
import math
import sys
import segmentation_models_pytorch as smp
import torch
import cv2
from torchvision import transforms as T
import torch.nn as nn
from nets.unetplusplus import UnetPlusPlus
from nets.hrnetv2 import hrnetv2
from nets.hrnetv2_ocr import hrnetv2_ocr
from loss.SoftBCE import SoftBCELoss
import torch.utils.data as D

# 读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    # bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    # data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    # return data, geotrans, proj
    return geotrans, proj

# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
        
        
 #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                          j * (256 - SideLength * 2) : j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                      (img.shape[1] - 256) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256) : img.shape[0],
                      j * (256-SideLength*2) : j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256) : img.shape[0],
                  (img.shape[1] - 256) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, 0 : 256-RepetitiveLength] = img[0 : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength : 256, 0 : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength] = img[RepetitiveLength : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 256 - RepetitiveLength, 256 -  RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[256 - ColumnOver : 256, 256 - RowOver : 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 256 - RepetitiveLength, 256 - RowOver : 256]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[256 - ColumnOver : 256, RepetitiveLength : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]
    return result

class MyModel(nn.Module):
    def __init__(self, decoder_name, in_channels, classes):
        super().__init__()
        if  decoder_name=="HR-Net":
            self.model = hrnetv2(
                n_class=classes, 
                activation = "sigmoid",
                pretrained_model="pretrained_model/hrnetv2_w48-imagenet.pth",
                )
        elif decoder_name=="HR-Net_OCR":
            self.model = hrnetv2_ocr(
                n_class=classes, 
                activation = "softmax",
                pretrained_model="pretrained_model/hrnetv2_w48-imagenet.pth",
                )
        self.name = decoder_name
    def forward(self, x):
        out = self.model(x)
        return out

class decoder_names:
    HRNet = "HR-Net"
    HRNetOCR = "HR-Net_OCR"

class OurDataset(D.Dataset):
    def __init__(self, images):
        self.images = images
        self.len = len(images)*len(images[0])
        self.trfm = T.Compose([
            T.ToTensor(),
            ])
    # 获取数据操作
    def __getitem__(self, index):
        
        index_1 = int(index / len(self.images[0]))
        index_2 = int(index % len(self.images[0]))
        image = self.images[index_1][index_2]
        
        image = self.trfm(image)
        return image
    
    # 数据集数量
    def __len__(self):
        return self.len
    
def get_dataloader(images, batch_size, num_workers):
    dataset = OurDataset(images)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

if __name__ == "__main__":
    
    model = MyModel(decoder_name = decoder_names.HRNetOCR, in_channels=3, classes=4)
    
    area_perc = 0.5
    batch_size = 128
    num_workers = 6      #多线程
    
    TifPath =r"/data/test/zhengzhou_maincity.tif"
    model_paths = [
        f"/model/HRNetOCR_model.pth",
        ]
    ResultPath = r"/data/predict/zhengzhou_HRNetOCR_pre.tif"
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2)
    
    #  记录测试消耗时间
    testtime = []
    #  获取当前时间
    starttime = datetime.datetime.now()
    
    
    im_geotrans, im_proj = readTif(TifPath)
    
    big_image = cv2.imread(TifPath, cv2.IMREAD_UNCHANGED)
    big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB)
    
    TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)
    
    endtime = datetime.datetime.now()
    text = "读取tif并裁剪预处理完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
    print(text)
    testtime.append(text)

    # 将模型加载到指定设备DEVICE上
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    model.to(DEVICE)
    
    predicts = []
    test_loader = get_dataloader(TifArray, batch_size, 
                                 num_workers=num_workers)
    
    for image in test_loader:
        image = image.cuda()
        pred = np.zeros((image.shape[0],4,256,256))
        for model_path in model_paths:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                #pred = model(image).cpu().numpy()
                output1, output1_ = model(image)
                pred1 = output1_.cpu().numpy()
        
                
                #pred2 = model(torch.flip(image, [0, 3]))
                output2, output2_ = model(torch.flip(image, [0, 3]))
                pred2 = torch.flip(output2_, [3, 0]).cpu().numpy()
        
                #pred3 = model(torch.flip(image, [0, 2]))
                output3, output3_ = model(torch.flip(image, [0, 3]))
                pred3 = torch.flip(output3_, [2, 0]).cpu().numpy()
                
                pred += pred1 + pred2 + pred3
        
        for i in range(image.shape[0]):
            pred_one = pred[i]
            # pred = pred / (len(model_paths) * 3)
            pred_one = np.argmax(pred_one, axis = 0)
            pred_one = pred_one.astype(np.uint8)
            # print(pred.shape)
            pred_one = pred_one.reshape((256,256))
            predicts.append((pred_one))
    
    endtime = datetime.datetime.now()
    text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
    print(text)
    testtime.append(text)

    #保存结果predictspredicts
    result_shape = (big_image.shape[0], big_image.shape[1])
    result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)
    writeTiff(result_data, im_geotrans, im_proj, ResultPath)
    
    endtime = datetime.datetime.now()
    text = "结果拼接完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
    print(text)
    testtime.append(text)
    
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    with open('timelog_%s.txt'%time, 'w') as f:
        for i in range(len(testtime)):
            f.write(testtime[i])
            f.write("\r\n")

