import torch.utils.data as D
from torchvision import transforms as T
import numpy as np
import torch
import albumentations as A
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


@torch.no_grad()
# 计算验证集IoU
def cal_val_IoU(model, loader):
    val_IoU = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        #output = model(image)
        output1, output2 = model(image)
        #output1 = output1.argmax(1)
        output = output2.argmax(1)
        IoU = cal_IoU(output, target)
        val_IoU.append(IoU)
    return val_IoU

# 计算IoU
#括号里的数字代表类别数
def cal_IoU(pred, mask):
    ious = []
    for i in range(4):
        p = (mask == i).int().reshape(-1)
        t = (pred == i).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        #  0.0001防止除零
        IoU = 2*overlap/(uion + 0.0001)
        ious.append(IoU.abs().data.cpu().numpy())
        
    return np.stack(ious)

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.transform = A.Compose([
    
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == "train":
            
            label = cv2.imread(self.label_paths[index],0)
            
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
  
            #标签是灰色图像1-4,计算机识别从0开始
            # label = label-1
            
            #标签是彩色图像
            label[label==29] = 0
            label[label==76] = 1
            label[label==150] = 2
            label[label==226] = 3
            
            # label = label.reshape((1,) + label.shape)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index],0)
            
#            label = label-1
            
            label[label==29] = 0
            label[label==76] = 1
            label[label==150] = 2
            label[label==226] = 3
            
            # label = label.reshape((1,) + label.shape)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":
            return self.as_tensor(image), self.image_paths[index]
    
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, batch_size,
                   shuffle, num_workers):
    
    dataset = OurDataset(image_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

