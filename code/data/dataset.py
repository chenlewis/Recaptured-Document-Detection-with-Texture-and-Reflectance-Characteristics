
from torch.utils import data
import os
from torchvision.transforms import transforms as T
from PIL import Image
import random

class Copy_Detection(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        super(Copy_Detection, self).__init__()
        self.test = test

        #加载路径
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        random.shuffle(imgs)
        # if self.test:l
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]))
        # # else:
        #     if len(imgs.split('-')) == 1:
        #         imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]), reverse=True) #reverse=true 从高到低进行排序
        #     elif len(imgs.split('-')) == 2:
        #         imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1].split('-')[-2]), reverse=True)
        imgs_num = len(imgs)
        # 划分训练集和测试集 验证：训练 = 1：9
        # if self.test:
        #     self.imgs = imgs[int(0.9*imgs_num):]
        # elif train:
        #     self.imgs = imgs[:int(0.8*imgs_num)]
        # else:
        #     self.imgs = imgs[int(0.8*imgs_num):int(0.9*imgs_num)]

        # if self.test:
        #     self.imgs = imgs
        # elif train:
        #     self.imgs = imgs[:int(0.9*imgs_num)]
        # else:
        #     self.imgs = imgs[int(0.9*imgs_num):]

        self.imgs = imgs #跨库训练

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
            )
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize((224,224)),       #???
                    T.CenterCrop(224),
                    # T.RandomHorizontalFlip(), #以0.5的概率进行水平翻转
                    # T.RandomRotation(45),
                    # T.ColorJitter(brightness=1), #随机从0-2 之间亮度变化， 1表示原图
                    # T.ColorJitter(contrast=1), #随机从0-2 之间对比度变化， 1表示原图
                    # T.ColorJitter(hue=0.5), #随机从 -0.5-0.5之间对颜色变化
                    # T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                    # T.RandomVerticalFlip(), #以0.5的概率进行垂直翻转
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((224,224)),
                    # T.CenterCrop(224),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = img_path.split('.')[-2].split('/')[-1]  #测试集数据上的名称
        else:
            label = 1 if len(img_path.split('/')[-1].split('_')) == 3 else 0
            # label = 1 if len(img_path.split('/')[-1].split('_')) == 6 else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)