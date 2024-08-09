import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image

class CLIPPCQA_Dataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir_color, data_dir_depth, datainfo_path, raw_score_path, transform, crop_size = 224, img_length_read = 6, is_train = True):
        super(CLIPPCQA_Dataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.raw_score = pd.read_csv(raw_score_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo[['name']]
        self.ply_mos = dataInfo['mos']
        self.crop_size = crop_size
        self.data_dir_color = data_dir_color
        self.data_dir_depth = data_dir_depth
        self.transform = transform
        self.img_length_read = img_length_read
        self.length = len(self.ply_name)
        self.is_train = is_train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name = self.ply_name.iloc[idx,0] 
        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
       
        img_length_read = self.img_length_read       
        projected_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        depth_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])

        # read images
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            image_name = os.path.join(self.data_dir_color, img_name.split('.')[0] + '_view_' + str(i) + '.png')
            depth_name = os.path.join(self.data_dir_depth, img_name.split('.')[0] + '_view_' + str(i) + '.png')
            if os.path.exists(image_name) and os.path.exists(depth_name):
                read_frame = Image.open(image_name)
                read_frame = read_frame.convert('RGB')
                depth_frame = Image.open(depth_name)
                depth_frame = depth_frame.convert('RGB')

                width, height = read_frame.size
                if width < self.crop_size and width < height:
                    read_frame = transforms.RandomCrop(width)(read_frame)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
                    depth_frame = transforms.RandomCrop(width)(depth_frame)
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(depth_frame)
                elif height < self.crop_size:
                    read_frame = transforms.RandomCrop(height)(read_frame)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
                    depth_frame = transforms.RandomCrop(height)(depth_frame)
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(depth_frame)

                read_frame = self.transform(read_frame)
                projected_img[i] = read_frame
                depth_frame = self.transform(depth_frame)
                depth_img[i] = depth_frame

                img_read_index += 1

            elif os.path.exists(image_name):
                print(depth_name)
                print('Depth_image do not exist!')
            else:
                print(image_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                projected_img[j] = projected_img[img_read_index-1]
                depth_img[j] = depth_img[img_read_index-1]

        y_mos = self.ply_mos.iloc[idx] 
        y_label = torch.FloatTensor(np.array(y_mos))

        raw_score_tensor_data = self.raw_score[self.raw_score['name'] == img_name].iloc[:, 1:].values.flatten()

        return projected_img, depth_img, y_label, raw_score_tensor_data


