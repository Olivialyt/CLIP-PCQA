import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils import data
from PIL import Image
import os, argparse, time
import torch.nn.functional as F
import clip
import numpy as np
import torch.nn as nn
from models.CLIPPCQA_Net import CLIPPCQA_Net
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau
import scipy
from scipy import stats

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--n_ctx', default=16, type=int, help = 'number of context vectors')
    parser.add_argument('--class_token_position', default='middle', type=str, help = "'middle' or 'end' or 'front'")
    args = parser.parse_args()

    return args

def extend_args(args):
    args.csc = True  # class-specific context
    args.ctx_init = ""  # initialization words
    args.prec = "fp32"  # fp16, fp32, amp
    args.subsample_classes = "all"  # all, base or new

args = parse_args()
extend_args(args)

quality_classes =['bad', 'poor', 'fair', 'good', 'excellent'] 

# sjtu ls basics
true_csvs = ['csvfiles/sjtu_data_info/total.csv',
             'csvfiles/ls_pcqa_data_info/total.csv',
             'csvfiles/basics_data_info/total.csv']

data_dir_colors =['dataset/SJTU-PCQA/10view',
               'dataset/LS-PCQA/6view',
               'dataset/BASICS/6view']

data_dir_depths = ['dataset/SJTU-PCQA/10view_depth',
                   'dataset/LS-PCQA/6view_depth',
                   'dataset/BASICS/6view_depth']

score_lists = [[2.0, 4.0, 6.0, 8.0, 10.0],
               [1.0, 2.0, 3.0, 4.0, 5.0],
               [0.0, 1.0, 2.0, 3.0, 4.0]]

load_paths = ['./total_ckpts/SJTU_best_model.pth',
              './total_ckpts/LS_PCQA_part_best_model.pth',
              './total_ckpts/BASICS_best_model.pth']

trained_ckpts = ['sjtu', 'ls', 'basics']
test_datasets = ['sjtu', 'ls', 'basics']

for train_index, trained_ckpt in enumerate(trained_ckpts):
    for test_index, test_dataset in enumerate(test_datasets):
        if trained_ckpt == test_dataset:
            continue
        else:
            print('======================================')
            print('trained_ckpt:', trained_ckpt)
            print('test_dataset:', test_dataset)
            true_csv = true_csvs[test_index]
            data_dir_color = data_dir_colors[test_index]
            data_dir_depth = data_dir_depths[test_index]
            score_list = score_lists[test_index]
            load_path = load_paths[train_index]

            dataInfo_true = pd.read_csv(true_csv, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
            ply_name = dataInfo_true['name']
            ply_mos = dataInfo_true['mos'] 

            transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

            n_test = len(ply_mos)
            y_true = np.zeros(n_test)
            y_pre = np.zeros(n_test)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            crop_size = 224
            img_length_read = 6
            model = CLIPPCQA_Net(device, args, score_list, quality_classes)
            model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(load_path).items()})
            model = model.to(device)
            model.eval()
            for idx, single_ply_name in enumerate(ply_name):
                img_name = single_ply_name 

                img_channel = 3
                img_height_crop = crop_size
                img_width_crop = crop_size
                    
                projected_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
                depth_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])

                # read images
                img_read_index = 0
                for i in range(img_length_read):
                    # load images
                    image_name = os.path.join(data_dir_color, img_name.split('.')[0] + '_view_' + str(i) + '.png')
                    depth_name = os.path.join(data_dir_depth, img_name.split('.')[0] + '_view_' + str(i) + '.png')
                    if os.path.exists(image_name) and os.path.exists(depth_name):
                        read_frame = Image.open(image_name)
                        read_frame = read_frame.convert('RGB')
                        depth_frame = Image.open(depth_name)
                        depth_frame = depth_frame.convert('RGB')

                        width, height = read_frame.size
                        if width < crop_size and width < height:
                            read_frame = transforms.RandomCrop(width)(read_frame)
                            read_frame = transforms.Resize((crop_size, crop_size))(read_frame)
                            depth_frame = transforms.RandomCrop(width)(depth_frame)
                            depth_frame = transforms.Resize((crop_size, crop_size))(depth_frame)
                        elif height < crop_size:
                            read_frame = transforms.RandomCrop(height)(read_frame)
                            read_frame = transforms.Resize((crop_size, crop_size))(read_frame)
                            depth_frame = transforms.RandomCrop(height)(depth_frame)
                            depth_frame = transforms.Resize((crop_size, crop_size))(depth_frame)

                        read_frame = transform(read_frame)
                        projected_img[i] = read_frame
                        depth_frame = transform(depth_frame)
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

                y_mos = ply_mos.iloc[idx] 
                y_label = torch.FloatTensor(np.array(y_mos))

                y_true[idx] = y_label

                with torch.no_grad():
                    model.eval()
                    projected_img = projected_img.to(device).unsqueeze(0)
                    depth_img = depth_img.to(device).unsqueeze(0)

                    texture_f, depth_f, quality_score, pred_dis, pred_CDF = model(projected_img, depth_img)
                    score = quality_score.item()
                
                y_pre[idx] = score 

            y_output_logistic = fit_function(y_true, y_pre)
            test_PLCC = stats.pearsonr(y_output_logistic, y_true)[0]
            test_SROCC = stats.spearmanr(y_pre, y_true)[0]
            test_RMSE = np.sqrt(((y_output_logistic-y_true) ** 2).mean())
            test_KROCC = scipy.stats.kendalltau(y_pre, y_true)[0]

            print("Test results: PLCC={:.4f}, SROCC={:.4f}, RMSE={:.4f}, KROCC={:.4f}".format(test_PLCC, test_SROCC, test_RMSE, test_KROCC))
