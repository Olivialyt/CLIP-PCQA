import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from models.CLIPPCQA_Net import CLIPPCQA_Net
from utils.CLIPPCQA_Dataset import CLIPPCQA_Dataset
from utils.loss import EMD_Quan_Con_Loss
import math
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.nn import functional as F

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # fix the random seed

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=60, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.000004, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--data_dir_color', default='', type=str, help = 'path to the images')
    parser.add_argument('--data_dir_depth', default='', type=str, help = 'path to the depth images')
    parser.add_argument('--img_length_read', default=4, type=int, help = 'number of the using images')
    parser.add_argument('--database', default='SJTU', type=str)
    args = parser.parse_args()

    return args

def extend_args(args):
    """
    Add new config variables.
    """
    args.n_ctx = 16  # number of context vectors
    args.csc = True  # class-specific context
    args.ctx_init = ""  # initialization words
    args.prec = "fp32"  # fp16, fp32, amp
    args.class_token_position = "middle"  # 'middle' or 'end' or 'front'
    args.subsample_classes = "all"  # all, base or new

if __name__=='__main__':
    print('*************************************************************************************************************************')
    args = parse_args()
    extend_args(args)

    set_rand_seed()
    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    img_length_read = args.img_length_read
    data_dir_color = args.data_dir_color
    data_dir_depth = args.data_dir_depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    
    quality_classes =['bad', 'poor', 'fair', 'good', 'excellent'] 

    if database == 'SJTU':           
        train_filename_list = 'csvfiles/sjtu_data_info/total.csv'
        raw_score_path = 'csvfiles/sjtu_data_info/total_raw_score.csv'
        score_list = [2.0, 4.0, 6.0, 8.0, 10.0]

    elif database == 'LS_PCQA_part':
        train_filename_list = 'csvfiles/ls_pcqa_data_info/total.csv'
        raw_score_path = 'csvfiles/ls_pcqa_data_info/total_raw_score.csv'
        score_list = [1.0, 2.0, 3.0, 4.0, 5.0]

    elif database == 'BASICS':
        train_filename_list = 'csvfiles/basics_data_info/total.csv'
        raw_score_path = 'csvfiles/basics_data_info/total_raw_score.csv'
        score_list = [0.0, 1.0, 2.0, 3.0, 4.0]

    elif database == 'WPC':
        train_filename_list = 'csvfiles/wpc_data_info/total.csv'
        raw_score_path = 'csvfiles/wpc_data_info/total_raw_score.csv'
        score_list = [0.0, 1.0, 2.0, 3.0, 4.0]

    transformations_train = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    print('Trainging set: ' + train_filename_list)
    
    # load the network
    model = CLIPPCQA_Net(device, args, score_list, quality_classes)
    model = model.to(device)
    
    criterion = EMD_Quan_Con_Loss(score_list).to(device)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        if "visual" in name:
            param.requires_grad_(True)

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = args.learning_rate, weight_decay=args.decay_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.decay_rate)
    print('Using Adam optimizer, initial learning rate: ' + str(args.learning_rate))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f'Using dataset: {database}')

    print("Ready to train network")
    print('*************************************************************************************************************************')

    min_training_loss = 10000
    
    train_dataset = CLIPPCQA_Dataset(data_dir_color = data_dir_color, data_dir_depth = data_dir_depth, datainfo_path = train_filename_list, raw_score_path = raw_score_path, transform = transformations_train, img_length_read = img_length_read)

    for epoch in range(num_epochs):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        model.train()
        start = time.time()
        batch_losses = []
        batch_losses_each_disp = []

        for i, (imgs, depth_imgs, mos, original_scores) in enumerate(train_loader):
            imgs = imgs.to(device)
            depth_imgs = depth_imgs.to(device)
            mos = mos[:,np.newaxis]
            mos = mos.to(device)
            texture_f, depth_f, quality_score, pred_dis, pred_CDF = model(imgs, depth_imgs)

            # compute loss
            loss = criterion(texture_f, depth_f, original_scores, pred_CDF)
            loss.requires_grad_(True)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train

            torch.autograd.backward(loss)
            optimizer.step()

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr_current = scheduler.get_last_lr()
        print('The current learning rate is {:.10f}'.format(lr_current[0]))

        end = time.time()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end-start))

        if avg_loss < min_training_loss:
            print('--------------------------------')
            print("Update best model using best_val_criterion ")
            torch.save(model.state_dict(), 'total_ckpts/' + database + '_best_model.pth')
            min_training_loss = avg_loss
