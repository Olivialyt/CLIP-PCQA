import os, argparse, time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from lightly.loss.ntx_ent_loss import NTXentLoss

class EMD_Quan_Con_Loss(torch.nn.Module):
    def __init__(self, score_list, **kwargs):
        super(EMD_Quan_Con_Loss, self).__init__()
        self.score_list = score_list
        self.quantile_weight = 1 / len(self.score_list)
        self.align_loss_weight = 0.08
        self.contrastive_criterion = NTXentLoss(temperature = 0.07)

    def forward(self, texture_img_f, depth_img_f, original_scores, predicted_cdf):

        original_cdf = self.calculate_CDF(original_scores)
        predicted_cdf = predicted_cdf.to(original_cdf.device)

        emd_loss = torch.sqrt(torch.square(torch.abs(predicted_cdf - original_cdf)).mean(dim=1, keepdim=False))

        batch_size = original_scores.size(0)
        nodes = torch.tensor([self.score_list] * batch_size, dtype=torch.float32)
        theta = torch.tensor([[0.25, 0.5, 0.75]] * batch_size, dtype=torch.float32)

        original_theta_quantile = self.linear_interpolation(nodes, original_cdf, theta)
        predicted_theta_quantile = self.linear_interpolation(nodes, predicted_cdf, theta)

        quantile_loss = torch.abs(predicted_theta_quantile - original_theta_quantile).mean(dim=1, keepdim=False)

        image_loss = self.contrastive_criterion(texture_img_f, depth_img_f)

        total_loss = emd_loss.sum() + quantile_loss.sum() * self.quantile_weight + \
                        image_loss * self.align_loss_weight

        return total_loss

    def calculate_CDF(self, tensor):
        nodes = torch.tensor(self.score_list, dtype=torch.float32)

        nan_count_per_row = torch.sum(torch.isnan(tensor), dim=1).unsqueeze(-1)
        sorted_tensor, _ = torch.sort(tensor, dim=1)

        cum_probs = (sorted_tensor.unsqueeze(2) <= nodes).float().mean(dim=1) / (tensor.size(1) - nan_count_per_row) * tensor.size(1)
        return cum_probs

    def linear_interpolation(self, x, y, y_new):
        # Calculate slopes (dy/dx)
        slopes = (y[:, 1:] - y[:, :-1]) / (x[:, 1:] - x[:, :-1])

        idx = torch.searchsorted(y, y_new)
        zeros_tensor = torch.zeros_like(idx)
        idx_modified = torch.where(idx == 0, 1, idx)

        x_left = torch.where((idx - 1) < 0, x.gather(1, zeros_tensor), x.gather(1, idx_modified - 1))
        y_left = torch.where((idx - 1) < 0, y.gather(1, zeros_tensor), y.gather(1, idx_modified - 1))

        idx_modified = torch.where(idx_modified == y.size(1), y.size(1)-1, idx_modified)
        slope_left = torch.where((idx - 1) < 0, slopes.gather(1, zeros_tensor), slopes.gather(1, idx_modified - 1))

        x_new = torch.where(y_new < y_left, 0.0, x_left + (y_new - y_left) / slope_left)
        y_max = torch.max(y)
        x_max = torch.max(x)
        x_new = torch.where(y_new > y_max, x_max, x_new)

        zero_slope_mask = slope_left == 0
        x_new = torch.where(zero_slope_mask, x_left, x_new)

        return x_new