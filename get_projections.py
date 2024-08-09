import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

def normalize_verts(verts):
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid
    max_length = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
    verts_normalized = verts / max_length
    return verts_normalized

def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq)) / math.pi * 180  # theta
    az = math.atan2(y, x) / math.pi * 180  # phi
    return r, elev, az

def rotation_matrix(vec1, vec2):
    vec1, vec2 = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    if np.sum(vec1 == vec2) == 3:
        return np.eye(3)
    if np.sum(vec1 == -vec2) == 3:
        return -np.eye(3)
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.sum(vec1 * vec2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + (1 - c) / (np.square(s)) * np.matmul(vx, vx)
    return R

def find_min_bounding_rect(mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    white_pixel_coords = cv2.findNonZero(mask_image_gray)

    min_x = np.min(white_pixel_coords[:, 0, 0])
    max_x = np.max(white_pixel_coords[:, 0, 0])
    min_y = np.min(white_pixel_coords[:, 0, 1])
    max_y = np.max(white_pixel_coords[:, 0, 1])
    
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

def crop_image_with_mask(original_image, mask_image):
    min_bounding_rect = find_min_bounding_rect(mask_image)
    cropped_image = original_image[
        min_bounding_rect[0][1]:min_bounding_rect[2][1],
        min_bounding_rect[0][0]:min_bounding_rect[2][0]
    ]
    return cropped_image

view_num = '6'
dataset = 'LS-PCQA'
data_dir = f'./dataset/{dataset}/data'

output_img_dir = f'./dataset/{dataset}/proj_{view_num}view/'
output_depth_dir = f'./dataset/{dataset}/proj_{view_num}view_depth/'
output_mask_dir = f'./dataset/{dataset}/proj_{view_num}view_mask/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_depth_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# transform = transforms.CenterCrop(235)
df_total = pd.read_excel(f'./dataset/{dataset}/total.xlsx')

obj_filename_list = list(df_total['path'])
device = torch.device("cuda:1")

if view_num == '6':
    render_view_list = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]

for i, obj_filename in tqdm(enumerate(obj_filename_list), total=len(obj_filename_list), smoothing=0.9, leave=False):
    obj_file_path = os.path.join(data_dir, obj_filename)
    pointcloud = np.load(obj_file_path)
    normalized_verts = normalize_verts(pointcloud[:, :3])

    # for j, cart in enumerate(cart_13_view_list[0]):
    # rotated_verts = np.matmul(normalized_verts, rotation_matrix(np.array([0, 0, 1]), np.array(cart)))
    verts = torch.Tensor(normalized_verts).to(device)
    rgb = torch.Tensor(pointcloud[:, 3:]).to(device)
    if torch.sum(rgb > 1) >= 1:
        rgb = rgb / 255
    pc = Pointclouds(points=[verts], features=[rgb])
    # angle_crop_array = np.zeros((len(render_view_list), 235, 235, 3))

    for k, render_view in enumerate(render_view_list):
        R, T = look_at_view_transform(1, render_view[0], render_view[1])  # 角度超过90会翻转方向
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        raster_settings = PointsRasterizationSettings(
            image_size=1024,
            radius=0.003,
            points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        fragments = rasterizer(pc)
        depth = fragments[1].cpu()
        depth = depth[:,:,:,0]

    # mask map
        binary_mask = (depth != -1).float()
        binary_mask_data = binary_mask.squeeze().cpu().numpy()
        binary_mask_image = Image.fromarray((binary_mask_data * 255).astype(np.uint8))
        mask_savename = output_mask_dir + '_'.join([obj_filename.split('/')[-1].split('.')[0],'view',str(k)+'.png'])
        binary_mask_image.save(mask_savename)

    # depth map
        depth = torch.where(depth == -1, torch.tensor(0.0), depth)
        depth_data = depth.squeeze().cpu().numpy()
        filtered_depth = depth[depth != 0.0]
        min_depth = torch.min(filtered_depth).cpu().numpy()
        max_depth = torch.max(filtered_depth).cpu().numpy()
        depth_data = depth.squeeze().cpu().numpy()
        normalized_depth = (depth_data/max_depth * 255).astype(np.uint8)
        depth_image = Image.fromarray(normalized_depth)
        depth_savename = output_depth_dir + '_'.join([obj_filename.split('/')[-1].split('.')[0],'view',str(k)+'.png'])
        depth_image.save(depth_savename)
    # img_map

        images = renderer(pc)
        img_array = images[0, ..., :3].cpu().numpy()

        # 使用掩码将背景设为白色
        is_background = (binary_mask_data == 0)
        img_array[is_background] = 1  # 将背景像素设为白色

        img_array = (img_array * 255).astype(np.uint8)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_savename = output_img_dir + '_'.join([obj_filename.split('/')[-1].split('.')[0],'view',str(k)+'.png'])
        cv2.imwrite(img_savename, img_array)
    # print(angle_crop_array.shape,angle_crop_array[0])

# crop
final_img_path = f'./dataset/{dataset}/{view_num}view/'
final_depth_path = f'./dataset/{dataset}/{view_num}view_depth/'

os.makedirs(final_img_path, exist_ok = True)
os.makedirs(final_depth_path, exist_ok = True)

for filename in os.listdir(output_img_dir):
    if filename.endswith('.png'):
        image = cv2.imread(os.path.join(output_img_dir, filename))
        depth_image = cv2.imread(os.path.join(output_depth_dir, filename))
        mask_image = cv2.imread(os.path.join(output_mask_dir, filename))

    cropped_image = crop_image_with_mask(image, mask_image)
    cropped_depth_image = crop_image_with_mask(depth_image, mask_image)

    cv2.imwrite(os.path.join(final_img_path, filename), cropped_image)
    cv2.imwrite(os.path.join(final_depth_path, filename), cropped_depth_image)
