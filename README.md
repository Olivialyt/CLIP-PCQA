<h1 align="center"> CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for Point Cloud Quality Assessment</h1>
<div align="center"> Yating Liu<sup>*</sup>, Yujie Zhang<sup>*</sup>, Ziyu Shan, Yiling Xu<sup>†</sup>

<div align="center"> Shanghai Jiao Tong University</small>
<div align="center"> <small><sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp <sup>†</sup>Corresponding author</small>
<div align="center"> AAAI2025</small>
<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32607" target='_**blank**'><img src="https://img.shields.io/badge/AAAI2025 paper-blue?"></a> 
  <a href="https://arxiv.org/abs/2501.10071" target='_**blank**'><img src="https://img.shields.io/badge/arXiv paper-2501.10071-green?"></a> 
</p>
</div>
</div>
</div>
</div>

## 🚩 Introduction
In recent years, No-Reference Point Cloud Quality Assessment (NR-PCQA) research has achieved significant progress. However, existing methods mostly seek a direct mapping function from visual data to the Mean Opinion Score (MOS), which is contradictory to the mechanism of practical subjective evaluation. To address this, we propose a novel language-driven PCQA method named CLIP-PCQA. Considering that human beings prefer to describe visual quality using discrete quality descriptions (e.g., "excellent" and "poor") rather than specific scores, we adopt a retrieval-based mapping strategy to simulate the process of subjective assessment. More specifically, based on the philosophy of CLIP, we calculate the cosine similarity between the visual features and multiple textual features corresponding to different quality descriptions, in which process an effective contrastive loss and learnable prompts are introduced to enhance the feature extraction. Meanwhile, given the personal limitations and bias in subjective experiments, we further covert the feature similarities into probabilities and consider the Opinion Score Distribution (OSD) rather than a single MOS as the final target. Experimental results show that our CLIP-PCQA outperforms other State-Of-The-Art (SOTA) approaches.
<p align="center">
<img width="866" alt="framework" src="https://github.com/user-attachments/assets/a307bb9e-d8c8-4699-9d71-074c263c0b09">
</p>

## 🛠️ Installation
All experiments are conducted on Ubuntu 20.04 and CUDA 12.4.
```
conda create --name clip_pcqa python=3.8

conda activate clip_pcqa

git clone https://github.com/Olivialyt/CLIP-PCQA.git

cd CLIP-PCQA

pip install -r requirements.txt
```

## 📦 Data Preparation
We provide the download links of the projected images, which can be accessed here ([BaiduYunpan](https://pan.baidu.com/s/1jgkaA7GYp6VZuONwPBq65g?pwd=0jgx), [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/olivialyt_sjtu_edu_cn/Eie4YnUaMzhPvNF9et0z4LgBauVcFQi5cUb3wtMtWTcUyw?e=dBTc8b)).

You can also run `get_projections.py` to generate the projected color maps and depth maps. Note that you need to install `pytorch3d` to use this script.

By unzipping the files, you should get the file structure like:
```
├── LS-PCQA_maps
│   ├── 6view
│   │   ├── Wood_Octree_3_view_0.png
│   │   ├── Wood_Octree_3_view_1.png
│   │   ├── Wood_Octree_3_view_2.png
...
│   ├── 6view_depth
│   │   ├── Wood_Octree_3_view_0.png
│   │   ├── Wood_Octree_3_view_1.png
│   │   ├── Wood_Octree_3_view_2.png
...
```

Given the limited size of the databases, k-fold cross-validation is employed to provide a more accurate estimate of the proposed method's performance. We partition the databases according to content (reference point clouds), and the K-fold Training and Test Set Files are provided in the `csvfiles` folder.

## 🚆 Training
Take LS-PCQA for example, you can simply train the CLIP-PCQA by referring to train.sh with the following command:
```
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
--learning_rate 0.000004 \
--batch_size  16 \
--database LS_PCQA_part  \
--img_length_read 6 \
--data_dir_color ./dataset/LS-PCQA_maps/6view \
--data_dir_depth ./dataset/LS-PCQA_maps/6view_depth \
--num_epochs 50 \
--k_fold_num 5 \
>> logs/LS_PCQA_part.log
```
Then change the path of `data_dir_color` and `data_dir_depth` to `path.../LS-PCQA_maps/6view` and `path.../LS-PCQA_maps/6view_depth`, respectively.

If you want to train the model on the complete database, simply run the provided script `train_total.sh`.

## 🧺 Pretrained models
We provide the pretrained models trained on three datasets with available raw opinion scores: SJTU-PCQA, LS-PCQA Part I and BASICS. The pretrained models can be downloaded here ([BaiduYunpan](https://pan.baidu.com/s/1TA3A4ScB_81y49gVvLQRlg?pwd=rmag), [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/olivialyt_sjtu_edu_cn/EcNVcC-1WLdPoNWzcmPuWrMB9Lk2E2EGexwhC0GkS6INCA?e=85gutB)).

Using these pretrained models, you can evaluate cross-database generalizability by running `cross_validation.py`.

## 📈 Results
| **Metric** | SJTU-PCQA | WPC | LS-PCQA | BASICS |
|:-:|:-:|:-:|:-:|:-:|
| PLCC | 0.956 | 0.894 | 0.755 | 0.932 |
| SRCC | 0.936 | 0.890 | 0.736 | 0.872 |
| RMSE | 0.693 | 10.112 | 0.533 | 0.382 |
 
## 📖 Acknowlegement
This repo is built upon several opensourced codebases, shout out to them for their amazing works.
* ([MM-PCQA](https://github.com/zzc-1998/MM-PCQA))
* ([CoOp](https://github.com/KaiyangZhou/CoOp))
* ([CLIP](https://github.com/openai/CLIP))

## 🔖 Citation
If you find this work useful in your research, please cite
```
@article{liu2025clip,
  title={CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for Point Cloud Quality Assessment},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Liu, Yating and Zhang, Yujie and Shan, Ziyu and Xu, Yiling},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/32607},
  year={2025},
  month={Apr.},
  volume={39},
  number={6},
  pages={5694-5702}
}
```

## 🔍 Bugs
If you find any bugs in this repo, please let me know!
