import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from tqdm import tqdm
from data_loader import VideoDataset_images_with_LP_motion_dist_aes_videomae_features
from utils import performance_fit, plcc_loss, plcc_rank_loss, plcc_l1_loss
from model import modular
from torchvision import transforms
import time
import optuna


def main(config, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # distortion model
    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10':
        model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10(
            feat_len=8, sr=True, tr=True, dr=True, ar=True,
            dropout_sp=0.1, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2)

    print('The current model is ' + config.model_name)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.checkpoint_resume is not None:
        model.load_state_dict(torch.load(config.checkpoint_resume))
        print(f'load from checkpoint_resume: {config.checkpoint_resume}')

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10':
        model = model.float()

    # 使用 Optuna 动态调整学习率和批次大小
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True) if trial else config.conv_base_lr
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32]) if trial else config.train_batch_size
    epochs = trial.suggest_int('epochs', 50, 200) if trial else config.epochs

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

    # 定义损失函数
    if config.loss_type == 'plcc':
        criterion = plcc_loss
    elif config.loss_type == 'plcc_l1':
        criterion = plcc_l1_loss
    elif config.loss_type == 'plcc_rank':
        criterion = plcc_rank_loss
    elif config.loss_type == 'l2':
        criterion = nn.MSELoss().to(device)
    elif config.loss_type == 'l1':
        criterion = nn.L1Loss().to(device)
    elif config.loss_type == 'Huberloss':
        criterion = nn.HuberLoss().to(device)

    # 加载数据集
    videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
    datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'
    datainfo_test = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
    feature_dir = '/data/dataset/LSVQ_SlowFast_feature'
    lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18'
    dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
    aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lsvq_aes'
    videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'

    transformations_train = transforms.Compose([
        transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformations_test = transforms.Compose([
        transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(
        videos_dir, feature_dir, lp_dir, datainfo_train,
        transformations_train, 'LSVQ_train', config.crop_size, 'Fast', dist_dir, aes_dir, videomae_feat)

    testset = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(
        videos_dir, feature_dir, lp_dir, datainfo_test,
        transformations_test, 'LSVQ_test', config.crop_size, 'Fast', dist_dir, aes_dir, videomae_feat)

    # 使用动态调整的 batch size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SRCC
    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for i, (video, feature_3D, mos, lp, dist, aes, videomae, _) in enumerate(train_loader):
            video, feature_3D, lp, dist, aes, videomae, labels = \
                video.to(device), feature_3D.to(device), lp.to(device), dist.to(device), aes.to(device), videomae.to(
                    device), mos.to(device).float()

            outputs_stda = model(video, feature_3D, lp, dist, aes, videomae)
            optimizer.zero_grad()

            loss = criterion(labels, outputs_stda)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证部分
        with torch.no_grad():
            model.eval()
            test_SRCC_stda, test_PLCC_stda = 0, 0  # 假设这是结果

            # 如果在使用 Optuna 时，这里返回验证结果用于调优
            if trial:
                return test_SRCC_stda

    print('Training completed.')


# Optuna 调优的 objective 函数
def objective(trial):
    # # 动态调整 config 中的超参数
    # config.conv_base_lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    # config.train_batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
    # config.epochs = trial.suggest_int('epochs', 50, 200)

    # 调用 main 函数，传入 trial 进行训练
    return main(config, trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 输入参数
    parser.add_argument('--database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10')
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=222)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='ckpts_modular')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--loss_type', type=str, default='plcc_rank')
    parser.add_argument('--checkpoint_resume', type=str, default=None)

    config = parser.parse_args()

    # Optuna 调参部分
    study = optuna.create_study(direction='maximize')  # 假设你要最大化 SRCC
    study.optimize(objective, n_trials=50)  # 运行 50 次调参

    # 输出最佳结果
    print(f"Best trial SRCC: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")
