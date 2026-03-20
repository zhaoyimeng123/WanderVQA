# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random


from tqdm import tqdm

from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features, \
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss
from model import modular

from torchvision import transforms
import time

def save_checkpoint(model, optimizer, scheduler, epoch, old_save_name, path='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/checkpoint_v10_param.pth'):
    """保存模型的checkpoint"""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'old_save_name': old_save_name
    }
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, scheduler, path='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/checkpoint_v10_param.pth'):
    """加载模型的checkpoint"""
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        old_save_name = checkpoint['old_save_name']
        print(f"Checkpoint loaded, resuming from epoch {epoch}")
        return epoch, old_save_name
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0, None  # 从第0个epoch开始

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # distortion
    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param':
        model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(feat_len=8, sr=True, tr=True, dr=True, ar=True,
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

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param':
        model = model.float()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
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

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    start_epoch, old_save_name = load_checkpoint(model, optimizer, scheduler,
                                  path='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/checkpoint_v10_param.pth')

    ## training data
    if config.database == 'LSVQ':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'
        datainfo_test = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
        datainfo_test_1080p = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'

        transformations_train = transforms.Compose(
            # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        feature_dir = '/data/dataset/LSVQ_SlowFast_feature'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lsvq_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'

        trainset = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, feature_dir, lp_dir, datainfo_train,
                                                                    transformations_train, 'LSVQ_train',
                                                                    config.crop_size,
                                                                    'Fast', dist_dir, aes_dir, videomae_feat)

        testset = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, feature_dir, lp_dir, datainfo_test,
                                                                   transformations_test, 'LSVQ_test', config.crop_size,
                                                                   'Fast', dist_dir, aes_dir, videomae_feat)

        testset_1080p = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, feature_dir, lp_dir,
                                                                         datainfo_test_1080p,
                                                                         transformations_test, 'LSVQ_test_1080p',
                                                                         config.crop_size, 'Fast', dist_dir, aes_dir, videomae_feat)

        ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
                                                    shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SROCC min
    best_test_stda = []
    best_test_stda_1080p = []

    print('Starting training:')

    for epoch in range(start_epoch, config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, mos, lp, dist, aes, videomae, _) in enumerate(train_loader):

            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            dist = dist.to(device)
            aes = aes.to(device)
            videomae = videomae.to(device)
            labels = mos.to(device).float()

            outputs_stda = model(video, feature_3D, lp, dist, aes, videomae)
            optimizer.zero_grad()

            loss_st = criterion(labels, outputs_stda)

            loss = loss_st

            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                      (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()

            label = np.zeros([len(testset)])
            y_output_stda = np.zeros([len(testset)])
            for i, (video, feature_3D, mos, lp, dist, aes, videomae, _) in enumerate(test_loader):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                lp = lp.to(device)
                dist = dist.to(device)
                aes = aes.to(device)
                videomae = videomae.to(device)
                label[i] = mos.item()
                outputs_stda = model(video, feature_3D, lp, dist, aes, videomae)

                y_output_stda[i] = outputs_stda.item()
                if i % 1000 == 0:
                    print(f'test {epoch}: iter: {i}')
            test_PLCC_stda, test_SRCC_stda, test_KRCC_stda, test_RMSE_stda = performance_fit(label, y_output_stda)

            print(
                'Epoch {} completed. The result on the STDA test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1,
                    test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda))

            if test_SRCC_stda > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC_stda

                best_test_stda = [test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda]

                if epoch > 20:
                    label_1080p = np.zeros([len(testset_1080p)])
                    y_output_stda_1080p = np.zeros([len(testset_1080p)])
                    for i, (video, feature_3D, mos, lp, dist, aes, videomae, _) in enumerate(test_loader_1080p):
                        video = video.to(device)
                        feature_3D = feature_3D.to(device)
                        lp = lp.to(device)
                        dist = dist.to(device)
                        aes = aes.to(device)
                        videomae = videomae.to(device)
                        label_1080p[i] = mos.item()
                        outputs_stda_1080p = model(video, feature_3D, lp, dist, aes, videomae)
                        y_output_stda_1080p[i] = outputs_stda_1080p.item()
                        if i % 1000 == 0:
                            print(f'test_1080p {epoch}: iter: {i}')
                    test_PLCC_stda_1080p, test_SRCC_stda_1080p, test_KRCC_stda_1080p, test_RMSE_stda_1080p = performance_fit(
                        label_1080p, y_output_stda_1080p)
                    print(
                        'Epoch {} completed. The result on the STDA test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                            epoch + 1,
                            test_SRCC_stda_1080p, test_KRCC_stda_1080p, test_PLCC_stda_1080p, test_RMSE_stda_1080p))

                    best_test_stda_1080p = [test_SRCC_stda_1080p, test_KRCC_stda_1080p, test_PLCC_stda_1080p,
                                            test_RMSE_stda_1080p]

                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                                               config.database + '_' + config.loss_type + '_NR_v' + str(config.exp_version) \
                                               + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC_stda))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name

        # checkpoint save
        save_checkpoint(model, optimizer, scheduler, epoch, old_save_name)

    print('Training completed.')
    print(
        'The best training result on the STDA test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_test_stda[0], best_test_stda[1], best_test_stda[2], best_test_stda[3]))
    print(
        'The best training result on the STDA test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_test_stda_1080p[0], best_test_stda_1080p[1], best_test_stda_1080p[2], best_test_stda_1080p[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param')

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=None)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=200)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts_modular')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])


    parser.add_argument('--loss_type', type=str, default='plcc')

    parser.add_argument('--checkpoint_resume', type=str, default=None)

    config = parser.parse_args()

    torch.manual_seed(3407)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
