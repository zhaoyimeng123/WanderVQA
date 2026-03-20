# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random

from tqdm import tqdm

from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss
from model import modular

from torchvision import transforms
import time


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # distortion
    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat':
        model = modular.ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat(feat_len=8, sr=True, tr=True, dr=True, ar=True,
                                                                           dropout_sp=0.1, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2)

    if config.checkpoint_resume is not None:
        model.load_state_dict(torch.load(config.checkpoint_resume))
        print(f'load from checkpoint_resume: {config.checkpoint_resume}')

    print('The current model is ' + config.model_name)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat':
        model = model.float()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

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

        trainset = VideoDataset_images_with_LP_motion_dist_aes_features(videos_dir, feature_dir, lp_dir, datainfo_train,
                                                                    transformations_train, 'LSVQ_train',
                                                                    config.crop_size,
                                                                    'Fast', dist_dir, aes_dir)

        testset = VideoDataset_images_with_LP_motion_dist_aes_features(videos_dir, feature_dir, lp_dir, datainfo_test,
                                                                   transformations_test, 'LSVQ_test', config.crop_size,
                                                                   'Fast', dist_dir, aes_dir)

        testset_1080p = VideoDataset_images_with_LP_motion_dist_aes_features(videos_dir, feature_dir, lp_dir,
                                                                         datainfo_test_1080p,
                                                                         transformations_test, 'LSVQ_test_1080p',
                                                                         config.crop_size, 'Fast', dist_dir, aes_dir)

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

    old_save_name = None

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, mos, lp, dist, aes, _) in enumerate(train_loader):

            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            dist = dist.to(device)
            aes = aes.to(device)
            labels = mos.to(device).float()

            outputs_stda = model(video, feature_3D, lp, dist, aes)
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
            for i, (video, feature_3D, mos, lp, dist, aes, _) in enumerate(test_loader):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                lp = lp.to(device)
                dist = dist.to(device)
                aes = aes.to(device)
                label[i] = mos.item()
                outputs_stda = model(video, feature_3D, lp, dist, aes)

                y_output_stda[i] = outputs_stda.item()
                if i % 500 == 0:
                    print(f'test {epoch}:  iter: {i}')
            test_PLCC_stda, test_SRCC_stda, test_KRCC_stda, test_RMSE_stda = performance_fit(label, y_output_stda)

            print(
                'Epoch {} completed. The result on the STDA test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1,
                    test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda))

            label_1080p = np.zeros([len(testset_1080p)])

            y_output_stda_1080p = np.zeros([len(testset_1080p)])

            for i, (video, feature_3D, mos, lp, dist, aes, _) in enumerate(test_loader_1080p):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                lp = lp.to(device)
                dist = dist.to(device)
                aes = aes.to(device)
                label_1080p[i] = mos.item()
                outputs_stda_1080p = model(video, feature_3D, lp, dist, aes)
                y_output_stda_1080p[i] = outputs_stda_1080p.item()
                if i % 500 == 0:
                    print(f'test_1080p {epoch}: iter: {i}')

            test_PLCC_stda_1080p, test_SRCC_stda_1080p, test_KRCC_stda_1080p, test_RMSE_stda_1080p = performance_fit(
                label_1080p, y_output_stda_1080p)


            print(
                'Epoch {} completed. The result on the STDA test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1,
                    test_SRCC_stda_1080p, test_KRCC_stda_1080p, test_PLCC_stda_1080p, test_RMSE_stda_1080p))

            if test_SRCC_stda > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC_stda

                best_test_stda = [test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda]
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

    print('Training completed.')
    print(
        'The best training result on the STDA test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_test_stda[0], best_test_stda[1], best_test_stda[2], best_test_stda[3]))
    print(
        'The best training result on the STDA test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_test_stda_1080p[0], best_test_stda_1080p[1], best_test_stda_1080p[2], best_test_stda_1080p[3]))

    for name, param in model.state_dict().items():
        if name.startswith('weight'):
            print(f"Parameter name: {name}, Value: {param}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat')

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts_modular')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    parser.add_argument('--loss_type', type=str, default='plcc')

    parser.add_argument('--checkpoint_resume', type=str, default='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat_LSVQ_plcc_NR_vNone_epoch_3_SRCC_0.874231.pth')


    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
