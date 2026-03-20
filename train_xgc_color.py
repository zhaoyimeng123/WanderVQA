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
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features, xgc_VideoDataset_images_with_dist_videomae_features
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss, performance_no_fit
from model import modular

from torchvision import transforms
import time

def save_checkpoint(model, optimizer, scheduler, epoch, old_save_name, path='/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_aes/checkpoint.pth'):
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

def load_checkpoint(model, optimizer, scheduler, path=None):
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
    if config.model_name == 'xgc_color':
        model = modular.xgc_color(feat_len=8)

    print('The current model is ' + config.model_name)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.trained_model is not None:
        # load the trained model
        print('loading the pretrained model')
        model.load_state_dict(torch.load(config.trained_model))

        # 模型使用DataParallel后多了"module."
        # state_dict = torch.load(config.trained_model)
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     new_key = key.replace("module.", "")
        #     new_state_dict[new_key] = value
        # model.load_state_dict(new_state_dict)

        print('loading model success!')

    if config.checkpoint_resume is not None:
        model.load_state_dict(torch.load(config.checkpoint_resume))
        print(f'load from checkpoint_resume: {config.checkpoint_resume}')

    if config.model_name == 'xgc_color':
        model = model.float()

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.000001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

    optimizer = optim.AdamW(model.parameters(), lr=config.conv_base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

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
                                  path='/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_aes/checkpoint.pth')

    ## training data
    if config.database == 'xgc':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
        datainfo_train = '/data/dataset/XGC-dataset/Share/cvprw-datasets/train/train.txt'
        datainfo_val = '/data/dataset/XGC-dataset/DATA/DATA3/yl/data/cvprw_dataset/dataset_final/val.txt'


        transformations_train = transforms.Compose(
            # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size),
             transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_dist_quality_aware'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/xgc_VideoMAE_feat'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_aes'

        trainset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo_train,
                                                                    transformations_train, 'xgc_train',
                                                                    config.crop_size, dist_dir, videomae_feat, aes_dir)
        valset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo_train,
                                                                       transformations_test, 'xgc_val',
                                                                       config.crop_size, dist_dir, videomae_feat, aes_dir)
        # testset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo_test,
        #                                                              transformations_test, 'xgc_test',
        #                                                              config.crop_size, dist_dir, videomae_feat)
    elif config.database == 'LSVQ':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'
        datainfo_test = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'

        transformations_train = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lsvq_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'

        trainset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo_train,
                                                                       transformations_train, 'LSVQ_train',
                                                                       config.crop_size, dist_dir, videomae_feat, aes_dir)

        testset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo_test,
                                                                      transformations_test, 'LSVQ_test',
                                                                      config.crop_size, dist_dir, videomae_feat, aes_dir)
        valset = testset
    elif config.database == 'KVQ':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_image_all_fps1'
        datainfo = '/data/dataset/KVQ/train_data.csv'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KVQ/train/kvq_VideoMAE_feat'
        transformations_train = transforms.Compose(
            # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
            [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size),
             transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo,
                                                                       transformations_train, 'KVQ_train',
                                                                       config.crop_size, dist_dir, videomae_feat,
                                                                       aes_dir)
        valset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo,
                                                                     transformations_test, 'KVQ_val',
                                                                     config.crop_size, dist_dir, videomae_feat, aes_dir)

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                               shuffle=False, num_workers=config.num_workers)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                           shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SROCC min
    best_test_stda = []

    print('Starting training:')

    for epoch in range(start_epoch, config.epochs):
        model.train()
        batch_losses1 = []
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, mos1, mos2, mos3, mos4, mos5, mos6, dist, videomae, _, transformed_aes) in tqdm(enumerate(train_loader)):

            video = video.to(device)
            videomae = videomae.to(device)
            transformed_aes = transformed_aes.to(device)
            labels1 = mos1.to(device).float()

            outputs_stda = model(video, videomae, transformed_aes)
            optimizer.zero_grad()

            loss_st1 = criterion(labels1, outputs_stda)


            loss = loss_st1

            batch_losses1.append(loss_st1.item())

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
        avg_loss1 = sum(batch_losses1) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss1: %.4f'% (epoch + 1, avg_loss1))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            labels1 = np.zeros([len(valset)])

            y_output_stda1 = np.zeros([len(valset)])

            for i, (video, mos1, mos2, mos3, mos4, mos5, mos6, dist, videomae, _, transformed_aes) in tqdm(enumerate(val_loader)):
                video = video.to(device)
                videomae = videomae.to(device)
                transformed_aes = transformed_aes.to(device)
                labels1[i] = mos1.to(device).float()
                outputs_stda = model(video, videomae, transformed_aes)
                y_output_stda1[i] = outputs_stda.item()

                if i % 1000 == 0:
                    print(f'test {epoch}: iter: {i}')

            test_PLCC_stda1, test_SRCC_stda1, test_KRCC_stda1, test_RMSE_stda1 = performance_fit(labels1, y_output_stda1)

            print(
                'Epoch {} completed. The result on the STDA test loss_1 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1,
                    test_SRCC_stda1, test_KRCC_stda1, test_PLCC_stda1, test_RMSE_stda1))

            if test_SRCC_stda1 > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC_stda1

                best_test_stda = [test_SRCC_stda1, test_KRCC_stda1, test_PLCC_stda1, test_RMSE_stda1]

                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(config.ckpt_path, config.database + 'color_epoch_%d_SRCC_%f.pth' % (epoch + 1, best_test_criterion))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name

        # checkpoint save
        # save_checkpoint(model, optimizer, scheduler, epoch, old_save_name)

    print('Training completed.')
    print(
        'The best training result on the STDA test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_test_stda[0], best_test_stda[1], best_test_stda[2], best_test_stda[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='xgc')
    parser.add_argument('--model_name', type=str, default='xgc_color')

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=None)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=200)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts_xgc_final')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])


    parser.add_argument('--loss_type', type=str, default='plcc_rank')

    parser.add_argument('--checkpoint_resume', type=str, default=None)

    parser.add_argument('--trained_model', type=str,
                        default=None)

    config = parser.parse_args()

    torch.manual_seed(3407)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
