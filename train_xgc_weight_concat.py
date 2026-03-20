# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from data_loader import VideoDataset_images_with_LP_motion_features, \
    VideoDataset_images_with_LP_motion_dist_aes_features, VideoDataset_images_with_LP_motion_dist_aes_videomae_features, \
    xgc_VideoDataset_images_with_dist_videomae_features, xgc_weight_concat_features
from model import modular
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss


def main(config):
    all_test_SRCC_stda, all_test_KRCC_stda, all_test_PLCC_stda, all_test_RMSE_stda = [], [], [], []

    for i in range(10):
        config.exp_version = i
        print('%d round training starts here' % i)
        seed = i * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == 'Xgc_weight_concat':
            model = modular.Xgc_weight_concat(feat_len=8)

        print('The current model is ' + config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.model_name == 'Xgc_weight_concat':
            model = model.float()

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

        # optimizer
        # optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

        optimizer = optim.AdamW(model.parameters(), lr=config.conv_base_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=1e-7
        )

        if config.loss_type == 'plcc':
            criterion = plcc_loss
        elif config.loss_type == 'plcc_rank':
            criterion = plcc_rank_loss
        elif config.loss_type == 'L2':
            criterion = nn.MSELoss().to(device)
        elif config.loss_type == 'L1':
            criterion = nn.L1Loss().to(device)

        elif config.loss_type == 'Huberloss':
            criterion = nn.HuberLoss().to(device)

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        ## training data
        if config.database == 'xgc':
            videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
            datainfo_train = '/data/dataset/XGC-dataset/Share/cvprw-datasets/train/train.txt'
            datainfo_val = '/data/dataset/XGC-dataset/DATA/DATA3/yl/data/cvprw_dataset/dataset_final/val.txt'

            transformations_train = transforms.Compose(
                [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.RandomCrop(config.crop_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transformations_test = transforms.Compose(
                [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.CenterCrop(config.crop_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_dist_quality_aware'
            videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/xgc_VideoMAE_feat'
            aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_aes'
            slowfast_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_ugc_slowfast'
            lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_LP_ResNet18'



            trainset = xgc_weight_concat_features(videos_dir, datainfo_train, transformations_train, 'xgc_train',
                                                   config.crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir, "SlowFast", lp_dir, seed)
            valset = xgc_weight_concat_features(videos_dir, datainfo_train, transformations_test, 'xgc_val',
                                                   config.crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir, "SlowFast", lp_dir, seed)
        elif config.database == 'LSVQ':
            videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
            datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'
            datainfo_val = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'

            transformations_train = transforms.Compose(
                # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
                [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transformations_test = transforms.Compose(
                [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
            videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'
            aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lsvq_aes'
            slowfast_dir = '/data/dataset/LSVQ_SlowFast_feature'
            lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18'

            random = np.random.randint(42)

            trainset = xgc_weight_concat_features(videos_dir, datainfo_train, transformations_train, 'LSVQ_train',
                                                  config.crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir,
                                                  "SlowFast", lp_dir, seed + random)
            valset = xgc_weight_concat_features(videos_dir, datainfo_val, transformations_test, 'LSVQ_test',
                                                config.crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir,
                                                "SlowFast", lp_dir, seed + random)


        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                  shuffle=False, num_workers=config.num_workers)

        best_test_criterion = -1  # SROCC min
        best_test_stda = []

        print('Starting training:')

        old_save_name = None

        patience, wait = 5, 0

        for epoch in range(config.epochs):
            model.train()
            batch_losses1 = []
            batch_losses2 = []
            batch_losses3 = []
            batch_losses4 = []
            batch_losses5 = []
            batch_losses6 = []
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (video, mos1, mos2, mos3, mos4, mos5, mos6, dist, videomae, _, transformed_aes, slowfast_feat, lp_feat) in enumerate(train_loader):
                video = video.to(device)
                dist = dist.to(device)
                videomae = videomae.to(device)
                transformed_aes = transformed_aes.to(device)
                slowfast_feat = slowfast_feat.to(device)
                lp_feat = lp_feat.to(device)
                labels1 = mos1.to(device).float()
                labels2 = mos2.to(device).float()
                labels3 = mos3.to(device).float()
                labels4 = mos4.to(device).float()
                labels5 = mos5.to(device).float()
                labels6 = mos6.to(device).float()

                outputs_stda = model(video, slowfast_feat, lp_feat, dist, transformed_aes, videomae)

                optimizer.zero_grad()

                loss_st1 = criterion(labels1, outputs_stda[:, 0])
                loss_st2 = criterion(labels2, outputs_stda[:, 1])
                loss_st3 = criterion(labels3, outputs_stda[:, 2])
                loss_st4 = criterion(labels4, outputs_stda[:, 3])
                loss_st5 = criterion(labels5, outputs_stda[:, 4])
                loss_st6 = criterion(labels6, outputs_stda[:, 5])
                loss = loss_st1 + loss_st2 + loss_st3 + loss_st4 + loss_st5 + loss_st6
                batch_losses1.append(loss_st1.item())
                batch_losses2.append(loss_st2.item())
                batch_losses3.append(loss_st3.item())
                batch_losses4.append(loss_st4.item())
                batch_losses5.append(loss_st5.item())
                batch_losses6.append(loss_st6.item())
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                loss.backward()
                optimizer.step()

                if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                          (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss1 = sum(batch_losses1) / (len(trainset) // config.train_batch_size)
            avg_loss2 = sum(batch_losses2) / (len(trainset) // config.train_batch_size)
            avg_loss3 = sum(batch_losses3) / (len(trainset) // config.train_batch_size)
            avg_loss4 = sum(batch_losses4) / (len(trainset) // config.train_batch_size)
            avg_loss5 = sum(batch_losses5) / (len(trainset) // config.train_batch_size)
            avg_loss6 = sum(batch_losses6) / (len(trainset) // config.train_batch_size)
            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print(
                'Epoch %d averaged training loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4: %.4f, loss5: %.4f, loss6: %.4f, loss_total: %.4f, '
                % (epoch + 1, avg_loss1, avg_loss2, avg_loss3, avg_loss4, avg_loss5, avg_loss6, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))

            # do validation after each epoch
            with torch.no_grad():
                model.eval()

                labels1 = np.zeros([len(valset)])
                labels2 = np.zeros([len(valset)])
                labels3 = np.zeros([len(valset)])
                labels4 = np.zeros([len(valset)])
                labels5 = np.zeros([len(valset)])
                labels6 = np.zeros([len(valset)])
                y_output_stda1 = np.zeros([len(valset)])
                y_output_stda2 = np.zeros([len(valset)])
                y_output_stda3 = np.zeros([len(valset)])
                y_output_stda4 = np.zeros([len(valset)])
                y_output_stda5 = np.zeros([len(valset)])
                y_output_stda6 = np.zeros([len(valset)])
                for i, (video, mos1, mos2, mos3, mos4, mos5, mos6, dist, videomae, _, transformed_aes, slowfast_feat, lp_feat) in enumerate(val_loader):
                    video = video.to(device)
                    dist = dist.to(device)
                    videomae = videomae.to(device)
                    transformed_aes = transformed_aes.to(device)
                    slowfast_feat = slowfast_feat.to(device)
                    lp_feat = lp_feat.to(device)
                    labels1[i] = mos1.to(device).float()
                    labels2[i] = mos2.to(device).float()
                    labels3[i] = mos3.to(device).float()
                    labels4[i] = mos4.to(device).float()
                    labels5[i] = mos5.to(device).float()
                    labels6[i] = mos6.to(device).float()

                    outputs_stda = model(video, slowfast_feat, lp_feat, dist, transformed_aes, videomae)

                    y_output_stda1[i] = outputs_stda[:, 0].item()
                    y_output_stda2[i] = outputs_stda[:, 1].item()
                    y_output_stda3[i] = outputs_stda[:, 2].item()
                    y_output_stda4[i] = outputs_stda[:, 3].item()
                    y_output_stda5[i] = outputs_stda[:, 4].item()
                    y_output_stda6[i] = outputs_stda[:, 5].item()
                    if i % 1000 == 0:
                        print(f'test {epoch}: iter: {i}')

                test_PLCC_stda1, test_SRCC_stda1, test_KRCC_stda1, test_RMSE_stda1 = performance_fit(labels1, y_output_stda1)
                test_PLCC_stda2, test_SRCC_stda2, test_KRCC_stda2, test_RMSE_stda2 = performance_fit(labels2, y_output_stda2)
                test_PLCC_stda3, test_SRCC_stda3, test_KRCC_stda3, test_RMSE_stda3 = performance_fit(labels3, y_output_stda3)
                test_PLCC_stda4, test_SRCC_stda4, test_KRCC_stda4, test_RMSE_stda4 = performance_fit(labels4, y_output_stda4)
                test_PLCC_stda5, test_SRCC_stda5, test_KRCC_stda5, test_RMSE_stda5 = performance_fit(labels5, y_output_stda5)
                test_PLCC_stda6, test_SRCC_stda6, test_KRCC_stda6, test_RMSE_stda6 = performance_fit(labels6, y_output_stda6)

                print(
                    'Epoch {} completed. The result on the STDA test loss_1 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda1, test_KRCC_stda1, test_PLCC_stda1, test_RMSE_stda1))
                print(
                    'Epoch {} completed. The result on the STDA test loss_2 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda2, test_KRCC_stda2, test_PLCC_stda2, test_RMSE_stda2))
                print(
                    'Epoch {} completed. The result on the STDA test loss_3 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda3, test_KRCC_stda3, test_PLCC_stda3, test_RMSE_stda3))
                print(
                    'Epoch {} completed. The result on the STDA test loss_4 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda4, test_KRCC_stda4, test_PLCC_stda4, test_RMSE_stda4))
                print(
                    'Epoch {} completed. The result on the STDA test loss_5 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda5, test_KRCC_stda5, test_PLCC_stda5, test_RMSE_stda5))
                print(
                    'Epoch {} completed. The result on the STDA test loss_6 databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda6, test_KRCC_stda6, test_PLCC_stda6, test_RMSE_stda6))

                test_SRCC_stda = (test_SRCC_stda1+test_SRCC_stda2+test_SRCC_stda3+test_SRCC_stda4+test_SRCC_stda5+test_SRCC_stda6)/6
                test_PLCC_stda = (test_PLCC_stda1+test_PLCC_stda2+test_PLCC_stda3+test_PLCC_stda4+test_PLCC_stda5+test_PLCC_stda6)/6
                test_KRCC_stda = (test_KRCC_stda1+test_KRCC_stda2+test_KRCC_stda3+test_KRCC_stda4+test_KRCC_stda5+test_KRCC_stda6)/6
                test_RMSE_stda = (test_RMSE_stda1+test_RMSE_stda2+test_RMSE_stda3+test_RMSE_stda4+test_RMSE_stda5+test_RMSE_stda6)/6
                test_criterion = test_SRCC_stda
                ##########todo 使用6个平均loss作为总loss作为模型选取依据##############################
                if test_criterion > best_test_criterion:
                    print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                    best_test_criterion = test_criterion

                    best_test_stda = [test_SRCC_stda, test_PLCC_stda, test_KRCC_stda, test_RMSE_stda]

                    print('Saving model...')
                    if not os.path.exists(config.ckpt_path):
                        os.makedirs(config.ckpt_path)

                    if epoch > 0:
                        if os.path.exists(old_save_name):
                            os.remove(old_save_name)

                    save_model_name = os.path.join(config.ckpt_path, config.database + '_round_' + str(seed) +
                                                   '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC_stda5))
                    torch.save(model.state_dict(), save_model_name)
                    old_save_name = save_model_name

            # checkpoint save
            # save_checkpoint(model, optimizer, scheduler, epoch, old_save_name)

        print('Training completed.')
        print(
            'The best training result on the STDA test dataset SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_test_stda[0], best_test_stda[1], best_test_stda[2], best_test_stda[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', default='xgc', type=str)
    parser.add_argument('--model_name', default='Xgc_weight_concat', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', default=1, type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts_xgc_depth_conv')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])

    parser.add_argument('--loss_type', type=str, default='plcc_rank')

    parser.add_argument('--trained_model', type=str, default='/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_depth_conv/LSVQ_round_0_epoch_5_SRCC_0.876486.pth')
    config = parser.parse_args()

    torch.manual_seed(3407)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
