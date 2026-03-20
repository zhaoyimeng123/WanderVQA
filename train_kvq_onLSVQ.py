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
    kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features
from model import modular
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss
from fvcore.nn import FlopCountAnalysis


def main(config):
    all_test_SRCC_stda, all_test_KRCC_stda, all_test_PLCC_stda, all_test_RMSE_stda = [], [], [], []

    for i in range(10):
        config.exp_version = i
        print('%d round training starts here' % i)
        seed = i * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == 'kvq_onLSVQ':
            model = modular.kvq_onLSVQ(feat_len=4)

        print('The current model is ' + config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.model_name == 'kvq_onLSVQ':
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

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

        # 计算 FLOPs
        # 创建三个输入张量
        in_video = torch.randn(1, 4, 3, 224, 224)
        in_dist = torch.randn(1, 4, 4096)
        in_videomae = torch.randn(1, 4, 1408)
        flop_analyzer = FlopCountAnalysis(model, (in_video.to(device), in_dist.to(device), in_videomae.to(device)))
        flops = flop_analyzer.total()
        # 检查是否低于 120G FLOPs
        print(f"Total FLOPs: {flops:.2e}")  # 以科学计数法输出
        print('Total GFLOPs: %.2f ' % (flops / 1e9))

        transformations_train = transforms.Compose(
            # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  transforms.Resize(config.resize)
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size),
             transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ## training data
        if config.database == 'KVQ':
            random = np.random.randint(low=0, high=42)
            videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_image_all_fps1'
            datainfo = '/data/dataset/KVQ/train_data.csv'
            dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_dist_quality_aware'
            videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KVQ/train/kvq_VideoMAE_feat'

            trainset = kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, datainfo,
                                                                          transformations_train, 'KVQ_train',
                                                                          config.crop_size, dist_dir, videomae_feat, random)

            valset = kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, datainfo,
                                                                          transformations_test, 'KVQ_val',
                                                                          config.crop_size, dist_dir, videomae_feat, random)
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
            videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'

            trainset = kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, datainfo_train,
                                                                                            transformations_train,
                                                                                            'LSVQ_train',
                                                                                            config.crop_size, dist_dir,
                                                                                            videomae_feat)

            testset = kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, datainfo_test,
                                                                                           transformations_test,
                                                                                           'LSVQ_test',
                                                                                           config.crop_size, dist_dir,
                                                                                           videomae_feat)
            valset = testset



        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                  shuffle=False, num_workers=config.num_workers)

        best_test_criterion = -1  # SROCC min
        best_test_stda = []

        print('Starting training:')

        old_save_name = None


        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (video, mos, dist, videomae, _) in tqdm(enumerate(train_loader)):
                video = video.to(device)
                dist = dist.to(device)
                videomae = videomae.to(device)
                labels = mos.to(device).float()

                outputs_stda = model(video, dist, videomae)

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
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
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
                label = np.zeros([len(valset)])
                y_output_stda = np.zeros([len(valset)])
                for i, (video, mos, dist, videomae, video_name_str) in tqdm(enumerate(val_loader)):
                    video = video.to(device)
                    dist = dist.to(device)
                    videomae = videomae.to(device)
                    label[i] = mos.item()
                    outputs_stda = model(video, dist, videomae)
                    y_output_stda[i] = outputs_stda.item()
                test_PLCC_stda, test_SRCC_stda, test_KRCC_stda, test_RMSE_stda = performance_fit(label, y_output_stda)
                print(
                    'Epoch {} completed. The result on the STDA test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda))

                if test_SRCC_stda > best_test_criterion:
                    new_save_model_name = os.path.join(config.ckpt_path, config.database + '_' + str(seed) + '_' + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC_stda))
                    torch.save(model.state_dict(), new_save_model_name)
                    print("Saving model......")
                    print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                    best_test_criterion = test_SRCC_stda
                    best_test_stda = [test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda]

                    if epoch > 0:
                        if os.path.exists(old_save_name):
                            os.remove(old_save_name)

                    old_save_name = new_save_model_name

        print('Training completed.')
        print(
            'The best training result on the STDA test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_test_stda[0], best_test_stda[1], best_test_stda[2], best_test_stda[3]))
        all_test_SRCC_stda.append(best_test_stda[0])
        all_test_KRCC_stda.append(best_test_stda[1])
        all_test_PLCC_stda.append(best_test_stda[2])
        all_test_RMSE_stda.append(best_test_stda[3])
    print(
        'The STDA median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            np.median(all_test_SRCC_stda), np.median(all_test_KRCC_stda), np.median(all_test_PLCC_stda),
            np.median(all_test_RMSE_stda)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', default='LSVQ', type=str)
    parser.add_argument('--model_name', default='kvq_onLSVQ', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', default=1, type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=60)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts_kvq/final')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])

    parser.add_argument('--loss_type', type=str, default='plcc_rank')

    parser.add_argument('--trained_model', type=str,
                        default='/data/user/zhaoyimeng/ModularBVQA/ckpts_kvq/final/LSVQ_0__epoch_4_SRCC_0.867214.pth')

    config = parser.parse_args()

    torch.manual_seed(3407)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
