import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features, \
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param':
        model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(feat_len=8, sr=True, tr=True, dr=True, ar=True,
                                                                           dropout_sp=0.1, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))
    print('success load the trained model')

    ## training data
    if config.database == 'LiveVQC':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LiveVQC_VideoMAE_feat'

    elif config.database == 'KoNViD-1k':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KoNViD_1k_VideoMAE_feat'

    elif config.database == 'CVD2014':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_Realignment_MOS.csv'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/CVD204_VideoMAE_feat'

    elif config.database == 'youtube_ugc':
        videos_dir = 'data/youtube_ugc_image_all_fps05'
        datainfo = 'data/youtube_ugc_data.mat'
        lp_dir = 'data/youtube_ugc_LP_ResNet18'
        feature_dir = 'data/youtube_ugc_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_dist_quality_aware'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/YouTubeUGC_VideoMAE_feat'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_aes'

    elif config.database == 'LIVEYTGaming':
        videos_dir = 'data/liveytgaming_image_all_fps1'
        datainfo = 'data/LIVEYTGaming.mat'
        lp_dir = 'data/liveytgaming_LP_ResNet18'
        feature_dir = 'data/LIVEYTGaming_slowfast'


    elif config.database == 'LBVD':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lbvd_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lbvd_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LBVD_VideoMAE_feat'


    elif config.database == 'LIVE-Qualcomm':
        videos_dir = 'data/livequalcomm_image_all_fps1'
        datainfo = 'data/LIVE-Qualcomm_qualcommSubjectiveData.mat'
        lp_dir = 'data/livequalcomm_LP_ResNet18'
        feature_dir = 'data/LIVE-Qualcomm_slowfast'

    elif config.database == 'LSVQ_test':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18'
        feature_dir = '/data/dataset/LSVQ_SlowFast_feature'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_dist_quality_aware'
        aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/lsvq_aes'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         # transforms.Resize(config.resize),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, config.database, config.crop_size,
                                                                   'Fast', dist_dir, aes_dir, videomae_feat)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()

        label = np.zeros([len(testset)])
        y_output_stda = np.zeros([len(testset)])
        for i, (video, feature_3D, mos, lp, dist, aes, videomae, video_name_str) in enumerate(test_loader):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            dist = dist.to(device)
            aes = aes.to(device)
            videomae = videomae.to(device)
            label[i] = mos.item()
            outputs_stda = model(video, feature_3D, lp, dist, aes, videomae)
            y_output_stda[i] = outputs_stda.item()
            print(i, video_name_str, 'mos: {}'.format(label[i]), 'predict: {}'.format(y_output_stda[i]))

        test_PLCC_stda, test_SRCC_stda, test_KRCC_stda, test_RMSE_stda = performance_fit(label, y_output_stda)

        print(config.database)
        print(
            'The result on the STDA test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LBVD')  #
    parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_LSVQ_plcc_NR_vNone_epoch_82_SRCC_0.892096.pth')
    parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    config = parser.parse_args()

    main(config)
