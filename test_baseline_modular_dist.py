import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from tqdm import tqdm

from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_features, VideoDataset_images_with_LP_motion_dist_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporalDistortion_modular_dropout(feat_len=8, sr=True, tr=True, dr=True)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)


    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_modular_dropout':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))

    # 模型使用DataParallel后多了"module."
    # state_dict = torch.load(config.trained_model)
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     new_key = key.replace("module.", "")
    #     new_state_dict[new_key] = value
    # model.load_state_dict(new_state_dict)


    ## training data
    if config.database == 'LiveVQC':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_slowfast'

    elif config.database == 'KoNViD-1k':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_slowfast'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_dist_quality_aware'


    elif config.database == 'CVD2014':
        videos_dir = 'data/cvd2014_image_all_fps1'
        datainfo = 'data/CVD2014_Realignment_MOS.csv'
        lp_dir = 'data/CVD2014_LP_ResNet18'
        feature_dir = 'data/CVD2014_slowfast'

    elif config.database == 'youtube_ugc':
        videos_dir = 'data/youtube_ugc_image_all_fps05'
        datainfo = 'data/youtube_ugc_data.mat'
        lp_dir = 'data/youtube_ugc_LP_ResNet18'
        feature_dir = 'data/youtube_ugc_slowfast'

    elif config.database == 'LIVEYTGaming':
        videos_dir = 'data/liveytgaming_image_all_fps1'
        datainfo = 'data/LIVEYTGaming.mat'
        lp_dir = 'data/liveytgaming_LP_ResNet18'
        feature_dir = 'data/LIVEYTGaming_slowfast'


    elif config.database == 'LBVD':
        videos_dir = 'data/LBVD_image_all_fps1'
        datainfo = 'data/LBVD_data.mat'
        lp_dir = 'data/LBVD_LP_ResNet18'
        feature_dir = 'data/LBVD_slowfast'


    elif config.database == 'LIVE-Qualcomm':
        videos_dir = 'data/livequalcomm_image_all_fps1'
        datainfo = 'data/LIVE-Qualcomm_qualcommSubjectiveData.mat'
        lp_dir = 'data/livequalcomm_LP_ResNet18'
        feature_dir = 'data/LIVE-Qualcomm_slowfast'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC), #transforms.Resize(config.resize), 
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = VideoDataset_images_with_LP_motion_dist_features(videos_dir, feature_dir, lp_dir, datainfo, transformations_test, config.database, config.crop_size, 'Fast', dist_dir)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset)])
        y_output_b = np.zeros([len(testset)])
        y_output_s = np.zeros([len(testset)])
        y_output_t = np.zeros([len(testset)])
        y_output_d = np.zeros([len(testset)])
        y_output_std = np.zeros([len(testset)])
        for i, (video, feature_3D, mos, lp, dist, video_name) in tqdm(enumerate(test_loader)):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            dist = dist.to(device)
            label[i] = mos.item()
            outputs_b, outputs_s, outputs_t, outputs_d, outputs_std = model(video, feature_3D, lp, dist)
            y_output_b[i] = outputs_b.item()
            y_output_s[i] = outputs_s.item()
            y_output_t[i] = outputs_t.item()
            y_output_d[i] = outputs_t.item()
            y_output_std[i] = outputs_std.item()


        test_PLCC_b, test_SRCC_b, test_KRCC_b, test_RMSE_b = performance_fit(label, y_output_b)
        test_PLCC_s, test_SRCC_s, test_KRCC_s, test_RMSE_s = performance_fit(label, y_output_s)
        test_PLCC_t, test_SRCC_t, test_KRCC_t, test_RMSE_t = performance_fit(label, y_output_t)
        test_PLCC_d, test_SRCC_d, test_KRCC_d, test_RMSE_d = performance_fit(label, y_output_d)
        test_PLCC_std, test_SRCC_std, test_KRCC_std, test_RMSE_std = performance_fit(label, y_output_std)

        print(config.database)
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
        print('The result on the D test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_d, test_KRCC_d, test_PLCC_d, test_RMSE_d))
        print('The result on the STD test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_std, test_KRCC_std, test_PLCC_std, test_RMSE_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='KoNViD-1k')
    parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='ViTbCLIP_SpatialTemporalDistortion_modular_dropout')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str, default='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_modular_dropout_LSVQ_plcc_NR_vNone_epoch_29_SRCC_0.883130.pth')
    parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224) 

    config = parser.parse_args()

    main(config)
