import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8, sr=True, tr=True)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)


    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    # model.load_state_dict(torch.load(config.trained_model))

    """
    from fvcore.nn import FlopCountAnalysis
    input1 = torch.randn(1, 8, 3, 224, 224)  # RGB 图像
    input2 = torch.randn(1, 8, 256)
    input3 = torch.randn(1, 8, 1280)
    input1 = input1.to(device)
    input2 = input2.to(device)
    input3 = input3.to(device)
    flops = FlopCountAnalysis(model, (input1, input2, input3))
    print(f"FLOPs: {flops.total() / 1e9} GFLOPs")
    """

    # 模型使用DataParallel后多了"module."
    state_dict = torch.load(config.trained_model)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))


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
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_image_all_fps1'
        datainfo = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_LP_ResNet18'
        feature_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_slowfast'


    elif config.database == 'LIVE-Qualcomm':
        videos_dir = 'data/livequalcomm_image_all_fps1'
        datainfo = 'data/LIVE-Qualcomm_qualcommSubjectiveData.mat'
        lp_dir = 'data/livequalcomm_LP_ResNet18'
        feature_dir = 'data/LIVE-Qualcomm_slowfast'

    elif config.database == 'LSVQ':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        datainfo_test = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
        datainfo_test_1080p = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'
        feature_dir = '/data/dataset/LSVQ_SlowFast_feature'
        lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC), #transforms.Resize(config.resize), 
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo, transformations_test, config.database, config.crop_size, 'Fast')
    #
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                           shuffle=False, num_workers=config.num_workers)



    testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo_test,
                                                          transformations_test, 'LSVQ_test', config.crop_size,
                                                          'Fast')
    testset_1080p = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir,
                                                                datainfo_test_1080p,
                                                                transformations_test, 'LSVQ_test_1080p',
                                                                config.crop_size, 'Fast')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
                                                    shuffle=False, num_workers=config.num_workers)


    with torch.no_grad():
        model.eval()

        label = np.zeros([len(testset)])
        y_output_b = np.zeros([len(testset)])
        y_output_s = np.zeros([len(testset)])
        y_output_t = np.zeros([len(testset)])
        y_output_st = np.zeros([len(testset)])
        for i, (video, feature_3D, mos, lp, video_name) in enumerate(test_loader):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            label[i] = mos.item()
            outputs_b, outputs_s, outputs_t, outputs_st = model(video, feature_3D, lp)
            y_output_b[i] = outputs_b.item()
            y_output_s[i] = outputs_s.item()
            y_output_t[i] = outputs_t.item()
            y_output_st[i] = outputs_st.item()


        test_PLCC_b, test_SRCC_b, test_KRCC_b, test_RMSE_b = performance_fit(label, y_output_b)
        test_PLCC_s, test_SRCC_s, test_KRCC_s, test_RMSE_s = performance_fit(label, y_output_s)
        test_PLCC_t, test_SRCC_t, test_KRCC_t, test_RMSE_t = performance_fit(label, y_output_t)
        test_PLCC_st, test_SRCC_st, test_KRCC_st, test_RMSE_st = performance_fit(label, y_output_st)

        print(config.database)
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st))

        label_1080p = np.zeros([len(testset_1080p)])
        y_output_b_1080p = np.zeros([len(testset_1080p)])
        y_output_s_1080p = np.zeros([len(testset_1080p)])
        y_output_t_1080p = np.zeros([len(testset_1080p)])
        y_output_st_1080p = np.zeros([len(testset_1080p)])
        for i, (video, feature_3D, mos, lp, _) in enumerate(test_loader_1080p):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            lp = lp.to(device)
            label_1080p[i] = mos.item()
            outputs_b_1080p, outputs_s_1080p, outputs_t_1080p, outputs_st_1080p = model(video, feature_3D, lp)
            y_output_b_1080p[i] = outputs_b_1080p.item()
            y_output_s_1080p[i] = outputs_s_1080p.item()
            y_output_t_1080p[i] = outputs_t_1080p.item()
            y_output_st_1080p[i] = outputs_st_1080p.item()
        test_PLCC_b_1080p, test_SRCC_b_1080p, test_KRCC_b_1080p, test_RMSE_b_1080p = performance_fit(label_1080p,
                                                                                                     y_output_b_1080p)
        test_PLCC_s_1080p, test_SRCC_s_1080p, test_KRCC_s_1080p, test_RMSE_s_1080p = performance_fit(label_1080p,
                                                                                                     y_output_s_1080p)
        test_PLCC_t_1080p, test_SRCC_t_1080p, test_KRCC_t_1080p, test_RMSE_t_1080p = performance_fit(label_1080p,
                                                                                                     y_output_t_1080p)
        test_PLCC_st_1080p, test_SRCC_st_1080p, test_KRCC_st_1080p, test_RMSE_st_1080p = performance_fit(
            label_1080p, y_output_st_1080p)

        print(config.database)
        print(
            'The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                test_SRCC_b_1080p, test_KRCC_b_1080p, test_PLCC_b_1080p, test_RMSE_b_1080p))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_s_1080p, test_KRCC_s_1080p, test_PLCC_s_1080p, test_RMSE_s_1080p))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_t_1080p, test_KRCC_t_1080p, test_PLCC_t_1080p, test_RMSE_t_1080p))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_st_1080p, test_SRCC_st_1080p, test_PLCC_st_1080p, test_RMSE_st_1080p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LSVQ')
    parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='ViTbCLIP_SpatialTemporal_modular_dropout')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str, default='/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth')
    parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=224) 
    parser.add_argument('--crop_size', type=int, default=224) 

    config = parser.parse_args()

    main(config)
