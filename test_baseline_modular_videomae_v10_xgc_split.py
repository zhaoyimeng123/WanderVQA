import argparse
import numpy as np
import time
import torch
import torch.nn
from torchvision import transforms
from model import modular
from tqdm import tqdm
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features, \
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features, xgc_VideoDataset_images_with_dist_videomae_features


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


    videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
    datainfo = '/data/dataset/XGC-dataset/test.txt'
    dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_dist_quality_aware'
    videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/xgc_VideoMAE_feat'
    aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_aes'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = xgc_VideoDataset_images_with_dist_videomae_features(videos_dir, datainfo,
                                                        transformations_test, config.database,
                                                        config.crop_size, dist_dir, videomae_feat, aes_dir)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        start_time = time.time()

        output_file = "ckpts_xgc_LSVQ_kvq/output_dim6.txt"
        with open(output_file, "w") as f:
            for i, (video, mos1, mos2, mos3, mos4, mos5, mos6, dist, videomae, video_name_str, transformed_aes) in tqdm(enumerate(test_loader)):
                video = video.to(device)
                dist = dist.to(device)
                videomae = videomae.to(device)
                outputs_stda = model(video, None, None, dist, None, videomae)
                print('{}.mp4,{:.4f}'.format(video_name_str[0], outputs_stda.item()))

                # 格式化字符串，使用 "\t" 进行分割
                line = "{}.mp4,{:.4f}\n".format(
                    video_name_str[0], outputs_stda.item()
                )
                f.write(line)


        end_time = time.time()
        print(config.database, end_time - start_time)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='xgc_test')  #
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_LSVQ_kvq/xgc_round_2_dim_6_epoch_10_SRCC_0.841537.pth')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    config = parser.parse_args()

    main(config)
