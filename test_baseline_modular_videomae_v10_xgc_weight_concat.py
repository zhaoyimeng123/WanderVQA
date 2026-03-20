import argparse
import numpy as np
import time
import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features, \
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features, xgc_VideoDataset_images_with_dist_videomae_features, \
    xgc_weight_concat_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'Xgc_weight_concat':
        model = modular.Xgc_weight_concat(feat_len=8)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'Xgc_weight_concat':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))
    print('success load the trained model')


    videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
    datainfo = '/data/dataset/XGC-dataset/DATA/DATA3/yl/data/cvprw_dataset/dataset_final/val.txt'
    dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_dist_quality_aware'
    videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/xgc_VideoMAE_feat'
    aes_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_aes'
    slowfast_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_ugc_slowfast'
    lp_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_LP_ResNet18'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = xgc_weight_concat_features(videos_dir, datainfo, transformations_test, config.database,
                               config.crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir, "SlowFast", lp_dir)


    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        start_time = time.time()
        output_file = "ckpts_xgc_weight_concat/output.txt"
        with open(output_file, "w") as f:
            for i, (video, _, _, _, _, _, _, dist, videomae, video_name_str, transformed_aes, slowfast_feat, lp_feat) in enumerate(test_loader):
                video = video.to(device)
                dist = dist.to(device)
                transformed_aes = transformed_aes.to(device)
                slowfast_feat = slowfast_feat.to(device)
                lp_feat = lp_feat.to(device)
                outputs_stda = model(video, slowfast_feat, lp_feat, dist, transformed_aes)
                print('{}.mp4,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
                      .format(video_name_str[0],
                              outputs_stda[0, 0].item(), outputs_stda[0, 1].item(),
                              outputs_stda[0, 2].item(), outputs_stda[0, 3].item(),
                              outputs_stda[0, 4].item(), outputs_stda[0, 5].item()))

                # 格式化字符串，使用 "\t" 进行分割
                line = "{}.mp4,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    video_name_str[0],
                    outputs_stda[0, 0].item(), outputs_stda[0, 1].item(),
                    outputs_stda[0, 2].item(), outputs_stda[0, 3].item(),
                    outputs_stda[0, 4].item(), outputs_stda[0, 5].item()
                )
                f.write(line)


        end_time = time.time()
        print(config.database, end_time - start_time)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='xgc_test')  #
    parser.add_argument('--model_name', type=str,
                        default='Xgc_weight_concat')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_weight_concat/xgc_round_0_epoch_7_SRCC_0.844100.pth')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    config = parser.parse_args()

    main(config)
