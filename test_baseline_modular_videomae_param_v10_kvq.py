import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_dist_aes_features, \
    VideoDataset_images_with_LP_motion_dist_aes_videomae_features, \
    kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_kvq':
        model = modular.kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(feat_len=8)
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_kvq':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))
    print('success load the trained model')

    ## training data
    if config.database == 'KVQ_test':
        videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_image_all_fps1'
        datainfo = '/data/dataset/KVQ/groundtruth_label 2/truth.csv'
        dist_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_dist_quality_aware'
        videomae_feat = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/KVQ/val/kvq_VideoMAE_feat'

    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(videos_dir, datainfo,
                                                                          transformations_test, 'KVQ_test',
                                                                          config.crop_size, dist_dir, videomae_feat)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()

        output_file = "output_kvq.csv"
        data_list = []

        label = np.zeros([len(testset)])
        y_output_stda = np.zeros([len(testset)])
        for i, (video, mos, dist, videomae, video_name_str) in tqdm(enumerate(test_loader)):
            video = video.to(device)
            dist = dist.to(device)
            videomae = videomae.to(device)
            label[i] = mos.item()
            outputs_stda = model(video, dist, videomae)
            y_output_stda[i] = outputs_stda.item()
            # print('val/' + str(video_name_str[0]) + '.mp4', 'groundtruth: ' + str(label[i]))
            print('val/' + str(video_name_str[0]) + '.mp4', str(y_output_stda[i]))

            # 添加数据到列表
            data_list.append(["val/{}.mp4".format(video_name_str[0]), y_output_stda[i]])

        # 创建 DataFrame 并保存为 CSV 文件
        df = pd.DataFrame(data_list, columns=["filename", "score"])
        df.to_csv(output_file, index=False)

        test_PLCC_stda, test_SRCC_stda, test_KRCC_stda, test_RMSE_stda = performance_fit(label, y_output_stda)

        print(config.database)
        print(
            'The result on the STDA test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                test_SRCC_stda, test_KRCC_stda, test_PLCC_stda, test_RMSE_stda))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='KVQ_test')  #
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_kvq')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='/data/user/zhaoyimeng/ModularBVQA/ckpts_kvq/kvq_2_plcc_rank_epoch_16_SRCC_0.960885.pth')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    config = parser.parse_args()

    main(config)
