import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random
from argparse import ArgumentParser
import pandas as pd

import cv2
import scipy.io as scio
import skvideo.io
import csv


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return True


def pyramidsGL(image, num_levels, dim=224):
    """ Creates Gaussian (G) and Laplacian (L) pyramids of level "num_levels" from image im.
    G and L are list where G[i], L[i] stores the i-th level of Gaussian and Laplacian pyramid, respectively. """
    o_width = image.shape[1]  # 960
    o_height = image.shape[0]  # 540
    if o_width > (dim + num_levels) and o_height > (dim + num_levels):
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width * f_height) / o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height * f_width) / o_width)
        else:
            f_width = f_height = dim

        height_step = int((o_height - f_height) / (num_levels - 1)) * (-1)
        width_step = int((o_width - f_width) / (num_levels - 1)) * (-1)
        height_list = [i for i in range(o_height, f_height - 1, height_step)]
        width_list = [i for i in range(o_width, f_width - 1, width_step)]

    elif o_width == dim or o_height == dim:
        height_list = [o_height for i in range(num_levels)]
        width_list = [o_width for i in range(num_levels)]

    else:
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width * f_height) / o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height * f_width) / o_width)
        else:
            f_width = f_height = dim
        image = cv2.resize(image, (f_width, f_height), interpolation=cv2.INTER_CUBIC)
        height_list = [f_height for i in range(num_levels)]
        width_list = [f_width for i in range(num_levels)]

    layer = image.copy()
    gaussian_pyramid = [layer]  # Gaussian Pyramid
    laplacian_pyramid = []  # Laplacian Pyramid

    for i in range(num_levels - 1):
        blur = cv2.GaussianBlur(gaussian_pyramid[i], (5, 5), 5)
        layer = cv2.resize(blur, (width_list[i + 1], height_list[i + 1]), interpolation=cv2.INTER_CUBIC)
        gaussian_pyramid.append(layer)

        uplayer = cv2.resize(blur, (width_list[i], height_list[i]), interpolation=cv2.INTER_CUBIC)
        laplacian = cv2.subtract(gaussian_pyramid[i], uplayer)
        laplacian_pyramid.append(laplacian)
    gaussian_pyramid.pop(-1)
    return gaussian_pyramid, laplacian_pyramid


def resizedpyramids(gaussian_pyramid, laplacian_pyramid, num_levels, width, height):
    gaussian_pyramid_resized, laplacian_pyramid_resized = [], []
    for i in range(num_levels - 1):
        img_gaussian_pyramid = cv2.resize(gaussian_pyramid[i], (width, height), interpolation=cv2.INTER_CUBIC)
        img_laplacian_pyramid = cv2.resize(laplacian_pyramid[i], (width, height), interpolation=cv2.INTER_CUBIC)
        gaussian_pyramid_resized.append(img_gaussian_pyramid)
        laplacian_pyramid_resized.append(img_laplacian_pyramid)
    return gaussian_pyramid_resized, laplacian_pyramid_resized


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, database_name, videos_dir, video_names, num_levels=6):
        super(VideoDataset, self).__init__()
        self.database_name = database_name
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.num_levels = num_levels

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        if self.database_name[:4] == 'LSVQ':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 8
        elif self.database_name[:12] == 'LIVEYTGaming':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 8
        elif self.database_name[:13] == 'LIVE-Qualcomm':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 15
        elif self.database_name[:8] == 'Waterloo':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 9
        elif self.database_name[:5] == 'BVISR':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 5

        elif self.database_name == 'CVD2014':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 10

        elif self.database_name == 'LiveVQC':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 10

        elif self.database_name == 'youtube_ugc':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 10

        elif self.database_name == 'LBVD':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            # video_length_read = 10
            video_length_read = 4

        elif self.database_name == 'KoNViD-1k':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 8

        elif self.database_name == 'BVIHFR':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 10

        elif self.database_name == 'LIVEHFR':
            video_name_str = self.video_names[idx]
            print(video_name_str)
            video_length_read = 6

        elif self.database_name == 'LIVE_livestreaming_rescale':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 7

        elif self.database_name == 'ETRI':
            video_name_str = self.video_names[idx][:-4]
            print(video_name_str)
            video_length_read = 5
        elif self.database_name == 'kvq_train' or self.database_name == 'kvq_val':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]
            print(video_name_str)
            video_length_read = 4
        elif self.database_name == 'xgc_train' or self.database_name == 'xgc_val' or self.database_name == 'xgc_test':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
            print(video_name_str)
            video_length_read = 8

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        imge_name = os.path.join(path_name, '{:03d}'.format(int(2)) + '.png')
        read_frame = Image.open(imge_name)
        # 获取图像的宽度和高度  5972x3360超现存，缩小到二分之一
        video_width, video_height = read_frame.size
        # 判断图像尺寸是否大于5972x3360，如果是，缩小到原始大小的一半
        if video_width > 5970 or video_height > 3360:
            read_frame = read_frame.resize((video_width // 2, video_height // 2))

        video_width = read_frame.size[0]
        video_height = read_frame.size[1]

        transformed_video = torch.zeros(
            [video_length_read * (self.num_levels - 1), video_channel, video_height,
             video_width])  # todo video_length_read * (self.num_levels - 1)?

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(i)) + '.png')
            read_frame = cv2.imread(imge_name)
            if self.database_name == 'BVISR_rescale':
                if len(video_name_str.split('_')) == 7:
                    reso = video_name_str.split('_')[5]
                    reso_w = int(reso.split('x')[0])
                    reso_h = int(reso.split('x')[1])
                    video_width = reso_w
                    video_height = reso_h
                    transformed_video = torch.zeros(
                        [video_length_read * (self.num_levels - 1), video_channel, video_height, video_width])
                    if video_name_str.split('_')[6] == 'NN':
                        read_frame = cv2.resize(read_frame, (reso_w, reso_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        read_frame = cv2.resize(read_frame, (reso_w, reso_h), interpolation=cv2.INTER_CUBIC)
            if self.database_name == 'LIVE_livestreaming_rescale':
                name_list = video_name_str.split('_')
                for name_str in name_list:
                    if name_str == '2160':
                        break
                    if name_str == '1080':
                        read_frame = cv2.resize(read_frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                        break
            gaussian_pyramid, laplacian_pyramid = pyramidsGL(read_frame, self.num_levels)
            _, laplacian_pyramid_resized = resizedpyramids(gaussian_pyramid, laplacian_pyramid, self.num_levels,
                                                           video_width, video_height)
            for j in range(len(laplacian_pyramid_resized)):
                lp = laplacian_pyramid_resized[j]
                lp = cv2.cvtColor(lp, cv2.COLOR_BGR2RGB)  #
                lp = transform(lp)
                transformed_video[i * (self.num_levels - 1) + j] = lp

        return transformed_video, video_name_str


class ResNet18_LP(torch.nn.Module):
    """Modified ResNet18 for feature extraction"""

    def __init__(self, layer=2):
        super(ResNet18_LP, self).__init__()
        if layer == 1:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-5])  # [1, 128, 56, 56]
        elif layer == 2:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-4])  # [1, 256, 28, 28]
        elif layer == 3:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-3])  # [1, 512, 14, 14]
        else:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])  # [1, 512, 7, 7]
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, layer=3, frame_batch_size=10, device='cuda'):
    """feature extraction"""
    extractor = ResNet18_LP(layer=layer).to(device)  #
    video_length = video_data.shape[0]  # 40
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            print(f'video_length: {video_length}, frame_end: {frame_end}')
            batch = video_data[frame_start:frame_end].to(device)

            # from fvcore.nn import FlopCountAnalysis
            # flop_analyzer = FlopCountAnalysis(extractor, batch)
            # flops = flop_analyzer.total()
            # # 检查是否低于 120G FLOPs
            # # print(f"Total FLOPs: {flops:.2e}")  # 以科学计数法输出
            # print('Total GFLOPs: %.2f ' % (flops / 1e9))

            features_mean, features_std = extractor(batch)  # (b,128,1,1)  (b,128,1,1)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)  # (40,128,1,1)
        output2 = torch.cat((output2, features_std), 0)  # (40,128,1,1)
        output = torch.cat((output1, output2), 1).squeeze()
        print("output_shape:", output.shape)

    return output
    # return output1.squeeze()


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Laplacian Pyramids Features using Pre-Trained ResNet-18')
    parser.add_argument("--seed", type=int, default=20231001)
    parser.add_argument('--database', default='xgc_test', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=10,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--layer', type=int, default=2,
                        help='RN18 layer for feature extraction (default: 2)')
    parser.add_argument('--num_levels', type=int, default=6,
                        help='number of gaussian pyramids')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'BVIHFR':
        imgs_dir = 'data/BVI-HFR_image_all_fps1'
        filename_path = 'data/BVI-HFR-MOS.csv'
        save_folder = 'data/BVIHFR_LP_ResNet18'

        dataInfo = pd.read_csv(filename_path)
        video_names = dataInfo['file_name']
        video_length_read = 10

    elif args.database == 'LIVE_livestreaming_rescale':
        imgs_dir = 'data/livestreaming_image_all_fps1'
        filename_path = 'data/LIVE_livestreaming_scores.csv'
        save_folder = 'data/livestreaming_rescale_LP_ResNet18'

        dataInfo = pd.read_csv(filename_path)
        video_names = dataInfo['video']
        video_length_read = 7

    elif args.database == 'ETRI':
        imgs_dir = 'data/ETRI_image_all_fps1'
        filename_path = 'data/ETRI_LIVE_MOS.csv'
        save_folder = 'data/ETRI_LP_ResNet18'

        dataInfo = pd.read_csv(filename_path)
        video_names = dataInfo['new_videoname']
        video_length_read = 5


    elif args.database == 'LIVEHFR':
        imgs_dir = 'data/LIVEHFR_image_all_fps1'
        filename_path = 'data/LIVEHFR_MOS.csv'
        save_folder = 'data/LIVEHFR_LP_ResNet18'

        dataInfo = pd.read_csv(filename_path)
        video_names = dataInfo['filename']
        video_length_read = 6


    elif args.database == 'Waterloo':
        imgs_dir = 'data/waterloo_fps1'
        filename_path = 'data/waterloo_data.csv'
        save_folder = 'data/waterloo_LP_ResNet18_2'

        dataInfo = pd.read_csv(filename_path)
        video_names = dataInfo['path']
        video_names_list = []
        for i in range(dataInfo.shape[0]):
            video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                         video_names[i].split('/')[-1]
            video_names_list.append(video_name)
        dataInfo['vpath'] = video_names_list
        video_names = dataInfo['vpath']
        video_length_read = 9


    elif args.database == 'BVISR_rescale':
        imgs_dir = 'data/BVI-SR_image_all_fps1'
        filename_path = 'data/BVI-SR_SUB.mat'
        save_folder = 'data/bvisr_rescale_LP_ResNet18_2'
        m_file = scio.loadmat(filename_path)
        video_names = []
        for i in range(len(m_file['MOS'])):
            video_names.append(m_file['seqName'][i][0][0])
        video_length_read = 5



    elif args.database == 'LSVQ_train':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_train.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8

    elif args.database == 'LSVQ_test':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8

    elif args.database == 'LSVQ_test_1080p':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_LP_ResNet18/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'

        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")
        video_names = dataInfo['name'].tolist()
        n_video = len(video_names)
        video_length_read = 8

    elif args.database[:12] == 'LIVEYTGaming':
        imgs_dir = 'data/liveytgaming_image_all_fps1'
        save_folder = 'data/liveytgaming_LP_ResNet18/'
        filename_path = 'data/LIVEYTGaming.mat'

        dataInfo = scio.loadmat(filename_path)
        n = len(dataInfo['video_list'])
        video_names = []
        index_all = dataInfo['index'][0]
        for i in index_all:
            video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
        video_length_read = 8

    elif args.database[:13] == 'LIVE-Qualcomm':
        imgs_dir = 'data/livequalcomm_image_all_fps1'
        save_folder = 'data/livequalcomm_LP_ResNet18/'
        filename_path = 'data/data/LIVE-Qualcomm_qualcommSubjectiveData.mat'

        m = scio.loadmat(filename_path)
        dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
        dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 15


    elif args.database == 'CVD2014':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_LP_ResNet18/'

        def read_float_with_comma(num):
            return float(num.replace(",", "."))

        file_names = []
        mos = []
        openfile = open("/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_Realignment_MOS.csv", 'r', newline='')
        lines = csv.DictReader(openfile, delimiter=';')
        for line in lines:
            if len(line['File_name']) > 0:
                file_names.append(line['File_name'])
            if len(line['Realignment MOS']) > 0:
                mos.append(read_float_with_comma(line['Realignment MOS']))

        dataInfo = pd.DataFrame(file_names)
        dataInfo['MOS'] = mos
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'] + ".avi"
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 10

    elif args.database == 'LiveVQC':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_image_all_fps1/'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_LP_ResNet18/'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'

        m = scio.loadmat(filename_path)
        dataInfo = pd.DataFrame(m['video_list'])
        dataInfo['MOS'] = m['mos']
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 10

    elif args.database == 'KoNViD-1k':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_LP_ResNet18'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/KoNViD-1k_data.mat'

        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0].split('_')[0])
        video_length_read = 8

    elif args.database == 'youtube_ugc':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_image_all_fps05'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/youtube_ugc_LP_ResNet18'
        filename_path = '/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx'

        # dataInfo = scio.loadmat(filename_path)
        # n_video = len(dataInfo['video_names'])
        # video_names = []
        #
        # for i in range(n_video):
        #     video_names.append(dataInfo['video_names'][i][0][0])
        # video_length_read = 10

        # 读取指定的工作表
        df = pd.read_excel(filename_path, sheet_name='diff2')
        # 读取指定列并将其转换为列表
        video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in df['vid'].astype(str).tolist()]
        n_video = len(video_names)
        video_length_read = 10


    elif args.database == 'LBVD':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_LP_ResNet18_clip'
        filename_path = '/data/user/zhaoyimeng/ModularBVQA/data/LBVD_data.mat'

        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        if '298.mp4' in video_names:
            video_names.remove('298.mp4')
            n_video = n_video - 1
        # video_length_read = 10
        video_length_read = 4

    elif args.database == 'kvq_train':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/train/kvq_LP_ResNet18'
        filename_path = '/data/dataset/KVQ/train_data.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)

        # 需要删除的文件列表
        exclude_files = [
            'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4',
            'train/0245.mp4',
            'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4',
            'train/0833.mp4',
            'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4',
            'train/0875.mp4',
            'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4',
            'train/0903.mp4',
            'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4',
            'train/0910.mp4',
            'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4',
            'train/1106.mp4',
            'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4',
            'train/1232.mp4',
            'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4',
            'train/1624.mp4',
            'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4',
            'train/1645.mp4',
            'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4',
            'train/1750.mp4',
            'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4',
            'train/1960.mp4',
            'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4',
            'train/2037.mp4',
            'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4',
            'train/2758.mp4',
            'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4',
            'train/2786.mp4'
        ]

        # 使用 DataFrame 过滤掉要删除的文件
        dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 4

    elif args.database == 'kvq_val':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_image_all_fps1_clip'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/val/kvq_LP_ResNet18'
        filename_path = '/data/dataset/KVQ/groundtruth_label 2/truth.csv'
        # 读取 CSV 文件，假设是逗号（,）分隔的
        dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
        # 确保列名正确
        dataInfo.columns = ['file_names', 'score']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names'].tolist()
        video_length_read = 4
    elif args.database == 'xgc_train':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_LP_ResNet18'
        filename_path = '/data/dataset/XGC-dataset/Share/cvprw-datasets/train/train.txt'
        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)
        # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
        dataInfo.columns = ['file_names', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'MOS6']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['MOS5'] = dataInfo['MOS5'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names']
        video_length_read = 8
    elif args.database == 'xgc_val':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_LP_ResNet18'
        filename_path = '/data/dataset/XGC-dataset/DATA/DATA3/yl/data/cvprw_dataset/dataset_final/val.txt'
        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)
        # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
        dataInfo.columns = ['file_names']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names']
        video_length_read = 8
    elif args.database == 'xgc_test':
        imgs_dir = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_image_all_fps1'
        save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/xgc_LP_ResNet18'
        filename_path = '/data/dataset/XGC-dataset/test.txt'
        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)
        # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
        dataInfo.columns = ['file_names']
        # 确保文件名是字符串
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        # 获取视频文件名列表
        video_names = dataInfo['file_names']
        video_length_read = 8


    device = torch.device("cuda:1" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    dataset = VideoDataset(args.database, imgs_dir, video_names, args.num_levels)

    for i in range(len(dataset)):
        print(i)
        current_data, video_name_str = dataset[i]
        features = get_features(current_data, args.layer, args.frame_batch_size, device)
        exit_folder(os.path.join(save_folder, video_name_str))
        for j in range(video_length_read):
            img_features = features[j * (args.num_levels - 1): (j + 1) * (args.num_levels - 1)]
            np.save(os.path.join(save_folder, video_name_str, '{:03d}'.format(j)), img_features.to('cpu').numpy())
