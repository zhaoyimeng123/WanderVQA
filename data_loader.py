import os
import csv
import random

import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import math
from random import shuffle
from sklearn.model_selection import train_test_split


def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset_images_with_LP_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, lp_dir, filename_path, transform, database_name, crop_size, feature_type,
                 seed=0):
        super(VideoDataset_images_with_LP_motion_features, self).__init__()

        if database_name[0] == 'K':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'KoNViD-1k train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'KoNViD-1k val_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {train_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
            file_names = []
            mos = []
            openfile = open("/mnt/bn/wenwenwinwin-vqa/CVD2014/CVD2014_ratings/Realignment_MOS.csv", 'r', newline='')
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
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'youtube':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0])
                score.append(m['scores'][0][i])
            if database_name == 'youtube_ugc':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'youtube_ugc_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:13] == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LIVE-Qualcomm':
                self.video_names = video_names
                self.score = score
            else:
                random.seed(seed)
                np.random.seed(seed)
                length = dataInfo.shape[0]
                index_rd = np.random.permutation(length)
                train_full_index = index_rd[0:int(length * 0.6)]
                val_full_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_full_index = index_rd[int(length * 0.8):]

                if database_name == 'LIVE-Qualcomm_train':
                    self.video_names = dataInfo.iloc[train_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_val':
                    self.video_names = dataInfo.iloc[val_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_test':
                    self.video_names = dataInfo.iloc[test_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_full_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:12] == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_list'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
                score.append(dataInfo['MOS'][i][0])
            if database_name == 'LIVEYTGaming':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'LIVEYTGaming_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0] \
                        or '378.mp4' == dataInfo['video_names'][i][0][0] \
                        or '420.mp4' == dataInfo['video_names'][i][0][0] \
                        or '506.mp4' == dataInfo['video_names'][i][0][0] \
                        or '529.mp4' == dataInfo['video_names'][i][0][0] \
                        or '574.mp4' == dataInfo['video_names'][i][0][0] \
                        or '594.mp4' == dataInfo['video_names'][i][0][0] \
                        or '668.mp4' == dataInfo['video_names'][i][0][0] \
                        or '697.mp4' == dataInfo['video_names'][i][0][0] \
                        or '828.mp4' == dataInfo['video_names'][i][0][0] \
                        or '847.mp4' == dataInfo['video_names'][i][0][0] \
                        or '972.mp4' == dataInfo['video_names'][i][0][0] \
                        or '180.mp4' == dataInfo['video_names'][i][0][0] \
                        or '248.mp4' == dataInfo['video_names'][i][0][0] \
                        or '266.mp4' == dataInfo['video_names'][i][0][0] \
                        or '454.mp4' == dataInfo['video_names'][i][0][0] \
                        or '711.mp4' == dataInfo['video_names'][i][0][0] \
                        or '800.mp4' == dataInfo['video_names'][i][0][0] \
                        or '862.mp4' == dataInfo['video_names'][i][0][0] \
                        or '867.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            index_all = [x for x in index_all if x != 298]

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LBVD train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LBVD val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LBVD test_index: {test_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LBVD_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LBVD_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:13] == 'Livestreaming':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'Livestreaming':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:19] == 'Livestreaming_indep':
                    index_rd = np.random.permutation(45)
                    train_source = index_rd[:int(45 * 0.8)]
                    val_source = index_rd[int(45 * 0.8):]
                    test_source = index_rd[int(45 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 7, i * 7 + 7)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 7, i * 7 + 7)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 7, i * 7 + 7)]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['video'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['video'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['video'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:4] == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            dataInfo.columns = ['videoName', 'MOS', 'std', 'video']
            if database_name == 'ETRI':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:10] == 'ETRI_indep':
                    content_list = []
                    for name in dataInfo['video'].tolist():
                        content_list.append(name.split('_')[0])
                    dataInfo['content'] = content_list
                    content_set = list(set(content_list))
                    shuffle(content_set)

                    train_source = content_set[:12]
                    val_source = content_set[12:]
                    test_source = content_set[12:]

                    (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                    if database_name[-5:] == 'train':
                        self.video_names = (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(train_source)])['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = (dataInfo[dataInfo['content'].isin(val_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(val_source)])['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = (dataInfo[dataInfo['content'].isin(test_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(test_source)])['MOS'].tolist()


        elif database_name[:5] == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            if database_name == 'BVISR':
                self.video_names = video_names
                self.score = MOS
            else:
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': MOS})
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:11] == 'BVISR_indep':
                    index_rd = np.random.permutation(24)
                    train_source = index_rd[:int(24 * 0.6)]
                    val_source = index_rd[int(24 * 0.6):int(24 * 0.8)]
                    test_source = index_rd[int(24 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 10, i * 10 + 10)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 10, i * 10 + 10)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 10, i * 10 + 10)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:8] == 'Waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            MOS = dataInfo['MOS']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                             video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['file_names'] = video_names_list
            if database_name == 'Waterloo':
                self.video_names = video_names_list
                self.score = MOS
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:14] == 'Waterloo_indep':
                    index_rd = np.random.permutation(20) + 1
                    train_source = index_rd[:int(20 * 0.6)]
                    val_source = index_rd[int(20 * 0.6):int(20 * 0.8)]
                    test_source = index_rd[int(20 * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(train_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(train_source)]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(val_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(val_source)]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(test_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(test_source)]['MOS'].tolist()
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:6] == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'BVIHFR':
                self.video_names = dataInfo['file_name'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:12] == 'BVIHFR_indep':
                    index_rd = np.random.permutation(22)
                    train_source = index_rd[:int(22 * 0.6)]
                    val_source = index_rd[int(22 * 0.6):int(22 * 0.8)]
                    test_source = index_rd[int(22 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 4, i * 4 + 4)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 4, i * 4 + 4)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 4, i * 4 + 4)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:7] == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'LIVEHFR':
                self.video_names = dataInfo['filename'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:13] == 'LIVEHFR_indep':
                    index_rd = np.random.permutation(16)
                    train_source = index_rd[:int(16 * 0.6)]
                    val_source = index_rd[int(16 * 0.6):int(16 * 0.8)]
                    test_source = index_rd[int(16 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 30, i * 30 + 30)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 30, i * 30 + 30)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 30, i * 30 + 30)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['filename'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['filename'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['filename'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name == 'LSVQ_all':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_val':
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
            self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:10] == 'LSVQ_1080p':
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
            # self.video_names = dataInfo['name'].tolist()
            # self.score = dataInfo['mos'].tolist()
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if database_name == 'LSVQ_1080p_train':
                self.video_names = dataInfo.iloc[train_index]['name'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_val':
                self.video_names = dataInfo.iloc[val_index]['name'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_test':
                self.video_names = dataInfo.iloc[test_index]['name'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.lp_dir = lp_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[0] == 'K':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:7] == 'youtube' \
                or self.database_name[:7] == 'LiveVQC' \
                or self.database_name[:7] == 'CVD2014' \
                or self.database_name[:13] == 'LIVE-Qualcomm' \
                or self.database_name[:4] == 'LBVD' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                            :13] == 'Livestreaming':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name[:12] == 'LIVEYTGaming' or self.database_name[:8] == 'Waterloo' \
                or self.database_name[:6] == 'BVIHFR' or self.database_name[:7] == 'LIVEHFR':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        lp_name = os.path.join(self.lp_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        if self.database_name[:5] == 'BVISR':
            video_length_read = 5
        elif self.database_name[:8] == 'Waterloo':
            video_length_read = 9
        elif self.database_name[:6] == 'BVIHFR':
            video_length_read = 10
        elif self.database_name[:7] == 'LIVEHFR':
            video_length_read = 6
        elif self.database_name[:13] == 'Livestreaming':
            video_length_read = 7

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        resized_lp = torch.zeros([video_length_read, 5 * 256])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        if self.database_name[:13] == 'LIVE-Qualcomm':
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(2 * i)) + '.npy'))).view(-1)
        else:
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        if self.database_name[0] == 'K':
            video_name_str = video_name_str
        elif self.database_name[:7] == 'youtube' or self.database_name[:7] == 'LIVEHFR':
            video_name_str = video_name_str + '.mp4'
        elif self.database_name[:7] == 'CVD2014':
            video_name_str = video_name_str + '.avi'
        elif self.database_name[:13] == 'LIVE-Qualcomm' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                                :13] == 'Livestreaming':
            video_name_str = video_name_str + '.yuv'

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, resized_lp, video_name_str


class VideoDataset_images_with_LP_motion_dist_features(data.Dataset):

    def __init__(self, data_dir, data_dir_3D, lp_dir, filename_path, transform, database_name, crop_size, feature_type, dist_dir,
                 seed=0):
        super(VideoDataset_images_with_LP_motion_dist_features, self).__init__()

        if database_name[0] == 'K':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'KoNViD-1k train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'KoNViD-1k val_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {train_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
            file_names = []
            mos = []
            openfile = open("/mnt/bn/wenwenwinwin-vqa/CVD2014/CVD2014_ratings/Realignment_MOS.csv", 'r', newline='')
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
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'youtube':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0])
                score.append(m['scores'][0][i])
            if database_name == 'youtube_ugc':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'youtube_ugc_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:13] == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LIVE-Qualcomm':
                self.video_names = video_names
                self.score = score
            else:
                random.seed(seed)
                np.random.seed(seed)
                length = dataInfo.shape[0]
                index_rd = np.random.permutation(length)
                train_full_index = index_rd[0:int(length * 0.6)]
                val_full_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_full_index = index_rd[int(length * 0.8):]

                if database_name == 'LIVE-Qualcomm_train':
                    self.video_names = dataInfo.iloc[train_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_val':
                    self.video_names = dataInfo.iloc[val_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_test':
                    self.video_names = dataInfo.iloc[test_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_full_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                '''
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()
                '''
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LiveVQC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:12] == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_list'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
                score.append(dataInfo['MOS'][i][0])
            if database_name == 'LIVEYTGaming':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'LIVEYTGaming_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            index_all = [x for x in index_all if x != 298]

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LBVD train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LBVD val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LBVD test_index: {test_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LBVD_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LBVD_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:13] == 'Livestreaming':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'Livestreaming':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:19] == 'Livestreaming_indep':
                    index_rd = np.random.permutation(45)
                    train_source = index_rd[:int(45 * 0.8)]
                    val_source = index_rd[int(45 * 0.8):]
                    test_source = index_rd[int(45 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 7, i * 7 + 7)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 7, i * 7 + 7)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 7, i * 7 + 7)]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['video'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['video'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['video'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:4] == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            dataInfo.columns = ['videoName', 'MOS', 'std', 'video']
            if database_name == 'ETRI':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:10] == 'ETRI_indep':
                    content_list = []
                    for name in dataInfo['video'].tolist():
                        content_list.append(name.split('_')[0])
                    dataInfo['content'] = content_list
                    content_set = list(set(content_list))
                    shuffle(content_set)

                    train_source = content_set[:12]
                    val_source = content_set[12:]
                    test_source = content_set[12:]

                    (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                    if database_name[-5:] == 'train':
                        self.video_names = (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(train_source)])['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = (dataInfo[dataInfo['content'].isin(val_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(val_source)])['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = (dataInfo[dataInfo['content'].isin(test_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(test_source)])['MOS'].tolist()


        elif database_name[:5] == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            if database_name == 'BVISR':
                self.video_names = video_names
                self.score = MOS
            else:
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': MOS})
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:11] == 'BVISR_indep':
                    index_rd = np.random.permutation(24)
                    train_source = index_rd[:int(24 * 0.6)]
                    val_source = index_rd[int(24 * 0.6):int(24 * 0.8)]
                    test_source = index_rd[int(24 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 10, i * 10 + 10)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 10, i * 10 + 10)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 10, i * 10 + 10)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:8] == 'Waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            MOS = dataInfo['MOS']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                             video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['file_names'] = video_names_list
            if database_name == 'Waterloo':
                self.video_names = video_names_list
                self.score = MOS
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:14] == 'Waterloo_indep':
                    index_rd = np.random.permutation(20) + 1
                    train_source = index_rd[:int(20 * 0.6)]
                    val_source = index_rd[int(20 * 0.6):int(20 * 0.8)]
                    test_source = index_rd[int(20 * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(train_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(train_source)]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(val_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(val_source)]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(test_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(test_source)]['MOS'].tolist()
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:6] == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'BVIHFR':
                self.video_names = dataInfo['file_name'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:12] == 'BVIHFR_indep':
                    index_rd = np.random.permutation(22)
                    train_source = index_rd[:int(22 * 0.6)]
                    val_source = index_rd[int(22 * 0.6):int(22 * 0.8)]
                    test_source = index_rd[int(22 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 4, i * 4 + 4)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 4, i * 4 + 4)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 4, i * 4 + 4)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:7] == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'LIVEHFR':
                self.video_names = dataInfo['filename'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:13] == 'LIVEHFR_indep':
                    index_rd = np.random.permutation(16)
                    train_source = index_rd[:int(16 * 0.6)]
                    val_source = index_rd[int(16 * 0.6):int(16 * 0.8)]
                    test_source = index_rd[int(16 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 30, i * 30 + 30)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 30, i * 30 + 30)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 30, i * 30 + 30)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['filename'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['filename'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['filename'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name == 'LSVQ_all':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_val':
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
            self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:10] == 'LSVQ_1080p':
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
            # self.video_names = dataInfo['name'].tolist()
            # self.score = dataInfo['mos'].tolist()
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if database_name == 'LSVQ_1080p_train':
                self.video_names = dataInfo.iloc[train_index]['name'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_val':
                self.video_names = dataInfo.iloc[val_index]['name'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_test':
                self.video_names = dataInfo.iloc[test_index]['name'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.lp_dir = lp_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.dist_dir = dist_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[0] == 'K':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:7] == 'youtube' \
                or self.database_name[:7] == 'LiveVQC' \
                or self.database_name[:7] == 'CVD2014' \
                or self.database_name[:13] == 'LIVE-Qualcomm' \
                or self.database_name[:4] == 'LBVD' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                            :13] == 'Livestreaming':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name[:12] == 'LIVEYTGaming' or self.database_name[:8] == 'Waterloo' \
                or self.database_name[:6] == 'BVIHFR' or self.database_name[:7] == 'LIVEHFR':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        lp_name = os.path.join(self.lp_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        if self.database_name[:5] == 'BVISR':
            video_length_read = 5
        elif self.database_name[:8] == 'Waterloo':
            video_length_read = 9
        elif self.database_name[:6] == 'BVIHFR':
            video_length_read = 10
        elif self.database_name[:7] == 'LIVEHFR':
            video_length_read = 6
        elif self.database_name[:13] == 'Livestreaming':
            video_length_read = 7

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        resized_lp = torch.zeros([video_length_read, 5 * 256])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        if self.database_name[:13] == 'LIVE-Qualcomm':
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(2 * i)) + '.npy'))).view(-1)
        else:
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        if self.database_name[0] == 'K':
            video_name_str = video_name_str
        elif self.database_name[:7] == 'youtube' or self.database_name[:7] == 'LIVEHFR':
            video_name_str = video_name_str + '.mp4'
        elif self.database_name[:7] == 'CVD2014':
            video_name_str = video_name_str + '.avi'
        elif self.database_name[:13] == 'LIVE-Qualcomm' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                                :13] == 'Livestreaming':
            video_name_str = video_name_str + '.yuv'

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        # 读取dist
        dist_forder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                i_index = int(2 * i)
            dist = np.load(os.path.join(dist_forder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist


        return transformed_video, transformed_feature, video_score, resized_lp, transformed_dist, video_name_str


class VideoDataset_images_with_LP_motion_dist_aes_features(data.Dataset):

    def __init__(self, data_dir, data_dir_3D, lp_dir, filename_path, transform, database_name, crop_size, feature_type, dist_dir, aes_dir,
                 seed=0):
        super(VideoDataset_images_with_LP_motion_dist_aes_features, self).__init__()

        if database_name[0] == 'K':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'KoNViD-1k train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'KoNViD-1k val_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {train_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
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
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'youtube':
            # m = scio.loadmat(filename_path)
            # n = len(m['video_names'])
            video_names = []
            score = []
            # index_all = m['index'][0]
            # for i in index_all:
            #     video_names.append(m['video_names'][i][0][0])
            #     score.append(m['scores'][0][i])
            if database_name == 'youtube_ugc':
                self.video_names = video_names
                self.score = score
            else:

                # dataInfo = pd.DataFrame(video_names)
                # dataInfo['score'] = score
                # dataInfo.columns = ['file_names', 'MOS']
                # length = dataInfo.shape[0]

                # 读取指定的工作表
                df = pd.read_excel('/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx', sheet_name='diff2')
                # 读取文件名和MOS分数，并将它们分别存储
                video_names = df['vid'].astype(str).tolist()  # 将 'vid' 列转换为字符串列表
                scores = df['MOS full'].tolist()  # 获取 'MOS full' 列的得分
                # 添加后缀以保持一致性
                video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in video_names]
                # 创建DataFrame并将列命名为 file_names 和 MOS 以与上面的保持一致
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': scores})
                length = dataInfo.shape[0]

                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                # train_index = index_rd[0:int(length * 0.6)]
                # val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                # test_index = index_rd[int(length * 0.8):]
                train_index = index_rd[0:int(length * 0.8)]
                print(f'YouTube_UGC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'YouTube_UGC test_index: {test_index}')
                if database_name == 'youtube_ugc_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                # elif database_name == 'youtube_ugc_val':
                #     self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                #     self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:13] == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LIVE-Qualcomm':
                self.video_names = video_names
                self.score = score
            else:
                random.seed(seed)
                np.random.seed(seed)
                length = dataInfo.shape[0]
                index_rd = np.random.permutation(length)
                train_full_index = index_rd[0:int(length * 0.6)]
                val_full_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_full_index = index_rd[int(length * 0.8):]

                if database_name == 'LIVE-Qualcomm_train':
                    self.video_names = dataInfo.iloc[train_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_val':
                    self.video_names = dataInfo.iloc[val_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_test':
                    self.video_names = dataInfo.iloc[test_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_full_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                '''
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()
                '''
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LiveVQC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:12] == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_list'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
                score.append(dataInfo['MOS'][i][0])
            if database_name == 'LIVEYTGaming':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'LIVEYTGaming_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            index_all = [x for x in index_all if x != 298]

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LBVD train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LBVD val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LBVD test_index: {test_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LBVD_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LBVD_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:13] == 'Livestreaming':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'Livestreaming':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:19] == 'Livestreaming_indep':
                    index_rd = np.random.permutation(45)
                    train_source = index_rd[:int(45 * 0.8)]
                    val_source = index_rd[int(45 * 0.8):]
                    test_source = index_rd[int(45 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 7, i * 7 + 7)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 7, i * 7 + 7)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 7, i * 7 + 7)]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['video'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['video'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['video'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:4] == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            dataInfo.columns = ['videoName', 'MOS', 'std', 'video']
            if database_name == 'ETRI':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:10] == 'ETRI_indep':
                    content_list = []
                    for name in dataInfo['video'].tolist():
                        content_list.append(name.split('_')[0])
                    dataInfo['content'] = content_list
                    content_set = list(set(content_list))
                    shuffle(content_set)

                    train_source = content_set[:12]
                    val_source = content_set[12:]
                    test_source = content_set[12:]

                    (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                    if database_name[-5:] == 'train':
                        self.video_names = (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(train_source)])['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = (dataInfo[dataInfo['content'].isin(val_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(val_source)])['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = (dataInfo[dataInfo['content'].isin(test_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(test_source)])['MOS'].tolist()


        elif database_name[:5] == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            if database_name == 'BVISR':
                self.video_names = video_names
                self.score = MOS
            else:
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': MOS})
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:11] == 'BVISR_indep':
                    index_rd = np.random.permutation(24)
                    train_source = index_rd[:int(24 * 0.6)]
                    val_source = index_rd[int(24 * 0.6):int(24 * 0.8)]
                    test_source = index_rd[int(24 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 10, i * 10 + 10)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 10, i * 10 + 10)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 10, i * 10 + 10)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:8] == 'Waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            MOS = dataInfo['MOS']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                             video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['file_names'] = video_names_list
            if database_name == 'Waterloo':
                self.video_names = video_names_list
                self.score = MOS
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:14] == 'Waterloo_indep':
                    index_rd = np.random.permutation(20) + 1
                    train_source = index_rd[:int(20 * 0.6)]
                    val_source = index_rd[int(20 * 0.6):int(20 * 0.8)]
                    test_source = index_rd[int(20 * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(train_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(train_source)]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(val_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(val_source)]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(test_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(test_source)]['MOS'].tolist()
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:6] == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'BVIHFR':
                self.video_names = dataInfo['file_name'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:12] == 'BVIHFR_indep':
                    index_rd = np.random.permutation(22)
                    train_source = index_rd[:int(22 * 0.6)]
                    val_source = index_rd[int(22 * 0.6):int(22 * 0.8)]
                    test_source = index_rd[int(22 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 4, i * 4 + 4)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 4, i * 4 + 4)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 4, i * 4 + 4)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:7] == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'LIVEHFR':
                self.video_names = dataInfo['filename'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:13] == 'LIVEHFR_indep':
                    index_rd = np.random.permutation(16)
                    train_source = index_rd[:int(16 * 0.6)]
                    val_source = index_rd[int(16 * 0.6):int(16 * 0.8)]
                    test_source = index_rd[int(16 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 30, i * 30 + 30)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 30, i * 30 + 30)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 30, i * 30 + 30)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['filename'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['filename'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['filename'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name == 'LSVQ_all':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_val':
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
            self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:10] == 'LSVQ_1080p':
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
            # self.video_names = dataInfo['name'].tolist()
            # self.score = dataInfo['mos'].tolist()
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if database_name == 'LSVQ_1080p_train':
                self.video_names = dataInfo.iloc[train_index]['name'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_val':
                self.video_names = dataInfo.iloc[val_index]['name'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_test':
                self.video_names = dataInfo.iloc[test_index]['name'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.lp_dir = lp_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.aes_dir = aes_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[0] == 'K':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:7] == 'youtube' \
                or self.database_name[:7] == 'LiveVQC' \
                or self.database_name[:7] == 'CVD2014' \
                or self.database_name[:13] == 'LIVE-Qualcomm' \
                or self.database_name[:4] == 'LBVD' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                            :13] == 'Livestreaming':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name[:12] == 'LIVEYTGaming' or self.database_name[:8] == 'Waterloo' \
                or self.database_name[:6] == 'BVIHFR' or self.database_name[:7] == 'LIVEHFR':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        lp_name = os.path.join(self.lp_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        if self.database_name[:5] == 'BVISR':
            video_length_read = 5
        elif self.database_name[:8] == 'Waterloo':
            video_length_read = 9
        elif self.database_name[:6] == 'BVIHFR':
            video_length_read = 10
        elif self.database_name[:7] == 'LIVEHFR':
            video_length_read = 6
        elif self.database_name[:13] == 'Livestreaming':
            video_length_read = 7

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        resized_lp = torch.zeros([video_length_read, 5 * 256])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        if self.database_name[:13] == 'LIVE-Qualcomm':
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(2 * i)) + '.npy'))).view(-1)
        else:
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        if self.database_name[0] == 'K' or self.database_name[:7] == 'youtube':
            video_name_str = video_name_str
        elif self.database_name[:7] == 'LIVEHFR':
            video_name_str = video_name_str + '.mp4'
        # elif self.database_name[:7] == 'CVD2014':
        #     video_name_str = video_name_str + '.avi'
        elif self.database_name[:13] == 'LIVE-Qualcomm' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                                :13] == 'Livestreaming':
            video_name_str = video_name_str + '.yuv'

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            # if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
            if self.database_name[:13] == 'LIVE-Qualcomm':
                i_index = int(2 * i)
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取aes
        aes_folder_name = os.path.join(self.aes_dir, video_name_str)
        transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
        aes = np.load(aes_folder_name + '.npy')
        transformed_aes = torch.from_numpy(aes)

        return transformed_video, transformed_feature, video_score, resized_lp, transformed_dist, transformed_aes, video_name_str


class VideoDataset_images_with_LP_motion_dist_aes_videomae_features(data.Dataset):

    def __init__(self, data_dir, data_dir_3D, lp_dir, filename_path, transform, database_name, crop_size, feature_type, dist_dir, aes_dir, videomae_feat,
                 seed=0):
        super(VideoDataset_images_with_LP_motion_dist_aes_videomae_features, self).__init__()

        if database_name[:6] == 'KoNViD':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.8)]
                print(f'KoNViD-1k train_index: {train_index}')
                # val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                # print(f'KoNViD-1k val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {test_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                # elif database_name == 'KoNViD-1k_val':
                #     self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                #     self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
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
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:3] == 'KVQ':
            # 读取 CSV 文件，假设是逗号（,）分隔的
            dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
            # 确保列名正确
            dataInfo.columns = ['file_names', 'score']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 需要删除的文件列表
            exclude_files = [
                'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4', 'train/0245.mp4',
                'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4', 'train/0833.mp4',
                'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4', 'train/0875.mp4',
                'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4', 'train/0903.mp4',
                'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4', 'train/0910.mp4',
                'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4', 'train/1106.mp4',
                'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4', 'train/1232.mp4',
                'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4', 'train/1624.mp4',
                'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4', 'train/1645.mp4',
                'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4', 'train/1750.mp4',
                'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4', 'train/1960.mp4',
                'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4', 'train/2037.mp4',
                'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4', 'train/2758.mp4',
                'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4', 'train/2786.mp4'
            ]

            # 使用 DataFrame 过滤掉要删除的文件
            dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

            if database_name == 'KVQ':
                self.video_names = dataInfo['file_names'].tolist()
                self.score = dataInfo['score'].tolist()
            else:
                length = len(dataInfo)
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)

                train_index = index_rd[:int(length * 0.9)]
                val_index = index_rd[int(length * 0.9):]

                if database_name == 'KVQ_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['score'].tolist()  # 修正为 `score`
                elif database_name == 'KVQ_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['score'].tolist()


        elif database_name[:7] == 'youtube':
            # m = scio.loadmat(filename_path)
            # n = len(m['video_names'])
            video_names = []
            score = []
            # index_all = m['index'][0]
            # for i in index_all:
            #     video_names.append(m['video_names'][i][0][0])
            #     score.append(m['scores'][0][i])
            if database_name == 'youtube_ugc':
                # 读取指定的工作表
                df = pd.read_excel('/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx',
                                   sheet_name='diff2')
                # 读取文件名和MOS分数，并将它们分别存储
                video_names = df['vid'].astype(str).tolist()  # 将 'vid' 列转换为字符串列表
                scores = df['MOS full'].tolist()  # 获取 'MOS full' 列的得分
                # 添加后缀以保持一致性
                video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in video_names]
                # 创建DataFrame并将列命名为 file_names 和 MOS 以与上面的保持一致
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': scores})
                self.video_names = dataInfo.iloc[:]['file_names'].tolist()
                self.score = dataInfo.iloc[:]['MOS'].tolist()

            else:

                # dataInfo = pd.DataFrame(video_names)
                # dataInfo['score'] = score
                # dataInfo.columns = ['file_names', 'MOS']
                # length = dataInfo.shape[0]

                # 读取指定的工作表
                df = pd.read_excel('/data/dataset/YoutubeYGC/original_videos_MOS_for_YouTube_UGC_dataset.xlsx', sheet_name='diff2')
                # 读取文件名和MOS分数，并将它们分别存储
                video_names = df['vid'].astype(str).tolist()  # 将 'vid' 列转换为字符串列表
                scores = df['MOS full'].tolist()  # 获取 'MOS full' 列的得分
                # 添加后缀以保持一致性
                video_names = [f"{filename}_crf_10_ss_00_t_20.0.mp4" for filename in video_names]
                # 创建DataFrame并将列命名为 file_names 和 MOS 以与上面的保持一致
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': scores})
                length = dataInfo.shape[0]

                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                # train_index = index_rd[0:int(length * 0.6)]
                # val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                # test_index = index_rd[int(length * 0.8):]
                train_index = index_rd[0:int(length * 0.8)]
                print(f'YouTube_UGC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'YouTube_UGC test_index: {test_index}')
                if database_name == 'youtube_ugc_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                # elif database_name == 'youtube_ugc_val':
                #     self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                #     self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:13] == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LIVE-Qualcomm':
                self.video_names = video_names
                self.score = score
            else:
                random.seed(seed)
                np.random.seed(seed)
                length = dataInfo.shape[0]
                index_rd = np.random.permutation(length)
                train_full_index = index_rd[0:int(length * 0.6)]
                val_full_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_full_index = index_rd[int(length * 0.8):]

                if database_name == 'LIVE-Qualcomm_train':
                    self.video_names = dataInfo.iloc[train_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_val':
                    self.video_names = dataInfo.iloc[val_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_test':
                    self.video_names = dataInfo.iloc[test_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_full_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                '''
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()
                '''
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LiveVQC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:12] == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_list'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
                score.append(dataInfo['MOS'][i][0])
            if database_name == 'LIVEYTGaming':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'LIVEYTGaming_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0] or '378.mp4' == dataInfo['video_names'][i][0][0] or '420.mp4' == dataInfo['video_names'][i][0][0] or '506.mp4' == dataInfo['video_names'][i][0][0] or '529.mp4' == dataInfo['video_names'][i][0][0] or '574.mp4' == dataInfo['video_names'][i][0][0] or '594.mp4' == dataInfo['video_names'][i][0][0] or '668.mp4' == dataInfo['video_names'][i][0][0] or '697.mp4' == dataInfo['video_names'][i][0][0] or '828.mp4' == dataInfo['video_names'][i][0][0] or '847.mp4' == dataInfo['video_names'][i][0][0] or '972.mp4' == dataInfo['video_names'][i][0][0] or '180.mp4' == dataInfo['video_names'][i][0][0] or '248.mp4' == dataInfo['video_names'][i][0][0] or '266.mp4' == dataInfo['video_names'][i][0][0] or '454.mp4' == dataInfo['video_names'][i][0][0] or '711.mp4' == dataInfo['video_names'][i][0][0] or '800.mp4' == dataInfo['video_names'][i][0][0] or '862.mp4' == dataInfo['video_names'][i][0][0] or '867.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            index_all = [x for x in index_all if x != 298]

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LBVD train_index: {train_index}')
                # val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                # print(f'LBVD val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LBVD test_index: {test_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                # elif database_name == 'LBVD_val':
                #     self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                #     self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LBVD_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:13] == 'Livestreaming':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'Livestreaming':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:19] == 'Livestreaming_indep':
                    index_rd = np.random.permutation(45)
                    train_source = index_rd[:int(45 * 0.8)]
                    val_source = index_rd[int(45 * 0.8):]
                    test_source = index_rd[int(45 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 7, i * 7 + 7)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 7, i * 7 + 7)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 7, i * 7 + 7)]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['video'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['video'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['video'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:4] == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            dataInfo.columns = ['videoName', 'MOS', 'std', 'video']
            if database_name == 'ETRI':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:10] == 'ETRI_indep':
                    content_list = []
                    for name in dataInfo['video'].tolist():
                        content_list.append(name.split('_')[0])
                    dataInfo['content'] = content_list
                    content_set = list(set(content_list))
                    shuffle(content_set)

                    train_source = content_set[:12]
                    val_source = content_set[12:]
                    test_source = content_set[12:]

                    (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                    if database_name[-5:] == 'train':
                        self.video_names = (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(train_source)])['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = (dataInfo[dataInfo['content'].isin(val_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(val_source)])['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = (dataInfo[dataInfo['content'].isin(test_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(test_source)])['MOS'].tolist()


        elif database_name[:5] == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            if database_name == 'BVISR':
                self.video_names = video_names
                self.score = MOS
            else:
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': MOS})
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:11] == 'BVISR_indep':
                    index_rd = np.random.permutation(24)
                    train_source = index_rd[:int(24 * 0.6)]
                    val_source = index_rd[int(24 * 0.6):int(24 * 0.8)]
                    test_source = index_rd[int(24 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 10, i * 10 + 10)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 10, i * 10 + 10)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 10, i * 10 + 10)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:8] == 'Waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            MOS = dataInfo['MOS']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                             video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['file_names'] = video_names_list
            if database_name == 'Waterloo':
                self.video_names = video_names_list
                self.score = MOS
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:14] == 'Waterloo_indep':
                    index_rd = np.random.permutation(20) + 1
                    train_source = index_rd[:int(20 * 0.6)]
                    val_source = index_rd[int(20 * 0.6):int(20 * 0.8)]
                    test_source = index_rd[int(20 * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(train_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(train_source)]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(val_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(val_source)]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(test_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(test_source)]['MOS'].tolist()
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:6] == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'BVIHFR':
                self.video_names = dataInfo['file_name'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:12] == 'BVIHFR_indep':
                    index_rd = np.random.permutation(22)
                    train_source = index_rd[:int(22 * 0.6)]
                    val_source = index_rd[int(22 * 0.6):int(22 * 0.8)]
                    test_source = index_rd[int(22 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 4, i * 4 + 4)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 4, i * 4 + 4)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 4, i * 4 + 4)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:7] == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'LIVEHFR':
                self.video_names = dataInfo['filename'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:13] == 'LIVEHFR_indep':
                    index_rd = np.random.permutation(16)
                    train_source = index_rd[:int(16 * 0.6)]
                    val_source = index_rd[int(16 * 0.6):int(16 * 0.8)]
                    test_source = index_rd[int(16 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 30, i * 30 + 30)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 30, i * 30 + 30)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 30, i * 30 + 30)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['filename'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['filename'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['filename'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name == 'LSVQ_all':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_val':
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
            self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:10] == 'LSVQ_1080p':
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
            # self.video_names = dataInfo['name'].tolist()
            # self.score = dataInfo['mos'].tolist()
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if database_name == 'LSVQ_1080p_train':
                self.video_names = dataInfo.iloc[train_index]['name'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_val':
                self.video_names = dataInfo.iloc[val_index]['name'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_test':
                self.video_names = dataInfo.iloc[test_index]['name'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.lp_dir = lp_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.aes_dir = aes_dir
        self.videomae_dir = videomae_feat

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:6] == 'KoNViD':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:7] == 'youtube' \
                or self.database_name[:7] == 'LiveVQC' \
                or self.database_name[:7] == 'CVD2014' \
                or self.database_name[:13] == 'LIVE-Qualcomm' \
                or self.database_name[:4] == 'LBVD' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                            :13] == 'Livestreaming':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name[:12] == 'LIVEYTGaming' or self.database_name[:8] == 'Waterloo' \
                or self.database_name[:6] == 'BVIHFR' or self.database_name[:7] == 'LIVEHFR':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]
        elif self.database_name[:3] == 'KVQ':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)


        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        if self.database_name[:5] == 'BVISR':
            video_length_read = 5
        elif self.database_name[:8] == 'Waterloo':
            video_length_read = 9
        elif self.database_name[:6] == 'BVIHFR':
            video_length_read = 10
        elif self.database_name[:7] == 'LIVEHFR':
            video_length_read = 6
        elif self.database_name[:13] == 'Livestreaming':
            video_length_read = 7

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        resized_lp = torch.zeros([video_length_read, 5 * 256])
        if self.lp_dir is not None:
            lp_name = os.path.join(self.lp_dir, video_name_str)
        if self.database_name[:13] == 'LIVE-Qualcomm':
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                if self.lp_dir is not None:
                    resized_lp[i] = torch.from_numpy(
                        np.load(os.path.join(lp_name, '{:03d}'.format(int(2 * i)) + '.npy'))).view(-1)
        else:
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                if self.lp_dir is not None:
                    resized_lp[i] = torch.from_numpy(
                        np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        if self.database_name[:6] == 'KoNViD' or self.database_name[:7] == 'youtube':
            video_name_str = video_name_str
        elif self.database_name[:7] == 'LIVEHFR':
            video_name_str = video_name_str + '.mp4'
        # elif self.database_name[:7] == 'CVD2014':
        #     video_name_str = video_name_str + '.avi'
        elif self.database_name[:13] == 'LIVE-Qualcomm' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                                :13] == 'Livestreaming':
            video_name_str = video_name_str + '.yuv'

        # read 3D features
        transformed_feature = None
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            # if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
            if self.database_name[:13] == 'LIVE-Qualcomm':
                i_index = int(2 * i)
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取aes
        transformed_aes = None
        if self.aes_dir is not None:
            aes_folder_name = os.path.join(self.aes_dir, video_name_str)
            transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
            aes = np.load(aes_folder_name + '.npy')
            transformed_aes = torch.from_numpy(aes)

        # 读取videomae_feat
        videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
        videomae_feat = np.load(videomae_folder_name + '.npy')
        transformed_videomae = torch.from_numpy(videomae_feat)


        return transformed_video, transformed_feature, video_score, resized_lp, transformed_dist, transformed_aes, transformed_videomae, video_name_str


class VideoDataset_images_with_QCN_dist_features(data.Dataset):

    def __init__(self, data_dir, data_dir_3D, QCN_dir, filename_path, transform, database_name, crop_size, feature_type, dist_dir,
                 seed=0):
        super(VideoDataset_images_with_QCN_dist_features, self).__init__()

        if database_name[0] == 'K':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'KoNViD-1k train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'KoNViD-1k val_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {train_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
            file_names = []
            mos = []
            openfile = open("/mnt/bn/wenwenwinwin-vqa/CVD2014/CVD2014_ratings/Realignment_MOS.csv", 'r', newline='')
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
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'youtube':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0])
                score.append(m['scores'][0][i])
            if database_name == 'youtube_ugc':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'youtube_ugc_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'youtube_ugc_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:13] == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LIVE-Qualcomm':
                self.video_names = video_names
                self.score = score
            else:
                random.seed(seed)
                np.random.seed(seed)
                length = dataInfo.shape[0]
                index_rd = np.random.permutation(length)
                train_full_index = index_rd[0:int(length * 0.6)]
                val_full_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_full_index = index_rd[int(length * 0.8):]

                if database_name == 'LIVE-Qualcomm_train':
                    self.video_names = dataInfo.iloc[train_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_val':
                    self.video_names = dataInfo.iloc[val_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_full_index]['MOS'].tolist()
                elif database_name == 'LIVE-Qualcomm_test':
                    self.video_names = dataInfo.iloc[test_full_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_full_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                '''
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()
                '''
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LiveVQC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:12] == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_list'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_list'][i][0][0] + '.mp4')
                score.append(dataInfo['MOS'][i][0])
            if database_name == 'LIVEYTGaming':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'LIVEYTGaming_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LIVEYTGaming_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            index_all = [x for x in index_all if x != 298]

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LBVD train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LBVD val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LBVD test_index: {test_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LBVD_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LBVD_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:13] == 'Livestreaming':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'Livestreaming':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:19] == 'Livestreaming_indep':
                    index_rd = np.random.permutation(45)
                    train_source = index_rd[:int(45 * 0.8)]
                    val_source = index_rd[int(45 * 0.8):]
                    test_source = index_rd[int(45 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 7, i * 7 + 7)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 7, i * 7 + 7)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 7, i * 7 + 7)]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['video'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['video'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['video'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:4] == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            dataInfo.columns = ['videoName', 'MOS', 'std', 'video']
            if database_name == 'ETRI':
                self.video_names = dataInfo['video'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:10] == 'ETRI_indep':
                    content_list = []
                    for name in dataInfo['video'].tolist():
                        content_list.append(name.split('_')[0])
                    dataInfo['content'] = content_list
                    content_set = list(set(content_list))
                    shuffle(content_set)

                    train_source = content_set[:12]
                    val_source = content_set[12:]
                    test_source = content_set[12:]

                    (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                    if database_name[-5:] == 'train':
                        self.video_names = (dataInfo[dataInfo['content'].isin(train_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(train_source)])['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = (dataInfo[dataInfo['content'].isin(val_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(val_source)])['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = (dataInfo[dataInfo['content'].isin(test_source)])['video'].tolist()
                        self.score = (dataInfo[dataInfo['content'].isin(test_source)])['MOS'].tolist()


        elif database_name[:5] == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            if database_name == 'BVISR':
                self.video_names = video_names
                self.score = MOS
            else:
                dataInfo = pd.DataFrame({'file_names': video_names, 'MOS': MOS})
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:11] == 'BVISR_indep':
                    index_rd = np.random.permutation(24)
                    train_source = index_rd[:int(24 * 0.6)]
                    val_source = index_rd[int(24 * 0.6):int(24 * 0.8)]
                    test_source = index_rd[int(24 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 10, i * 10 + 10)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 10, i * 10 + 10)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 10, i * 10 + 10)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:8] == 'Waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            MOS = dataInfo['MOS']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3] + '_' + video_names[i].split('/')[-2] + '_' + \
                             video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['file_names'] = video_names_list
            if database_name == 'Waterloo':
                self.video_names = video_names_list
                self.score = MOS
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)

                if database_name[:14] == 'Waterloo_indep':
                    index_rd = np.random.permutation(20) + 1
                    train_source = index_rd[:int(20 * 0.6)]
                    val_source = index_rd[int(20 * 0.6):int(20 * 0.8)]
                    test_source = index_rd[int(20 * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(train_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(train_source)]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(val_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(val_source)]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo[dataInfo['Video Number'].isin(test_source)]['file_names'].tolist()
                        self.score = dataInfo[dataInfo['Video Number'].isin(test_source)]['MOS'].tolist()
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]
                    if database_name[-5:] == 'train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name[-3:] == 'val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name[-4:] == 'test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:6] == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'BVIHFR':
                self.video_names = dataInfo['file_name'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:12] == 'BVIHFR_indep':
                    index_rd = np.random.permutation(22)
                    train_source = index_rd[:int(22 * 0.6)]
                    val_source = index_rd[int(22 * 0.6):int(22 * 0.8)]
                    test_source = index_rd[int(22 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 4, i * 4 + 4)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 4, i * 4 + 4)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 4, i * 4 + 4)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['file_name'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name[:7] == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            if database_name == 'LIVEHFR':
                self.video_names = dataInfo['filename'].tolist()
                self.score = dataInfo['MOS'].tolist()
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                if database_name[:13] == 'LIVEHFR_indep':
                    index_rd = np.random.permutation(16)
                    train_source = index_rd[:int(16 * 0.6)]
                    val_source = index_rd[int(16 * 0.6):int(16 * 0.8)]
                    test_source = index_rd[int(16 * 0.8):]

                    train_index = []
                    for i in train_source:
                        train_index = train_index + [j for j in range(i * 30, i * 30 + 30)]
                    val_index = []
                    for i in val_source:
                        val_index = val_index + [j for j in range(i * 30, i * 30 + 30)]
                    test_index = []
                    for i in test_source:
                        test_index = test_index + [j for j in range(i * 30, i * 30 + 30)]
                else:
                    index_rd = np.random.permutation(length)
                    train_index = index_rd[0:int(length * 0.6)]
                    val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                    test_index = index_rd[int(length * 0.8):]

                if database_name[-5:] == 'train':
                    self.video_names = dataInfo.iloc[train_index]['filename'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name[-3:] == 'val':
                    self.video_names = dataInfo.iloc[val_index]['filename'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name[-4:] == 'test':
                    self.video_names = dataInfo.iloc[test_index]['filename'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()


        elif database_name == 'LSVQ_all':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_val':
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
            self.video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            self.score = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]

        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:10] == 'LSVQ_1080p':
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
            # self.video_names = dataInfo['name'].tolist()
            # self.score = dataInfo['mos'].tolist()
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            if database_name == 'LSVQ_1080p_train':
                self.video_names = dataInfo.iloc[train_index]['name'].tolist()
                self.score = dataInfo.iloc[train_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_val':
                self.video_names = dataInfo.iloc[val_index]['name'].tolist()
                self.score = dataInfo.iloc[val_index]['mos'].tolist()
            elif database_name == 'LSVQ_1080p_test':
                self.video_names = dataInfo.iloc[test_index]['name'].tolist()
                self.score = dataInfo.iloc[test_index]['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.QCN_dir = QCN_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.dist_dir = dist_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[0] == 'K':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:7] == 'youtube' \
                or self.database_name[:7] == 'LiveVQC' \
                or self.database_name[:7] == 'CVD2014' \
                or self.database_name[:13] == 'LIVE-Qualcomm' \
                or self.database_name[:4] == 'LBVD' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                            :13] == 'Livestreaming':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name[:12] == 'LIVEYTGaming' or self.database_name[:8] == 'Waterloo' \
                or self.database_name[:6] == 'BVIHFR' or self.database_name[:7] == 'LIVEHFR':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        QCN_name = os.path.join(self.QCN_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        if self.database_name[:5] == 'BVISR':
            video_length_read = 5
        elif self.database_name[:8] == 'Waterloo':
            video_length_read = 9
        elif self.database_name[:6] == 'BVIHFR':
            video_length_read = 10
        elif self.database_name[:7] == 'LIVEHFR':
            video_length_read = 6
        elif self.database_name[:13] == 'Livestreaming':
            video_length_read = 7

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        QCN_feat = torch.zeros([video_length_read, 256 * 18])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)
        if self.database_name[:13] == 'LIVE-Qualcomm':
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                QCN_feat[i] = torch.from_numpy(
                    np.load(os.path.join(QCN_name, '{:03d}'.format(int(2 * i)) + '.npy'))).view(-1)
        else:
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
                QCN_feat[i] = torch.from_numpy(
                    np.load(os.path.join(QCN_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        if self.database_name[0] == 'K':
            video_name_str = video_name_str
        elif self.database_name[:7] == 'youtube' or self.database_name[:7] == 'LIVEHFR':
            video_name_str = video_name_str + '.mp4'
        elif self.database_name[:7] == 'CVD2014':
            video_name_str = video_name_str + '.avi'
        elif self.database_name[:13] == 'LIVE-Qualcomm' or self.database_name[:5] == 'BVISR' or self.database_name[
                                                                                                :13] == 'Livestreaming':
            video_name_str = video_name_str + '.yuv'

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        # 读取dist
        dist_forder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                i_index = int(2 * i)
            dist = np.load(os.path.join(dist_forder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist


        return transformed_video, transformed_feature, video_score, QCN_feat, transformed_dist, video_name_str


#################################xgc#################################################
class xgc_VideoDataset_images_with_dist_videomae_features(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, dist_dir, videomae_feat, aes_dir,
                 seed=0):
        super(xgc_VideoDataset_images_with_dist_videomae_features, self).__init__()

        print('random: ', seed)


        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)

        if database_name == 'xgc_train' or database_name == 'xgc_val':
            # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
            dataInfo.columns = ['file_names', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'MOS6']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)

            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.9)]
            val_index = index_rd[int(length * 0.9):]

            if database_name == 'xgc_train':
                print("train_index:", ', '.join(map(str, train_index.tolist())))
                self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                self.score1 = dataInfo.iloc[train_index]['MOS1'].tolist()
                self.score2 = dataInfo.iloc[train_index]['MOS2'].tolist()
                self.score3 = dataInfo.iloc[train_index]['MOS3'].tolist()
                self.score4 = dataInfo.iloc[train_index]['MOS4'].tolist()
                self.score5 = dataInfo.iloc[train_index]['MOS5'].tolist()
                self.score6 = dataInfo.iloc[train_index]['MOS6'].tolist()
            elif database_name == 'xgc_val':
                print(f'val_index: {val_index}')
                self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                self.score1 = dataInfo.iloc[val_index]['MOS1'].tolist()
                self.score2 = dataInfo.iloc[val_index]['MOS2'].tolist()
                self.score3 = dataInfo.iloc[val_index]['MOS3'].tolist()
                self.score4 = dataInfo.iloc[val_index]['MOS4'].tolist()
                self.score5 = dataInfo.iloc[val_index]['MOS5'].tolist()
                self.score6 = dataInfo.iloc[val_index]['MOS6'].tolist()

                # ========== 加标签分布统计 + 可视化 ==========
                import matplotlib.pyplot as plt
                # 拆分后的标签（以 MOS6 为例，其他标签同理）
                train_labels = dataInfo.iloc[train_index]['MOS6'].tolist()
                val_labels = dataInfo.iloc[val_index]['MOS6'].tolist()
                # 均值和标准差
                print("训练集标签均值: {:.3f}, 标准差: {:.3f}".format(np.mean(train_labels), np.std(train_labels)))
                print("验证集标签均值: {:.3f}, 标准差: {:.3f}".format(np.mean(val_labels), np.std(val_labels)))
                # 绘图
                plt.figure(figsize=(10, 5))
                plt.hist(train_labels, bins=20, alpha=0.6, label='Train MOS6', color='blue')
                plt.hist(val_labels, bins=20, alpha=0.6, label='Validation MOS6', color='orange')
                plt.xlabel("MOS6 Value")
                plt.ylabel("Frequency")
                plt.legend()
                plt.title("Label Distribution in Train and Validation Sets (MOS6)")
                plt.show()

        elif database_name[:3] == 'KVQ':
            # 读取 CSV 文件，假设是逗号（,）分隔的
            dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
            # 确保列名正确
            dataInfo.columns = ['file_names', 'score']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 需要删除的文件列表
            exclude_files = [
                'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4', 'train/0245.mp4',
                'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4', 'train/0833.mp4',
                'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4', 'train/0875.mp4',
                'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4', 'train/0903.mp4',
                'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4', 'train/0910.mp4',
                'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4', 'train/1106.mp4',
                'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4', 'train/1232.mp4',
                'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4', 'train/1624.mp4',
                'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4', 'train/1645.mp4',
                'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4', 'train/1750.mp4',
                'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4', 'train/1960.mp4',
                'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4', 'train/2037.mp4',
                'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4', 'train/2758.mp4',
                'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4', 'train/2786.mp4'
            ]

            # 使用 DataFrame 过滤掉要删除的文件
            dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

            if database_name == 'KVQ':
                self.video_names = dataInfo['file_names'].tolist()
                self.score = dataInfo['score'].tolist()
            else:
                length = len(dataInfo)
                random.seed(seed)
                np.random.seed(seed)

                # 生成顺序索引
                indices = np.arange(length)
                group_size = 7
                # 先处理完整的7个一组
                num_full_groups = length // group_size  # 整组数量
                groups = [indices[i * group_size: (i + 1) * group_size] for i in range(num_full_groups)]

                # 处理最后不足7个的那一组（如果有）
                remaining = indices[num_full_groups * group_size:]  # 取出剩余的索引（不足7个的最后一组）

                print(f"完整分组数量（每组7个）: {len(groups)}")
                print(f"最后不足7个的样本数量: {len(remaining)}")

                # 打乱所有完整的7个一组
                np.random.shuffle(groups)

                # 9:1划分完整的7个一组
                num_train_groups = int(len(groups) * 0.9)
                train_groups = groups[:num_train_groups]
                val_groups = groups[num_train_groups:]

                # 展平为索引
                train_index = np.concatenate(train_groups).tolist()
                val_index = np.concatenate(val_groups).tolist()

                # 加上不足7个的放入训练集
                if len(remaining) > 0:
                    train_index.extend(remaining.tolist())

                # 转 numpy 数组
                train_index = np.array(train_index)
                val_index = np.array(val_index)

                print("训练集数量：", len(train_index))
                print("验证集数量：", len(val_index))
                # 检查是否有重叠
                intersection = np.intersect1d(train_index, val_index)
                print(f"是否有重复索引？ {'有' if len(intersection) > 0 else '无'}")
                if len(intersection) > 0:
                    print(f"重复索引如下: {intersection}")
                # ================== 完整打印 ==================
                print("\n======== 训练集索引 (train_indices) ========")
                print(train_index.tolist())  # 转成 list 打印，方便阅读
                print("\n======== 验证集索引 (val_indices) ========")
                print(val_index.tolist())  # 转成 list 打印，方便阅读

                if database_name == 'KVQ_train':
                    # print("train_index:", ', '.join(map(str, train_index.tolist())))
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['score'].tolist()  # 修正为 `score`
                elif database_name == 'KVQ_val':
                    # print(f'val_index: {val_index}')
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['score'].tolist()
                elif database_name == 'KVQ_test':
                    self.video_names = dataInfo['file_names'].tolist()
                    self.score = dataInfo['score'].tolist()

        elif database_name == 'xgc_test':
            dataInfo.columns = ['file_names']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 获取视频文件名列表
            video_names = dataInfo['file_names'].tolist()
            self.video_names = video_names
            print(f"xgc_test length: {len(video_names)}")
        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.videomae_dir = videomae_feat
        self.aes_dir = aes_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'xgc_train' or self.database_name == 'xgc_val':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
            video_score1 = torch.FloatTensor(np.array(float(self.score1[idx])))
            video_score2 = torch.FloatTensor(np.array(float(self.score2[idx])))
            video_score3 = torch.FloatTensor(np.array(float(self.score3[idx])))
            video_score4 = torch.FloatTensor(np.array(float(self.score4[idx])))
            video_score5 = torch.FloatTensor(np.array(float(self.score5[idx])))
            video_score6 = torch.FloatTensor(np.array(float(self.score6[idx])))
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]
            # todo
            video_score1 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score2 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score3 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score4 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score5 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score6 = torch.FloatTensor(np.array(float(self.score[idx])))
        elif self.database_name[:3] == 'KVQ':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]
            # TODO
            video_score1 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score2 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score3 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score4 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score5 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score6 = torch.FloatTensor(np.array(float(self.score[idx])))
        else:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
            video_score1 = -100
            video_score2 = -100
            video_score3 = -100
            video_score4 = -100
            video_score5 = -100
            video_score6 = -100

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)

        # 读取帧并resize
        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取videomae_feat
        videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
        videomae_feat = np.load(videomae_folder_name + '.npy')
        transformed_videomae = torch.from_numpy(videomae_feat)

        # 读取美学信息
        aes_folder_name = os.path.join(self.aes_dir, video_name_str)
        transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
        aes = np.load(aes_folder_name + '.npy')
        transformed_aes = torch.from_numpy(aes)


        return transformed_video, video_score1, video_score2, video_score3, video_score4, video_score5, video_score6, transformed_dist, transformed_videomae, video_name_str, transformed_aes


class xgc_weight_concat_features(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, dist_dir, videomae_feat, aes_dir, slowfast_dir, feature_type, lp_dir,
                 seed=0):
        super(xgc_weight_concat_features, self).__init__()

        print('seed: ', seed)


        # 读取 TXT 文件，假设是以 Tab（\t）分隔的
        dataInfo = pd.read_csv(filename_path, sep="\t", header=None)

        if database_name == 'xgc_train' or database_name == 'xgc_val':
            # 重命名列，第一列是视频文件名，后面列是 MOS 相关数据
            dataInfo.columns = ['file_names', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'MOS6']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)

            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.9)]
            val_index = index_rd[int(length * 0.9):]

            if database_name == 'xgc_train':
                print("train_index:", ', '.join(map(str, train_index.tolist())))
                self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                self.score1 = dataInfo.iloc[train_index]['MOS1'].tolist()
                self.score2 = dataInfo.iloc[train_index]['MOS2'].tolist()
                self.score3 = dataInfo.iloc[train_index]['MOS3'].tolist()
                self.score4 = dataInfo.iloc[train_index]['MOS4'].tolist()
                self.score5 = dataInfo.iloc[train_index]['MOS5'].tolist()
                self.score6 = dataInfo.iloc[train_index]['MOS6'].tolist()
            elif database_name == 'xgc_val':
                print("val_index:", ', '.join(map(str, val_index.tolist())))
                self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                self.score1 = dataInfo.iloc[val_index]['MOS1'].tolist()
                self.score2 = dataInfo.iloc[val_index]['MOS2'].tolist()
                self.score3 = dataInfo.iloc[val_index]['MOS3'].tolist()
                self.score4 = dataInfo.iloc[val_index]['MOS4'].tolist()
                self.score5 = dataInfo.iloc[val_index]['MOS5'].tolist()
                self.score6 = dataInfo.iloc[val_index]['MOS6'].tolist()

        elif database_name == 'xgc_test':
            dataInfo.columns = ['file_names']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 获取视频文件名列表
            video_names = dataInfo['file_names'].tolist()
            self.video_names = video_names
            print(f"xgc_test length: {len(video_names)}")
        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.videomae_dir = videomae_feat
        self.aes_dir = aes_dir
        self.slowfast_dir = slowfast_dir
        self.feature_type = feature_type
        self.lp_dir = lp_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'xgc_train' or self.database_name == 'xgc_val':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
            video_score1 = torch.FloatTensor(np.array(float(self.score1[idx])))
            video_score2 = torch.FloatTensor(np.array(float(self.score2[idx])))
            video_score3 = torch.FloatTensor(np.array(float(self.score3[idx])))
            video_score4 = torch.FloatTensor(np.array(float(self.score4[idx])))
            video_score5 = torch.FloatTensor(np.array(float(self.score5[idx])))
            video_score6 = torch.FloatTensor(np.array(float(self.score6[idx])))
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]
            # todo
            video_score1 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score2 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score3 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score4 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score5 = torch.FloatTensor(np.array(float(self.score[idx])))
            video_score6 = torch.FloatTensor(np.array(float(self.score[idx])))
        elif self.database_name == 'xgc_test':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
            video_score1 = -100
            video_score2 = -100
            video_score3 = -100
            video_score4 = -100
            video_score5 = -100
            video_score6 = -100

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        lp_name = os.path.join(self.lp_dir, video_name_str)
        resized_lp = torch.zeros([video_length_read, 5 * 512])


        # fix random
        seed = np.random.randint(42)
        random.seed(seed)

        # 读取帧并resize 读取LP
        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame
            if self.database_name[:4] == 'LSVQ':
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1).repeat(1, 2)
            else:
                resized_lp[i] = torch.from_numpy(
                    np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i)) + '.npy'))).view(-1)

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * i_index)) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取videomae_feat
        videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
        videomae_feat = np.load(videomae_folder_name + '.npy')
        transformed_videomae = torch.from_numpy(videomae_feat)

        # 读取美学信息
        aes_folder_name = os.path.join(self.aes_dir, video_name_str)
        transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
        aes = np.load(aes_folder_name + '.npy')
        transformed_aes = torch.from_numpy(aes)

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.slowfast_dir, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.slowfast_dir, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i  # TODO
                if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
                    i_index = int(2 * i)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.slowfast_dir, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D


        return transformed_video, video_score1, video_score2, video_score3, video_score4, video_score5, video_score6, transformed_dist, transformed_videomae, video_name_str, transformed_aes, transformed_feature, resized_lp

#################################KVQ#################################################
class kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, dist_dir, videomae_feat,
                 seed=0):
        super(kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features, self).__init__()

        print('random: ', seed)

        if database_name[:3] == 'KVQ':
            # 读取 CSV 文件，假设是逗号（,）分隔的
            dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
            # 确保列名正确
            dataInfo.columns = ['file_names', 'score']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 需要删除的文件列表
            exclude_files = [
                'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4', 'train/0245.mp4',
                'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4', 'train/0833.mp4',
                'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4', 'train/0875.mp4',
                'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4', 'train/0903.mp4',
                'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4', 'train/0910.mp4',
                'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4', 'train/1106.mp4',
                'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4', 'train/1232.mp4',
                'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4', 'train/1624.mp4',
                'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4', 'train/1645.mp4',
                'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4', 'train/1750.mp4',
                'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4', 'train/1960.mp4',
                'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4', 'train/2037.mp4',
                'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4', 'train/2758.mp4',
                'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4', 'train/2786.mp4'
            ]

            # 使用 DataFrame 过滤掉要删除的文件
            dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

            if database_name == 'KVQ':
                self.video_names = dataInfo['file_names'].tolist()
                self.score = dataInfo['score'].tolist()
            else:
                length = len(dataInfo)
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)

                train_index = index_rd[:int(length * 0.9)]
                val_index = index_rd[int(length * 0.9):]

                if database_name == 'KVQ_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['score'].tolist()  # 修正为 `score`
                elif database_name == 'KVQ_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['score'].tolist()
                elif database_name == 'KVQ_test':
                    self.video_names = dataInfo['file_names'].tolist()
                    self.score = dataInfo['score'].tolist()

        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.videomae_dir = videomae_feat

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:3] == 'KVQ':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]


        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)


        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 4
        start_frame = 2

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i) + start_frame) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            # if self.database_name[:7] == 'youtube' or self.database_name[:13] == 'LIVE-Qualcomm':
            if self.database_name[:13] == 'LIVE-Qualcomm':
                i_index = int(2 * i)
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * i_index) + start_frame) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取videomae_feat
        videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
        videomae_feat = np.load(videomae_folder_name + '.npy')
        videomae_feat = videomae_feat[2:6, :]  # 取 2,3,4,5 行，所有列
        transformed_videomae = torch.from_numpy(videomae_feat)


        return transformed_video, video_score, transformed_dist, transformed_videomae, video_name_str

class kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features_v2(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, dist_dir, videomae_feat, aes_dir, seed=0):
        super(kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features_v2, self).__init__()

        print("random: ", seed)

        if database_name[:3] == 'KVQ':
            # 读取 CSV 文件，假设是逗号（,）分隔的
            dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
            # 确保列名正确
            dataInfo.columns = ['file_names', 'score']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 需要删除的文件列表
            exclude_files = [
                'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4', 'train/0245.mp4',
                'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4', 'train/0833.mp4',
                'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4', 'train/0875.mp4',
                'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4', 'train/0903.mp4',
                'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4', 'train/0910.mp4',
                'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4', 'train/1106.mp4',
                'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4', 'train/1232.mp4',
                'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4', 'train/1624.mp4',
                'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4', 'train/1645.mp4',
                'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4', 'train/1750.mp4',
                'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4', 'train/1960.mp4',
                'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4', 'train/2037.mp4',
                'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4', 'train/2758.mp4',
                'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4', 'train/2786.mp4'
            ]

            # 使用 DataFrame 过滤掉要删除的文件
            dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

            if database_name == 'KVQ':
                self.video_names = dataInfo['file_names'].tolist()
                self.score = dataInfo['score'].tolist()
            else:
                length = len(dataInfo)
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[:int(length * 0.9)]
                val_index = index_rd[int(length * 0.9):]


                # 生成顺序索引
                indices = np.arange(length)
                group_size = 7
                # 先处理完整的7个一组
                num_full_groups = length // group_size  # 整组数量
                groups = [indices[i * group_size: (i + 1) * group_size] for i in range(num_full_groups)]

                # 处理最后不足7个的那一组（如果有）
                remaining = indices[num_full_groups * group_size:]  # 取出剩余的索引（不足7个的最后一组）

                print(f"完整分组数量（每组7个）: {len(groups)}")
                print(f"最后不足7个的样本数量: {len(remaining)}")

                # 打乱所有完整的7个一组
                np.random.shuffle(groups)

                # 9:1划分完整的7个一组
                num_train_groups = int(len(groups) * 0.9)
                train_groups = groups[:num_train_groups]
                val_groups = groups[num_train_groups:]

                # 展平为索引
                train_index = np.concatenate(train_groups).tolist()
                val_index = np.concatenate(val_groups).tolist()

                # 加上不足7个的放入训练集
                if len(remaining) > 0:
                    train_index.extend(remaining.tolist())

                # 转 numpy 数组
                train_index = np.array(train_index)
                val_index = np.array(val_index)

                print("训练集数量：", len(train_index))
                print("验证集数量：", len(val_index))
                # 检查是否有重叠
                intersection = np.intersect1d(train_index, val_index)
                print(f"是否有重复索引？ {'有' if len(intersection) > 0 else '无'}")
                if len(intersection) > 0:
                    print(f"重复索引如下: {intersection}")
                # ================== 完整打印 ==================
                print("\n======== 训练集索引 (train_indices) ========")
                print(train_index.tolist())  # 转成 list 打印，方便阅读
                print("\n======== 验证集索引 (val_indices) ========")
                print(val_index.tolist())  # 转成 list 打印，方便阅读


                if database_name == 'KVQ_train':
                    print('len train_index: ', len(train_index))
                    print(train_index.tolist())
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['score'].tolist()
                elif database_name == 'KVQ_val':
                    print('len val_index: ', len(val_index))
                    print(val_index.tolist())
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['score'].tolist()
                elif database_name == 'KVQ_test':
                    self.video_names = dataInfo['file_names'].tolist()
                    self.score = dataInfo['score'].tolist()
        elif database_name == 'LSVQ_train':
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
            # self.video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            # self.score = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
        elif database_name == 'LSVQ_test':
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
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.videomae_dir = videomae_feat
        self.aes_dir = aes_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:3] == 'KVQ':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]
            frame_start = 2
        elif self.database_name[:4] == 'LSVQ':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]
            frame_start = 2

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)


        video_channel = 3

        # video_height_crop = self.crop_size
        # video_width_crop = self.crop_size
        video_height_crop = 224
        video_width_crop = 224

        video_length_read = 4

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * (i+frame_start))) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * (i_index+frame_start))) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取videomae_feat
        if self.database_name[:3] == 'KVQ':
            videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
            videomae_feat = np.load(videomae_folder_name + '.npy')
            transformed_videomae = torch.from_numpy(videomae_feat)
        elif self.database_name[:4] == 'LSVQ':
            videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
            videomae_feat = np.load(videomae_folder_name + '.npy')
            transformed_videomae = torch.from_numpy(videomae_feat)[2:6, :]

        #读取美学信息
        aes_folder_name = os.path.join(self.aes_dir, video_name_str)
        transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
        aes = np.load(aes_folder_name + '.npy')
        transformed_aes = torch.from_numpy(aes)

        return transformed_video, video_score, transformed_dist, transformed_videomae, video_name_str, transformed_aes

class kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features_v3(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, dist_dir, videomae_feat, aes_dir, lp_dir, seed=0):
        super(kvq_VideoDataset_images_with_LP_motion_dist_aes_videomae_features_v3, self).__init__()

        if database_name[:3] == 'KVQ':
            # 读取 CSV 文件，假设是逗号（,）分隔的
            dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
            # 确保列名正确
            dataInfo.columns = ['file_names', 'score']
            # 确保文件名是字符串
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            # 需要删除的文件列表
            exclude_files = [
                'train/0239.mp4', 'train/0240.mp4', 'train/0241.mp4', 'train/0242.mp4', 'train/0243.mp4', 'train/0244.mp4', 'train/0245.mp4',
                'train/0827.mp4', 'train/0828.mp4', 'train/0829.mp4', 'train/0830.mp4', 'train/0831.mp4', 'train/0832.mp4', 'train/0833.mp4',
                'train/0869.mp4', 'train/0870.mp4', 'train/0871.mp4', 'train/0872.mp4', 'train/0873.mp4', 'train/0874.mp4', 'train/0875.mp4',
                'train/0897.mp4', 'train/0898.mp4', 'train/0899.mp4', 'train/0900.mp4', 'train/0901.mp4', 'train/0902.mp4', 'train/0903.mp4',
                'train/0904.mp4', 'train/0905.mp4', 'train/0906.mp4', 'train/0907.mp4', 'train/0908.mp4', 'train/0909.mp4', 'train/0910.mp4',
                'train/1100.mp4', 'train/1101.mp4', 'train/1102.mp4', 'train/1103.mp4', 'train/1104.mp4', 'train/1105.mp4', 'train/1106.mp4',
                'train/1226.mp4', 'train/1227.mp4', 'train/1228.mp4', 'train/1229.mp4', 'train/1230.mp4', 'train/1231.mp4', 'train/1232.mp4',
                'train/1618.mp4', 'train/1619.mp4', 'train/1620.mp4', 'train/1621.mp4', 'train/1622.mp4', 'train/1623.mp4', 'train/1624.mp4',
                'train/1639.mp4', 'train/1640.mp4', 'train/1641.mp4', 'train/1642.mp4', 'train/1643.mp4', 'train/1644.mp4', 'train/1645.mp4',
                'train/1744.mp4', 'train/1745.mp4', 'train/1746.mp4', 'train/1747.mp4', 'train/1748.mp4', 'train/1749.mp4', 'train/1750.mp4',
                'train/1954.mp4', 'train/1955.mp4', 'train/1956.mp4', 'train/1957.mp4', 'train/1958.mp4', 'train/1959.mp4', 'train/1960.mp4',
                'train/2031.mp4', 'train/2032.mp4', 'train/2033.mp4', 'train/2034.mp4', 'train/2035.mp4', 'train/2036.mp4', 'train/2037.mp4',
                'train/2752.mp4', 'train/2753.mp4', 'train/2754.mp4', 'train/2755.mp4', 'train/2756.mp4', 'train/2757.mp4', 'train/2758.mp4',
                'train/2780.mp4', 'train/2781.mp4', 'train/2782.mp4', 'train/2783.mp4', 'train/2784.mp4', 'train/2785.mp4', 'train/2786.mp4'
            ]

            # 使用 DataFrame 过滤掉要删除的文件
            dataInfo = dataInfo[~dataInfo['file_names'].isin(exclude_files)]

            if database_name == 'KVQ':
                self.video_names = dataInfo['file_names'].tolist()
                self.score = dataInfo['score'].tolist()
            else:
                length = len(dataInfo)
                random.seed(seed)
                np.random.seed(seed)

                # 生成顺序索引
                indices = np.arange(length)
                group_size = 7
                # 先处理完整的7个一组
                num_full_groups = length // group_size  # 整组数量
                groups = [indices[i * group_size: (i + 1) * group_size] for i in range(num_full_groups)]

                # 处理最后不足7个的那一组（如果有）
                remaining = indices[num_full_groups * group_size:]  # 取出剩余的索引（不足7个的最后一组）

                print(f"完整分组数量（每组7个）: {len(groups)}")
                print(f"最后不足7个的样本数量: {len(remaining)}")

                # 打乱所有完整的7个一组
                np.random.shuffle(groups)

                # 9:1划分完整的7个一组
                num_train_groups = int(len(groups) * 0.9)
                train_groups = groups[:num_train_groups]
                val_groups = groups[num_train_groups:]

                # 展平为索引
                train_index = np.concatenate(train_groups).tolist()
                val_index = np.concatenate(val_groups).tolist()

                # 加上不足7个的放入训练集
                if len(remaining) > 0:
                    train_index.extend(remaining.tolist())

                # 转 numpy 数组
                train_index = np.array(train_index)
                val_index = np.array(val_index)

                print("训练集数量：", len(train_index))
                print("验证集数量：", len(val_index))
                # 检查是否有重叠
                intersection = np.intersect1d(train_index, val_index)
                print(f"是否有重复索引？ {'有' if len(intersection) > 0 else '无'}")
                if len(intersection) > 0:
                    print(f"重复索引如下: {intersection}")
                # ================== 完整打印 ==================
                print("\n======== 训练集索引 (train_indices) ========")
                print(train_index.tolist())  # 转成 list 打印，方便阅读
                print("\n======== 验证集索引 (val_indices) ========")
                print(val_index.tolist())  # 转成 list 打印，方便阅读

                if database_name == 'KVQ_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['score'].tolist()  # 修正为 `score`
                elif database_name == 'KVQ_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['score'].tolist()
                elif database_name == 'KVQ_test':
                    self.video_names = dataInfo['file_names'].tolist()
                    self.score = dataInfo['score'].tolist()

        elif database_name[:4] == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                if '298.mp4' == dataInfo['video_names'][i][0][0] \
                        or '378.mp4' == dataInfo['video_names'][i][0][0] \
                        or '420.mp4' == dataInfo['video_names'][i][0][0] \
                        or '506.mp4' == dataInfo['video_names'][i][0][0] \
                        or '529.mp4' == dataInfo['video_names'][i][0][0] \
                        or '574.mp4' == dataInfo['video_names'][i][0][0] \
                        or '594.mp4' == dataInfo['video_names'][i][0][0] \
                        or '668.mp4' == dataInfo['video_names'][i][0][0] \
                        or '697.mp4' == dataInfo['video_names'][i][0][0] \
                        or '828.mp4' == dataInfo['video_names'][i][0][0] \
                        or '847.mp4' == dataInfo['video_names'][i][0][0] \
                        or '972.mp4' == dataInfo['video_names'][i][0][0] \
                        or '180.mp4' == dataInfo['video_names'][i][0][0] \
                        or '248.mp4' == dataInfo['video_names'][i][0][0] \
                        or '266.mp4' == dataInfo['video_names'][i][0][0] \
                        or '454.mp4' == dataInfo['video_names'][i][0][0] \
                        or '711.mp4' == dataInfo['video_names'][i][0][0] \
                        or '800.mp4' == dataInfo['video_names'][i][0][0] \
                        or '862.mp4' == dataInfo['video_names'][i][0][0] \
                        or '867.mp4' == dataInfo['video_names'][i][0][0]:
                    continue  # 不处理298.mp4
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])

            if database_name == 'LBVD':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.9)]
                print(f'LBVD train_index: {train_index}')
                val_index = index_rd[int(length * 0.9):]
                print(f'LBVD val_index: {val_index}')
                if database_name == 'LBVD_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LBVD_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.dist_dir = dist_dir
        self.videomae_dir = videomae_feat
        self.aes_dir = aes_dir
        self.lp_dir = lp_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:3] == 'KVQ':
            video_name = self.video_names[idx].split('/')[1]
            video_name_str = video_name[:-4]
            frame_start = 2
        elif self.database_name[:4] == 'LBVD':
            video_name_str = self.video_names[idx][:-4]
            frame_start = 3

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        lp_name = os.path.join(self.lp_dir, video_name_str)


        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 4



        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        # fix random
        seed = np.random.randint(20231001)
        random.seed(seed)

        resized_lp = torch.zeros([video_length_read, 5 * 512])
        # 读取帧
        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * (i+frame_start))) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame
            # 读取lp
            # todo 1 * i + frame_start-1 LBVD提取lp代码没写好，保存的时候和KVQ一样了，需要 -1
            resized_lp[i] = torch.from_numpy(np.load(os.path.join(lp_name, '{:03d}'.format(int(1 * i + frame_start)) + '.npy'))).view(-1)

        # 读取dist
        dist_folder_name = os.path.join(self.dist_dir, video_name_str)
        transformed_dist = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            i_index = i  # TODO
            dist = np.load(os.path.join(dist_folder_name, '{:03d}'.format(int(1 * (i_index+frame_start))) + '.npy'))
            dist = torch.from_numpy(dist)
            dist = dist.squeeze()
            transformed_dist[i] = dist

        # 读取videomae_feat
        videomae_folder_name = os.path.join(self.videomae_dir, video_name_str)
        videomae_feat = np.load(videomae_folder_name + '.npy')
        transformed_videomae = torch.from_numpy(videomae_feat)

        # 读取美学信息
        aes_folder_name = os.path.join(self.aes_dir, video_name_str)
        transformed_aes = torch.zeros([1, 784])  # 一个视频只提取了一个(1, 784)
        aes = np.load(aes_folder_name + '.npy')
        transformed_aes = torch.from_numpy(aes)

        return transformed_video, video_score, transformed_dist, transformed_videomae, video_name_str, transformed_aes, resized_lp