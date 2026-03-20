import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio


# 剪裁
def extract_frame_clip(videos_dir, video_name, save_folder):
    try:
        video_name = video_name.split('/')[1]
        filename = os.path.join(videos_dir, video_name)
        print(filename)
        video_name_str = video_name[:-4]
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            print(f"无法打开视频文件: {filename}")
            return

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        # **防止 ZeroDivisionError**
        if video_frame_rate is None or video_frame_rate == 0:
            print(f"警告：视频 {video_name} 的帧率无效 ({video_frame_rate})，跳过该视频。")
            return

        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames


    except:
        print(filename)

    else:

        video_read_index = 0

        frame_idx = 0

        video_length_min = 8

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate/2)):
                    read_frame = frame

                    # 居中裁剪操作
                    if video_height > video_width:
                        crop_size = video_width
                        start_y = (video_height - crop_size) // 2
                        start_x = 0
                    else:
                        crop_size = video_height
                        start_x = (video_width - crop_size) // 2
                        start_y = 0

                    read_frame = read_frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), read_frame)

        return


def extract_frame(videos_dir, video_name, save_folder):
    try:
        video_name = video_name.split('/')[1]
        filename = os.path.join(videos_dir, video_name)
        print(filename)
        video_name_str = video_name[:-4]
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            print(f"无法打开视频文件: {filename}")
            return

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        # **防止 ZeroDivisionError**
        if video_frame_rate is None or video_frame_rate == 0:
            print(f"警告：视频 {video_name} 的帧率无效 ({video_frame_rate})，跳过该视频。")
            return

        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames


    except:
        print(filename)

    else:

        video_read_index = 0

        frame_idx = 0

        video_length_min = 8

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate/2)):
                    read_frame = frame
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), read_frame)

        return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


if __name__ == '__main__':
    videos_dir = '/data/dataset/KVQ/test'
    filename_path = '/data/dataset/KVQ/test_data.csv'
    save_folder = '/data/user/zhaoyimeng/ModularBVQA/data/KVQ/test/kvq_image_all_fps1_clip'

    # 读取 CSV 文件，假设是逗号（,）分隔的
    dataInfo = pd.read_csv(filename_path, sep=",", header=0)  # header=0 表示第一行为列名
    # 确保列名正确
    dataInfo.columns = ['file_names', 'score']
    # 确保文件名是字符串
    dataInfo['file_names'] = dataInfo['file_names'].astype(str)
    # 获取视频文件名列表
    video_names = dataInfo['file_names'].tolist()

    # 计算视频数量
    n_video = len(video_names)
    
    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame_clip(videos_dir, video_name, save_folder)

