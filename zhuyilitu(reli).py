import os
import numpy as np
import torch, cv2
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.models.archs.NAFNet_arch import NAFNet


def main():
    # 1加载模型
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    path_model = '/home/user/gongbingbing/NAFNet-main/experiments/NAFNet-GoPro-width32/models/net_g_latest.pth'
    model.load_state_dict(torch.load(path_model)['params'])
    model = model.cuda()
    target_layers = [model.middle_blks]

    # 2准备图像 构建输入图像的Tensor形式，归一化，使其能传送到model里面去计算
    img_path = "/home/user/gongbingbing/NAFNet-main/basicsr/4.jpg"
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (256, 256))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # print(input_tensor.shape)

    # 3初始化CAM对象，包括模型，目标层以及是否使用cuda等
    cam = GradCAM(model=model, target_layers=target_layers)
    # 4)选定目标类别，如果不设置，则默认为分数最高的那一类
    targets = None

    # 5)计算cam
    # cam.batch_size = 1
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # [batch,256,256]

    # 6.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    grayscale_cam = grayscale_cam[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'first_try.jpg', cam_image)


if __name__ == '__main__':
    main()
