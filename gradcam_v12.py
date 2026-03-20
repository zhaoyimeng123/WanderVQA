import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import modular


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化 Grad-CAM
    model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12(feat_len=8).to(device)
    # 模型使用DataParallel后多了"module."
    state_dict = torch.load(
        '/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12_LSVQ_plcc_rank_NR_v555_epoch_58_SRCC_0.884826.pth')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model = model.float()
    model.eval()
    print(model)
    target_layers = [model.feature_extraction.ln_post]

    # 2准备图像 构建输入图像的Tensor形式，归一化，使其能传送到model里面去计算
    # 2准备图像 构建输入图像的Tensor形式，归一化，使其能传送到model里面去计算
    img_path = "/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1/10008004183/000.png"
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (256, 256))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = torch.randn(1, 1248256).to(device)

    # 3初始化CAM对象，包括模型，目标层以及是否使用cuda等
    cam = GradCAM(model=model, target_layers=target_layers)

    with torch.no_grad():
        model_output = model(input_tensor)

    # 定义目标函数，返回回归模型的输出值作为目标
    def target_fn(output):
        return output  # 直接返回模型的预测值作为目标

    target_value = float(model_output)

    # 4)选定目标类别，如果不设置，则默认为分数最高的那一类
    targets = None

    # 5)计算cam
    # cam.batch_size = 1
    grayscale_cam = cam(input_tensor=input_tensor, targets=[target_fn])

    # 6.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    grayscale_cam = grayscale_cam[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'first_try.jpg', cam_image)


if __name__ == '__main__':
    main()
