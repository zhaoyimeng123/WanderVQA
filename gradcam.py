import argparse
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, \
    LayerCAM, FullGrad
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from model import modular

'''参数设置'''
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    return args


'''模型加载'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ourModel():
    # 实例化模型
    model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12(feat_len=8)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    model = model.float()
    model.load_state_dict(torch.load(
        '/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12_LSVQ_plcc_NR_v444_epoch_3_SRCC_0.014907.pth'))
    return model

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), tensor.size(1),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':

    args = get_args()
    methods = {"gradcam": GradCAM,
               "scorecam": ScoreCAM,
               "gradcam++": GradCAMPlusPlus,
               "ablationcam": AblationCAM,
               "xgradcam": XGradCAM,
               "eigencam": EigenCAM,
               "eigengradcam": EigenGradCAM,
               "layercam": LayerCAM,
               "fullgrad": FullGrad,
               }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = ourModel()
    model.eval()
    # target_layers = [model.module.feature_extraction.ln_post]
    target_layers = [model.feature_extraction.ln_post]

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=None)

    image_item_ab = os.path.join('/data/user/zhaoyimeng/ModularBVQA/data/konvid1k_image_all_fps1/10008004183/000.png')
    rgb_img = cv2.imread(image_item_ab, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
    input_tensor = input_tensor.unsqueeze(1).repeat(1, 8, 1, 1, 1)
    dist = torch.randn(1, 8, 4096)
    videomae = torch.randn(1, 8, 1408)

    batch_size = input_tensor.shape[0]
    # 将 video 展平到 (b, -1) 的形状
    video_flat = input_tensor.view(batch_size, -1)  # Shape: (b, 8*3*224*224)
    dist_flat = dist.view(batch_size, -1)  # Shape: (b, 8*4096)
    videomae_flat = videomae.view(batch_size, -1)  # Shape: (b, 8*1408)

    inputs = torch.cat([video_flat, dist_flat, videomae_flat], dim=1)  # Shape: (b, total_features)
    inputs = inputs.to(device)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=inputs,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, 0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    final_image_path = '/data/user/zhaoyimeng/ModularBVQA/result'
    cv2.imwrite(f'{final_image_path}_cam.jpg', cam_image)
