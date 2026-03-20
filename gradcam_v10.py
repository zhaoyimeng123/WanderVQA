import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

from model import modular


# 假设您的模型已经加载好，名称为 `model`，并且已加载权重
# 定义 Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # 注册前向和反向钩子
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def save_gradients_hook(grad):
            self.gradients = grad

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()



    def __call__(self, x, x_3D_features, lp, dist, aes, videomae):
        # Forward pass
        output = self.model(x, x_3D_features, lp, dist, aes, videomae)
        # Backward pass for the first sample in the batch (or mean)


        output.mean().backward(retain_graph=True)

        # Pool the gradients across the spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])

        # Weight activations by gradients
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)  # ReLU
        heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[4]))  # Resize to original image size
        heatmap = heatmap - np.min(heatmap)  # Normalize to [0, 1]
        heatmap = heatmap / np.max(heatmap)

        return heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化 Grad-CAM
model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10(feat_len=8,
                                                                                                     sr=True, tr=True,
                                                                                                     dr=True, ar=True,
                                                                                                     dropout_sp=0.1,
                                                                                                     dropout_tp=0.2,
                                                                                                     dropout_dp=0.2,
                                                                                                     dropout_ap=0.2).to(device)
# 模型使用DataParallel后多了"module."
state_dict = torch.load(
    '/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_LSVQ_plcc_rank_NR_vNone_epoch_50_SRCC_0.888277.pth')
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)

model = model.float()

model.eval()
print('load model success!')

# 设定用于 Grad-CAM 的目标层名称，例如 Transformer 最后一层
target_layer = "feature_extraction.transformer.resblocks[-1]"

# 初始化 Grad-CAM
grad_cam = GradCAM(model, target_layer)

# 预处理输入数据
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# 假设输入数据 x, x_3D_features, lp, dist, aes, videomae 都已准备好
# 生成输入张量
img_path = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_image_all_fps1_motion/ia-batch20/lpama-LHS_Boys_Varsity_Basketball_vs._Clinton_01.03.19/000.png'
img = cv2.imread(img_path)[:, :, ::-1]  # BGR 转为 RGB
img_pil = Image.fromarray(img)  # 将 numpy.ndarray 转换为 PIL 图像
# 预处理输入数据
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 将 PIL 图像转换为 tensor
x = preprocess(img_pil).unsqueeze(0)  # 添加 batch 维度
x = x.unsqueeze(1).repeat(1, 8, 1, 1, 1).to(device)
x_3D_features = torch.randn(1, 8, 512).to(device)
lp = torch.randn(1, 8, 4096).to(device)
dist = torch.randn(1, 8, 4096).to(device)
aes = torch.randn(1, 8, 512).to(device)
videomae = torch.randn(1, 8, 1408).to(device)

# 获取 Grad-CAM 热力图
heatmap = grad_cam(x, x_3D_features, lp, dist, aes, videomae)

# 将热力图叠加在原始图像上
img = cv2.imread(img_path)
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

# 保存或显示结果
cv2.imwrite("gradcam_result.jpg", superimposed_img)

# 移除钩子以释放资源
grad_cam.remove_hooks()
