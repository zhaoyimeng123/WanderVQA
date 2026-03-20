import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientImageSplitter(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(EfficientImageSplitter, self).__init__()

        # 1. 先对整个输入图像进行 CNN 处理
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)  # 共享卷积

        # 2. 处理每个 7×7 patch 的 CNN
        self.patch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 3. 上采样
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, groups=1)


    def forward(self, img):
            """
            img: (B, F, C, H, W)
            output: (B, F, C, 224, 224)
            """
            B, F, C, H, W = img.shape
            patch_size_H, patch_size_W = H // 7, W // 7

            outputs = []
            for f in range(F):  # 遍历帧
                frame = img[:, f]  # (B, C, H, W)

                # 1. 全局卷积提取特征
                feature_map = self.global_conv(frame)  # (B, out_channels, H, W)

                # 2. 用 unfold 提取 7×7 patch
                patches = feature_map.unfold(2, patch_size_H, patch_size_H).unfold(3, patch_size_W, patch_size_W)
                patches = patches.contiguous().view(B, -1, 49, patch_size_H, patch_size_W)  # (B, C, 49, patch_H, patch_W)

                # 3. 对每个 patch 进行独立卷积
                new_patches = []
                for i in range(49):
                    patch = patches[:, :, i, :, :]
                    patch = self.patch_conv(patch)  # 进一步卷积

                    # 深度卷积
                    patch = self.depth_conv(patch)
                    patch = self.point_conv(patch)


                    patch = self.upsample(patch)  # 上采样到 (32,32)
                    new_patches.append(patch)

                # 4. 重新组合为 224×224
                new_patches = torch.stack(new_patches, dim=1).view(B, 7, 7, -1, 32, 32)
                new_patches = new_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
                frame_output = new_patches.view(B, -1, 224, 224)  # (B, C, 224, 224)



                outputs.append(frame_output)

            return torch.stack(outputs, dim=1)  # (B, F, C, 224, 224)


# 测试
img = torch.randn(2, 8, 3, 1920, 1080)  # (B=2, F=8, C=3, H=256, W=256)
splitter = EfficientImageSplitter()
output = splitter(img)
print(output.shape)  # 预期 (2, 8, 32, 224, 224)
