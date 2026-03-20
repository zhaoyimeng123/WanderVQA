
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载权重和原图
weights = np.load('/data/user/zhaoyimeng/ModularBVQA/weights.npy')  # 替换为你的权重文件路径
original_image = cv2.imread('/data/user/zhaoyimeng/ModularBVQA/data/cvd2014_image_all_fps1/Test02_City_D04/006.png')  # 替换为你的图像路径
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 检查原图尺寸
print(f"Original image shape: {original_image.shape}")

# 将权重 reshape 为 28x28
weights_reshaped = weights.reshape(32, 24)

# 获取 weights 的中心区域
# start = (weights.shape[0] - 224) // 2
# end = start + 224
# 裁剪出中心区域
# center_weights = weights[start:end, start:end]

# 将 28x28 的权重图插值到 224x224
weights_resized = cv2.resize(weights_reshaped, (224, 224), interpolation=cv2.INTER_LINEAR)

# 将权重归一化到 0-1 范围
weights_normalized = (weights_resized - weights_resized.min()) / (weights_resized.max() - weights_resized.min())

# 生成热力图
cmap = plt.get_cmap('jet')  # 使用 'jet' 颜色映射
heatmap = cmap(weights_normalized)  # 生成热力图
heatmap = np.uint8(heatmap * 255)  # 转换到0-255范围并转换为8位整数

# 去掉 alpha 通道，只保留 RGB 通道
heatmap_rgb = heatmap[..., :3]
print(f"Heatmap RGB shape: {heatmap_rgb.shape}")

# 检查尺寸是否一致
if original_image.shape[:2] != heatmap_rgb.shape[:2]:
    heatmap_rgb = cv2.resize(heatmap_rgb, (original_image.shape[1], original_image.shape[0]))

# 将热力图与原图叠加
alpha = 0.6  # 设置叠加透明度
overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis('off')  # 不显示坐标轴
plt.show()
