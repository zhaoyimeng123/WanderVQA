import torch
import numpy as np

from model import modular

# 加载训练好的模型
model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10(feat_len=8, sr=True, tr=True, dr=True, ar=True,
                                                                           dropout_sp=0.1, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2)
state_dict = torch.load('/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_LSVQ_plcc_rank_NR_vNone_epoch_50_SRCC_0.888277.pth')
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)
model.eval()
# 打印模型结构，找到你想要的层
print(model)

# 获取 transformer 的最后一个 ResidualAttentionBlock
resblock = model.feature_extraction.transformer.resblocks[11]  # 获取最后一个 ResBlock

# 获取该 ResBlock 中的 attn.out_proj 权重
a = 2
attention_weight = resblock.ln_2.weight.data.to('cpu').numpy()  # 获取 attn.out_proj 的权重

# 打印权重形状以确认
print(f"Weight shape: {attention_weight.shape}")

# 保存为.npy文件
np.save("weights.npy", attention_weight)  # 替换为你要保存的路径
