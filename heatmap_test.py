import torch

from model import modular

# 创建全局变量来存储目标层的输出
extracted_features = None


# 定义钩子函数
def hook_fn(module, input, output):
    global extracted_features
    extracted_features = output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 实例化模型
model = modular.ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10(feat_len=8,
                                                                                                     sr=True, tr=True,
                                                                                                     dr=True, ar=True,
                                                                                                     dropout_sp=0.1,
                                                                                                     dropout_tp=0.2,
                                                                                                     dropout_dp=0.2,
                                                                                                     dropout_ap=0.2)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
model.load_state_dict(torch.load(
    '/data/user/zhaoyimeng/ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_LSVQ_plcc_rank_NR_v333_epoch_44_SRCC_0.887701.pth'))
print(model)

for name, param in model.named_parameters():
    print(name, param)


layer_name = 'feature_extraction.ln_post'  # 假设我们要获取第12个 Transformer Block
target_layer = model.module.feature_extraction.ln_post
hook = target_layer.register_forward_hook(hook_fn)

# 输入测试数据，执行一次前向传播
x_dummy = torch.randn(1, 8, 3, 224, 224)  # Dummy data
# output = model(x_dummy, None, None, None, None, None)

# 查看提取的特征
print('target_layer features: ', target_layer)
print('Extracted features: ', extracted_features)
print("Extracted features shape:", extracted_features.shape)

# 解除钩子
hook.remove()



