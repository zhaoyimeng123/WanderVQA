import numpy as np

# 读取.npy文件
npy_file = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LSVQ_VideoMAE_feat/backup/24918.npy'

# 加载npy文件
data = np.load(npy_file)

# 打印数据的形状
print(f"文件的形状是: {data.shape}")
