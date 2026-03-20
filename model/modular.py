import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
# from CLIP import clip
from clip import clip
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric
from . import kan, adapter

def get_network(name, pretrained=False):
    network = {
        "VGG16": torchvision.models.vgg16(pretrained=pretrained),
        "VGG16_bn": torchvision.models.vgg16_bn(pretrained=pretrained),
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]

class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, encoder, n_features, anchor_size=32,
                 patch_dim=(2, 2), normalize=True, projection_dim=128):
        super(CONTRIQUE_model, self).__init__()
        self.anchor_size = anchor_size
        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = n_features
        self.patch_dim = patch_dim

        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        # fragment patch
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w', b=b, h=h)
        x = (x.contiguous()
             .view(b * t, c, h // self.anchor_size, self.anchor_size, w // self.anchor_size, self.anchor_size)
             .permute(0, 2, 4, 1, 3, 5)
             .contiguous()
             .view(b * t * h // self.anchor_size * (w // self.anchor_size), c, self.anchor_size, self.anchor_size))
        num_grid = h // self.anchor_size * (w // self.anchor_size)
        h = self.encoder(x)

        h = h.view(-1, self.n_features)

        if self.normalize:
            h = nn.functional.normalize(h, dim=1)

        # global projections
        z = self.projector(h)
        z = z.view(b, t, num_grid, -1)

        return z

def distortion_contrastive(distortion_feature):
    b, t, num_grid, _ = distortion_feature.shape
    distortion_feature = distortion_feature.view(b * t * num_grid, -1)
    # distortion class
    dist_label = torch.eye(b)
    # expend to fragment

    dist_labels = dist_label.repeat(1, t * num_grid).view(b * t * num_grid, -1)
    # dist_label = dist_label.to(distortion_feature.device)
    # calculate similaroty and divide by temperature parameter
    z = nn.functional.normalize(distortion_feature, p=2, dim=1)
    sim = torch.mm(z, z.T) / 0.1
    # dist_labels=dist_labels.cpu()
    positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
    positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)

    N = b * t * num_grid
    zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)

    # calculate normalized cross entropy value
    positive_sum = torch.sum(positive_mask, dim=1)
    denominator = torch.sum(torch.exp(sim) * zero_diag, dim=1)
    loss = torch.mean(torch.log(denominator) - (torch.sum(sim * positive_mask, dim=1) / positive_sum))
    loss = loss.unsqueeze(0)
    return loss

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ViTbCLIP_SpatialTemporal_modular_dropout(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_modular_dropout, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.spatial_rec = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)  #
        self.temporal_rec = self.temporal_rectifier(256 * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

        self.sr = sr
        self.tr = tr

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def spatial_rectifier(self, in_channels, dropout_sp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)
        # x = self.avgpool(x)
        x = self.base_quality(x)

        x = x.view(x_size[0], -1)
        x = torch.mean(x, dim=1).unsqueeze(1)

        if self.sr:
            lp_size = lp.shape
            lp = lp.view(lp_size[0], -1)
            spatial = self.spatial_rec(lp)
            s_ones = torch.ones_like(x)  #
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]  # torch.chunk 将 spatial 沿着 dim=1 切分为两个张量，取第一个赋给 sa
            sa = torch.add(sa, s_ones)  # 防止乘法中的零因子问题
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(x)
            sb = torch.zeros_like(x)
        qs = torch.add(torch.mul(torch.abs(sa), x), sb).squeeze(1)

        if self.tr:
            x_3D_features_size = x_3D_features.shape
            x_3D_features = x_3D_features.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(x_3D_features)
            t_ones = torch.ones_like(x)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones)  #
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(x)
            tb = torch.zeros_like(x)
        qt = torch.add(torch.mul(torch.abs(ta), x), tb).squeeze(1)

        if self.sr and self.tr:
            modular_a = torch.sqrt(torch.abs(torch.mul(sa, ta)))
            modular_b = torch.div(torch.add(sb, tb), 2)
            qst = torch.add(torch.mul(modular_a, x), modular_b).squeeze(1)  # eq.9
        elif self.sr:
            qst = qs
        elif self.tr:
            qst = qt
        else:
            qst = x.squeeze(1)

        return x.squeeze(1), qs, qt, qst

class ViTbCLIP_SpatialTemporal_modular_dropout_2noise(torch.nn.Module):
    def __init__(self, feat_len=8, dr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_modular_dropout_2noise, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_dist = self.base_quality_regression(4096, 1024, 768)

        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)
        self.cross_atten = CrossAttention(64)
        self.down_rec = self.base_quality_regression(768, 64, 64)
        self.up_rec = self.base_quality_regression(64, 768, 768)

        encoder = get_network('resnet50', pretrained=False)  # todo
        model = CONTRIQUE_model(encoder, 2048)
        self.distortion_tool = model
        self.distortion_tool.load_state_dict(
            torch.load('/data/user/zhaoyimeng/KVQ2024/pretrained_checkpoint/CONTRIQUE_checkpoint25.tar'))
        self.dist_adapter = nn.Sequential(
            nn.Linear(128, 128 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 4, 128),
            nn.ReLU(inplace=True),
        )
        for param in self.distortion_tool.parameters():
            param.requires_grad = False

        self.dist_unify = self.base_quality_regression(49*128, 4096, 768)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, dist):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape

        # x_dist = x
        # x_dist = self.distortion_tool(x_dist)  # b,feat_len,49,128
        #
        # dist_token_ori = 0.8 * self.dist_adapter(x_dist) + 0.2 * x_dist
        # dis_contrast_loss = distortion_contrastive(dist_token_ori)
        dis_contrast_loss = 0

        # x_dist = x_dist.view(x_size[0], x_size[1], -1)
        # x_dist = self.dist_unify(x_dist)
        # x_dist = x_dist.view(-1, 768)
        # x_dist = x_dist.unsqueeze(1).repeat(1, 196, 1)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x_dist_cls_token = nn.Parameter(torch.zeros(x_size[0] * x_size[1], 1, 768)).to(device)
        # x_dist = torch.cat([x_dist_cls_token, x_dist], dim=1)

        dist = self.unify_dist(dist)
        dist_size = dist.shape  # b,feat_len, 4096
        dist = dist.view(-1, dist_size[2])
        dist = dist.unsqueeze(1).repeat(1, 197, 1)

        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_dist = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_dist = x_loda_dist.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_dist = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_dist.shape[0], 1, 1)
        x_loda_dist = torch.cat([cls_token_dist, x_loda_dist], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_dist = x_loda_dist + self.feature_extraction.positional_embedding
        x_loda_dist = self.feature_extraction.ln_pre(x_loda_dist)  # (b,197,768)
        # 清理不再需要的张量，节省显存
        del cls_token_dist
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_dist = self.down_rec(x_loda_dist).detach()
                dist_down = self.down_rec(dist).detach()
                x_down_dist = x_loda_down_dist + self.cross_atten(x_loda_down_dist, dist_down)
                x_up_dist = self.up_rec(x_down_dist)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_dist = x_loda_dist + x_up_dist * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_dist = block(x_loda_dist)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_dist = self.feature_extraction.ln_post(x_loda_dist[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_dist = x_loda_dist @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_dist = x_loda_dist.view(x_size[0], x_size[1], 512)
        # ###############################
        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_dist = x_loda_dist + x
        x_loda_dist = self.base_quality(x_loda_dist)
        x_loda_dist = torch.mean(x_loda_dist, dim=1).squeeze(1)
        # return x_loda_dist, dis_contrast_loss
        return x_loda_dist


class ViTbCLIP_SpatialTemporalDistortion_modular_dropout(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_modular_dropout, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp
        self.dropout_dp = dropout_dp

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.spatial_rec = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)  #
        self.temporal_rec = self.temporal_rectifier(256 * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.distortion_rec = self.distortion_rectifier(4096 * self.feat_len, self.dropout_dp)

        self.sr = sr
        self.tr = tr
        self.dr = dr

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def spatial_rectifier(self, in_channels, dropout_sp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def distortion_rectifier(self, in_channels, dropout_dp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_dp),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)
        # x = self.avgpool(x)
        x = self.base_quality(x)

        x = x.view(x_size[0], -1)
        x = torch.mean(x, dim=1).unsqueeze(1)

        if self.sr:
            lp_size = lp.shape
            lp = lp.view(lp_size[0], -1)
            spatial = self.spatial_rec(lp)
            s_ones = torch.ones_like(x)
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]
            sa = torch.add(sa, s_ones)  #
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(x)
            sb = torch.zeros_like(x)
        qs = torch.add(torch.mul(torch.abs(sa), x), sb).squeeze(1)

        if self.tr:
            x_3D_features_size = x_3D_features.shape
            x_3D_features = x_3D_features.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(x_3D_features)
            t_ones = torch.ones_like(x)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones)  #
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(x)
            tb = torch.zeros_like(x)
        qt = torch.add(torch.mul(torch.abs(ta), x), tb).squeeze(1)

        if self.dr:
            dist_size = dist.shape
            dist = dist.view(dist_size[0], -1)
            distortion = self.distortion_rec(dist)
            d_ones = torch.ones_like(x)
            # ax+b
            da = torch.chunk(distortion, 2, dim=1)[0]
            da = torch.add(da, d_ones)  #
            db = torch.chunk(distortion, 2, dim=1)[1]
        else:
            da = torch.ones_like(x)
            db = torch.zeros_like(x)
        qd = torch.add(torch.mul(torch.abs(da), x), db).squeeze(1)


        if self.sr and self.tr and self.dr:
            modular_a = torch.pow(torch.abs(torch.mul(torch.mul(sa, ta), da)), 1/3)
            modular_b = torch.div(torch.add(torch.add(sb, tb), db), 3)
            qstd = torch.add(torch.mul(modular_a, x), modular_b).squeeze(1)  # eq.(9)
        elif self.sr:
            qstd = qs
        elif self.tr:
            qstd = qt
        elif self.dr:
            qstd = qd
        else:
            qstd = x.squeeze(1)

        return x.squeeze(1), qs, qt, qd, qstd


class ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp
        self.dropout_dp = dropout_dp
        self.dropout_ap = dropout_ap

        self.sr = sr
        self.tr = tr
        self.dr = dr
        self.ar = ar

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.spatial_rec = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)  #
        self.temporal_rec = self.temporal_rectifier(256 * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.distortion_rec = self.distortion_rectifier(4096 * self.feat_len, self.dropout_dp)
        self.aesthetic_rec = self.aesthetic_rectifier(784, self.dropout_ap)


    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def spatial_rectifier(self, in_channels, dropout_sp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def distortion_rectifier(self, in_channels, dropout_dp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_dp),
        )
        return regression_block

    def aesthetic_rectifier(self, in_channels, dropout_ap):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_ap),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)
        # x = self.avgpool(x)
        x = self.base_quality(x)

        x = x.view(x_size[0], -1)
        x = torch.mean(x, dim=1).unsqueeze(1)

        if self.sr:
            lp_size = lp.shape
            lp = lp.view(lp_size[0], -1)
            spatial = self.spatial_rec(lp)
            s_ones = torch.ones_like(x)
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]
            sa = torch.add(sa, s_ones)  #
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(x)
            sb = torch.zeros_like(x)
        qs = torch.add(torch.mul(torch.abs(sa), x), sb).squeeze(1)

        if self.tr:
            x_3D_features_size = x_3D_features.shape
            x_3D_features = x_3D_features.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(x_3D_features)
            t_ones = torch.ones_like(x)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones)  #
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(x)
            tb = torch.zeros_like(x)
        qt = torch.add(torch.mul(torch.abs(ta), x), tb).squeeze(1)

        if self.dr:
            dist_size = dist.shape
            dist = dist.view(dist_size[0], -1)
            distortion = self.distortion_rec(dist)
            d_ones = torch.ones_like(x)
            # ax+b
            da = torch.chunk(distortion, 2, dim=1)[0]
            da = torch.add(da, d_ones)  #
            db = torch.chunk(distortion, 2, dim=1)[1]
        else:
            da = torch.ones_like(x)
            db = torch.zeros_like(x)
        qd = torch.add(torch.mul(torch.abs(da), x), db).squeeze(1)

        if self.ar:
            aes_size = aes.shape
            aes = aes.view(aes_size[0], -1)
            aesthetic = self.aesthetic_rec(aes)
            a_ones = torch.ones_like(x)
            # ax+b
            aa = torch.chunk(aesthetic, 2, dim=1)[0]
            aa = torch.add(aa, a_ones)  #
            ab = torch.chunk(aesthetic, 2, dim=1)[1]
        else:
            aa = torch.ones_like(x)
            ab = torch.zeros_like(x)
        qa = torch.add(torch.mul(torch.abs(aa), x), ab).squeeze(1)


        if self.sr and self.tr and self.dr and self.ar:
            modular_a = torch.pow(torch.abs(torch.mul(torch.mul(torch.mul(sa, ta), da), aa)), 1/4)
            modular_b = torch.div(torch.add(torch.add(torch.add(sb, tb), db), ab), 4)
            qstda = torch.add(torch.mul(modular_a, x), modular_b).squeeze(1)  # eq.(9)
        elif self.sr:
            qstda = qs
        elif self.tr:
            qstda = qt
        elif self.dr:
            qstda = qd
        elif self.ar:
            qstda = qa
        else:
            qstda = x.squeeze(1)

        return x.squeeze(1), qs, qt, qd, qa, qstda


class ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        dist = self.unify_distortion_rec(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)
        dist_aes = 0.572 * dist + 0.428 * aes

        lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        Q = lp
        K = dist_aes
        V = x_3D_features
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x + cross_attention_output
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v2(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v2, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.weight1 = nn.Parameter(torch.tensor(0.2))  # 初始化为0.2
        self.weight2 = nn.Parameter(torch.tensor(0.2))
        self.weight3 = nn.Parameter(torch.tensor(0.2))
        self.weight4 = nn.Parameter(torch.tensor(0.2))
        self.weight5 = nn.Parameter(torch.tensor(0.2))

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        # cross_attention_v2
        dist = self.unify_distortion_rec(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)

        lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        Q_lp = lp
        Q_dist = dist
        Q_aes = aes
        Q_x_3D = x_3D_features
        K = x
        V = x
        d_k = K.size(-1)

        scores_lp = torch.matmul(Q_lp, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights_lp = F.softmax(scores_lp, dim=-1)
        cross_attention_output_lp = torch.matmul(attention_weights_lp, V)

        scores_dist = torch.matmul(Q_dist, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights_dist = F.softmax(scores_dist, dim=-1)
        cross_attention_output_dist = torch.matmul(attention_weights_dist, V)

        scores_aes = torch.matmul(Q_aes, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights_aes = F.softmax(scores_aes, dim=-1)
        cross_attention_output_aes = torch.matmul(attention_weights_aes, V)

        scores_x_3D = torch.matmul(Q_x_3D, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights_x_3D = F.softmax(scores_x_3D, dim=-1)
        cross_attention_output_x_3D = torch.matmul(attention_weights_x_3D, V)

        x = self.weight1*cross_attention_output_lp + self.weight2*cross_attention_output_dist + self.weight3*cross_attention_output_aes + self.weight4*cross_attention_output_x_3D + self.weight5*x

        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_modular_dropout_cross_attention_v3_gat, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.conv1 = GCNConv(2560, 1024)
        self.conv2 = GCNConv(1024, 1024)
        self.fc = torch.nn.Linear(1024, 512)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        # cross_attention_v3
        dist = self.unify_distortion_rec(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)

        lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # 1. 特征拼接 (b, 8, 512) -> (b, 8, 2560)
        fused_features = torch.cat([x, dist, aes, lp, x_3D_features], dim=-1)  # (batch_size, 8, 5*512)

        # 2. 将 (b, 8, 2560) 转换为 (b * 8, 2560)
        fused_features = fused_features.view(x_size[0] * self.feat_len, -1)  # (batch_size * num_nodes, 2560)
        fused_features.to(device)


        # 3. 定义图的连接关系（边），假设是全连接图
        # 每个节点与其他节点都有连接（全连接），即任意两个节点之间有边
        edge_index = torch.randint(0, self.feat_len, (2, self.feat_len), dtype=torch.long).to(device)

        # 4. 创建批次索引，每个样本的8个节点属于一个batch
        batch = torch.arange(x_size[0]).repeat_interleave(self.feat_len).to(device)  # (batch_size * num_nodes)

        fused_features = self.conv1(fused_features, edge_index)
        fused_features = F.relu(fused_features)
        fused_features = self.conv2(fused_features, edge_index)
        fused_features = global_mean_pool(fused_features, batch)
        fused_features = self.fc(fused_features)


        # 输出形状 (batch_size, 512)，可以用于后续任务，如分类或回归
        # print(fused_features.shape)  # 输出 (batch_size, 512)

        x = self.base_quality(fused_features)
        x = x.squeeze(1)

        return x


class ViTbCLIP_SpatialTemporalDistortion_QCN_modular_dropout_v4(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_QCN_modular_dropout_v4, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")


        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(18 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, QCN_feat, dist):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)
        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        dist = self.unify_distortion_rec(dist)


        QCN_feat = self.unify_spatial_rec(QCN_feat)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        Q = QCN_feat
        K = dist
        V = x
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x_3D_features + cross_attention_output
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortion_LoDA_modular_dropout_v5(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_LoDA_modular_dropout_v5, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")


        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(18 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 768)

        self.cross_atten_1 = CrossAttention(64)

        self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)

        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, QCN_feat, dist):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)

        dist_size = dist.shape
        dist = dist.view(-1, dist_size[2])
        dist = self.unify_distortion_rec(dist)
        dist = dist.unsqueeze(1).repeat(1, 197, 1)


        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        # x = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        # x = x.flatten(2).transpose(1, 2)  # Flatten and Transpose
        #
        # # Step 2: Add CLS token
        # cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.cat([cls_token, x], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        #
        # # Step 3: Add Positional Embedding and LayerNorm
        # x = x + self.feature_extraction.positional_embedding
        # x = self.feature_extraction.ln_pre(x)
        #
        # # Step 4: Process through each Transformer Block
        # for i, block in enumerate(self.feature_extraction.transformer.resblocks):
        #     x = block(x)
        #     # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度
        #
        #     # if i % 3 == 0:
        #     #     # 在这里你可以操作每个 Block 的输出，例如：
        #     #     # x = x + 1  # 示例操作，实际中可以根据需求进行操作
        #     #     x_down = self.down_proj(x)
        #     #     dist_down = self.down_proj(dist)
        #     #     x_down = x_down + self.cross_atten_1(x_down, dist_down)
        #     #     x_up = self.up_proj(x_down)
        #     #     x = x + x_up * self.scale_factor
        #
        #
        #
        # # Step 5: Final LayerNorm and Class Token output
        # x = self.feature_extraction.ln_post(x[:, 0, :])
        #
        # if self.feature_extraction.proj is not None:
        #     x = x @ self.feature_extraction.proj  # Linear projection if it exists
        # ###############################


        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        Q = x_3D_features
        K = x
        V = x
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x_3D_features + cross_attention_output
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
                    patch = self.upsample(patch)  # 上采样到 (32,32)
                    # 深度卷积
                    patch = self.depth_conv(patch)
                    patch = self.point_conv(patch)

                    new_patches.append(patch)

                # 4. 重新组合为 224×224
                new_patches = torch.stack(new_patches, dim=1).view(B, 7, 7, -1, 32, 32)
                new_patches = new_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
                frame_output = new_patches.view(B, -1, 224, 224)  # (B, C, 224, 224)



                outputs.append(frame_output)

            return torch.stack(outputs, dim=1)  # (B, F, C, 224, 224)


class ViTbCLIP_SpatialTemporalDistortionAes_loda_modular_dropout_cross_attention_v6(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_loda_modular_dropout_cross_attention_v6, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.unify_distortion_rec_loda = nn.Linear(4096, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)
        self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

                if i % 3 == 0:
                    # 在这里你可以操作每个 Block 的输出，例如：
                    # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                    x_loda_down = self.down_proj(x_loda).detach()
                    dist_loda_down = self.down_proj(dist_loda).detach()
                    x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                    x_up = self.up_proj(x_down)
                    x_loda = x_loda + x_up * self.scale_factor
                    # 清理不再需要的中间变量
                    del x_loda_down, dist_loda_down, x_down, x_up
                    torch.cuda.empty_cache()

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        dist = self.unify_distortion_rec(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)
        dist_aes = 0.572 * dist + 0.428 * aes

        lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        Q = lp
        K = dist_aes
        V = x_3D_features
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x + cross_attention_output + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x



class ViTbCLIP_SpatialTemporalDistortionAes_loda_kan_modular_dropout_cross_attention_v7(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_loda_kan_modular_dropout_cross_attention_v7, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = kan.KAN([512, 128, 1])

        self.unify_spatial_rec = kan.KAN([5*256, 512])
        self.unify_temporal_rec = kan.KAN([256, 512])
        self.unify_distortion_rec = kan.KAN([4096, 512])
        self.unify_aesthetic_rec = kan.KAN([784, 512])

        self.unify_distortion_rec_loda = kan.KAN([4096, 768])
        self.cross_atten_1 = CrossAttention(64)
        self.down_proj = kan.KAN([768, 64, 64])
        self.up_proj = kan.KAN([64, 768, 768])
        self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token
        torch.cuda.empty_cache()

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

            if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                # 在这里操作 Block 的输出，处理前 detaching 避免累积计算图
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                x_loda = x_loda + x_up * self.scale_factor

                # 清理不再需要的中间变量
                del x_loda_down, dist_loda_down, x_down, x_up
                torch.cuda.empty_cache()

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        dist = self.unify_distortion_rec(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)
        dist_aes = 0.572 * dist + 0.428 * aes

        lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        Q = lp
        K = dist_aes
        V = x_3D_features
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x + cross_attention_output + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortionAes_loda_modular_dropout_cross_attention_v8_abla(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortionAes_loda_modular_dropout_cross_attention_v8_abla, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.unify_distortion_rec_loda = nn.Linear(4096, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.down_proj = self.base_quality_regression(512, 64, 64)
        self.up_proj = self.base_quality_regression(64, 512, 512)
        self.scale_factor = nn.Parameter(torch.randn(8, 512) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # dist_size = dist.shape
        # dist_loda = dist.view(-1, dist_size[2])
        # dist_loda = self.unify_distortion_rec_loda(dist_loda)
        # dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # # Step 1: Patch Embedding
        # x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        # x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose
        #
        # # Step 2: Add CLS token
        # cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        # x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        #
        # # Step 3: Add Positional Embedding and LayerNorm
        # x_loda = x_loda + self.feature_extraction.positional_embedding
        # x_loda = self.feature_extraction.ln_pre(x_loda)
        #
        # # 清理不再需要的张量，节省显存
        # del cls_token
        #
        # # Step 4: Process through each Transformer Block
        # for i, block in enumerate(self.feature_extraction.transformer.resblocks):
        #     # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
        #     with torch.no_grad():
        #         x_loda = block(x_loda)
        #         # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度
        #
        #         if i % 3 == 0:
        #             # 在这里你可以操作每个 Block 的输出，例如：
        #             # x = x + 1  # 示例操作，实际中可以根据需求进行操作
        #             x_loda_down = self.down_proj(x_loda).detach()
        #             dist_loda_down = self.down_proj(dist_loda).detach()
        #             x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
        #             x_up = self.up_proj(x_down)
        #             x_loda = x_loda + x_up * self.scale_factor
        #             # 清理不再需要的中间变量
        #             del x_loda_down, dist_loda_down, x_down, x_up
        #             torch.cuda.empty_cache()
        #
        # # Step 5: Final LayerNorm and Class Token output
        # x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])
        #
        # if self.feature_extraction.proj is not None:
        #     x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        # x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)

        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        dist = self.unify_distortion_rec(dist)
        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        # dist_aes = 0.572 * dist + 0.428 * aes

        # lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        # loda cross atten
        x_down = self.down_proj(x)
        dist_down = self.down_proj(dist)
        x_down = x_down + self.cross_atten_1(x_down, dist_down)
        x_up = self.up_proj(x_down)
        x = x + x_up * self.scale_factor


        Q = x
        K = x_3D_features
        V = x_3D_features
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        cross_attention_output = torch.matmul(attention_weights, V)
        x = x + cross_attention_output
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_v9(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_v9, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.unify_distortion_rec_loda = nn.Linear(4096, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)
        self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                x_loda = x_loda + x_up * self.scale_factor
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)
        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        #
        # dist = self.unify_distortion_rec(dist)
        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        # dist_aes = 0.572 * dist + 0.428 * aes

        # lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        # Q = x_loda
        # K = x_3D_features
        # V = x_3D_features
        # d_k = Q.size(-1)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # attention_weights = F.softmax(scores, dim=-1)
        # cross_attention_output = torch.matmul(attention_weights, V)
        x_loda = self.cross_atten_2(x_loda, x_3D_features)
        x = x + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        # self.unify_distortion_rec = self.base_quality_regression(4096, 1024, 512)  # todo v11 调参

        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = nn.Linear(4096, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)
        # self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。
        self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。  todo 应该初始化block个吧

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)
        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        #
        # dist = self.unify_distortion_rec(dist)
        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        # dist_aes = 0.572 * dist + 0.428 * aes

        # lp = self.unify_spatial_rec(lp)
        # x_3D_features = self.unify_temporal_rec(x_3D_features)

        x_videomae_features = self.unify_videomae_rec(videomae)


        # cross_attention
        # Q = x_loda
        # K = x_3D_features
        # V = x_3D_features
        # d_k = Q.size(-1)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # attention_weights = F.softmax(scores, dim=-1)
        # cross_attention_output = torch.matmul(attention_weights, V)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        # 冻结 CLIP 参数
        # for param in clip_vit_b_pretrained_features.parameters():
        #     param.requires_grad = False

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_w(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_w, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")

        self.adapter_sd = adapter.Adapter(2, 64, [64, 64, 64], 64, 8, 12)
        self.adapter_sdt = adapter.Adapter(2, 512, [512, 512, 512], 512, 8, 12)


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        # 冻结 CLIP 参数
        # for param in clip_vit_b_pretrained_features.parameters():
        #     param.requires_grad = False

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()

                '''wander'''
                # x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_down = x_loda_down + self.adapter_sd([x_loda_down, dist_loda_down])


                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)


        '''wander'''
        # x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x_loda = self.adapter_sdt([x_loda, x_videomae_features])


        x = x + x_loda
        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class Xgc_weight_concat(torch.nn.Module):
    def __init__(self, feat_len=8):
        super(Xgc_weight_concat, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")
        # ViT_L_14, _ = clip.load("ViT-L/14")

        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.unify_videomae = self.base_quality_regression(1408, 768, 512)
        self.unify_lp = self.base_quality_regression(2560, 1024, 512)
        self.unify_lp_2 = self.base_quality_regression(2560, 1024, 512)
        self.unify_dist = self.base_quality_regression(4096, 1024, 512)
        self.unify_aes = self.base_quality_regression(784, 768, 512)

        self.regression_aes = self.base_quality_regression(6*512, 1024, 512)
        self.regression_aes_score = self.base_quality_regression(512, 128, 1)

        self.regression_dist = self.base_quality_regression(6 * 512, 1024, 512)
        self.regression_dist_score = self.base_quality_regression(512, 128, 1)

        self.regression_lp_1 = self.base_quality_regression(6 * 512, 1024, 512)
        self.regression_lp_1_score = self.base_quality_regression(512, 128, 1)

        self.regression_lp_2 = self.base_quality_regression(6 * 512, 1024, 512)
        self.regression_lp_2_score = self.base_quality_regression(512, 128, 1)

        self.regression_videomae = self.base_quality_regression(6 * 512, 1024, 512)
        self.regression_videomae_score = self.base_quality_regression(512, 128, 1)

        self.regression_total = self.base_quality_regression(6 * 512, 1024, 512)
        self.regression_total_score = self.base_quality_regression(512, 128, 1)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * feat_len x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)

        # 1. 美学作为颜色
        # aes = (b, 784)
        aes = aes.repeat(1, self.feat_len, 1)
        aes = self.unify_aes(aes)

        # 2.失真作为噪声 todo
        dist = self.unify_dist(dist)

        # 3.lp作为伪影
        lp_1 = self.unify_lp(lp)

        # 4.lp_2作为模糊
        lp_2 = self.unify_lp_2(lp)

        # 5.时间 todo
        videomae_size = videomae.shape  # b,feat_len,1408
        videomae = self.unify_videomae(videomae)

        # 1. 美学作为颜色
        aes_score = torch.cat([0.5 * aes, 0.1 * dist, 0.1 * lp_1, 0.1 * lp_2, 0.1 * videomae, 0.1 * x], dim=2)
        aes_score = self.regression_aes(aes_score)
        aes_score = self.regression_aes_score(aes_score)
        aes_score = torch.mean(aes_score, dim=1)

        # 2.失真作为噪声
        dist_score = torch.cat([0.1 * aes, 0.5 * dist, 0.1 * lp_1, 0.1 * lp_2, 0.1 * videomae, 0.1 * x], dim=2)
        dist_score = self.regression_dist(dist_score)
        dist_score = self.regression_dist_score(dist_score)
        dist_score = torch.mean(dist_score, dim=1)

        # 3.lp作为伪影
        lp_1_score = torch.cat([0.1 * aes, 0.1 * dist, 0.5 * lp_1, 0.1 * lp_2, 0.1 * videomae, 0.1 * x], dim=2)
        lp_1_score = self.regression_lp_1(lp_1_score)
        lp_1_score = self.regression_lp_1_score(lp_1_score)
        lp_1_score = torch.mean(lp_1_score, dim=1)

        # 4.lp_2作为模糊
        lp_2_score = torch.cat([0.1 * aes, 0.1 * dist, 0.1 * lp_1, 0.5 * lp_2, 0.1 * videomae, 0.1 * x], dim=2)
        lp_2_score = self.regression_lp_2(lp_2_score)
        lp_2_score = self.regression_lp_2_score(lp_2_score)
        lp_2_score = torch.mean(lp_2_score, dim=1)

        # 5.时间
        videomae_score = torch.cat([0.1 * aes, 0.1 * dist, 0.1 * lp_1, 0.1 * lp_2, 0.5 * videomae, 0.1 * x], dim=2)
        videomae_score = self.regression_videomae(videomae_score)
        videomae_score = self.regression_videomae_score(videomae_score)
        videomae_score = torch.mean(videomae_score, dim=1)

        # 6.整体
        total_score = torch.cat([0.1 * aes, 0.1 * dist, 0.1 * lp_1, 0.1 * lp_2, 0.1 * videomae, 0.5 * x], dim=2)
        total_score = self.regression_total(total_score)
        total_score = self.regression_total_score(total_score)
        total_score = torch.mean(total_score, dim=1)

        result = torch.cat([aes_score, dist_score, lp_1_score, lp_2_score, videomae_score, total_score], dim=1)
        return result




        """
        # 5.时间 todo
        # videomae_size = videomae.shape  # b,feat_len,1408
        # videomae = self.unify_videomae(videomae)
        x_3D_features = self.unify_slowfast(x_3D_features)
        x_3D_features_size = x_3D_features.shape  # b,feat_len,2048+256=2304
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        x_3D_features = x_3D_features.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_3D = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_3D = x_loda_3D.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_3D = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_3D.shape[0], 1, 1)
        x_loda_3D = torch.cat([cls_token_3D, x_loda_3D], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_3D = x_loda_3D + self.feature_extraction.positional_embedding
        x_loda_3D = self.feature_extraction.ln_pre(x_loda_3D)
        # 清理不再需要的张量，节省显存
        del cls_token_3D
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_3D = self.down_rec(x_loda_3D).detach()
                x_3D_features_down = self.down_rec(x_3D_features).detach()
                x_down_3D = x_loda_down_3D + self.cross_atten(x_3D_features_down, x_loda_down_3D)
                x_up_3D = self.up_rec(x_down_3D)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_3D = x_loda_3D + x_up_3D * self.scale_factor[0, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_3D = block(x_loda_3D)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_3D = self.feature_extraction.ln_post(x_loda_3D[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_3D = x_loda_3D @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_3D = x_loda_3D.view(x_size[0], x_size[1], 512)
        # ###############################
        x_3D = self.feature_extraction(x)
        x_3D = x_3D.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_3D = x_3D + x_loda_3D
        x_3D_features_with_x = self.base_quality(x_loda_3D)
        x_3D_features_with_x = torch.mean(x_3D_features_with_x, dim=1).squeeze(1)
        return x_3D_features_with_x

        # videomae
        videomae = self.unify_videomae(videomae)
        videomae_size = videomae.shape  # b,feat_len,2048+256=2304
        videomae = videomae.view(-1, videomae_size[2])
        videomae = videomae.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_videomae = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_videomae = x_loda_videomae.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_videomae = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_videomae.shape[0], 1, 1)
        x_loda_videomae = torch.cat([cls_token_videomae, x_loda_videomae], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_videomae = x_loda_videomae + self.feature_extraction.positional_embedding
        x_loda_videomae = self.feature_extraction.ln_pre(x_loda_videomae)
        # 清理不再需要的张量，节省显存
        del cls_token_videomae
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_videomae = self.down_rec(x_loda_videomae).detach()
                videomae_down = self.down_rec(videomae).detach()
                x_down_videomae = x_loda_down_videomae + self.cross_atten(videomae_down, x_loda_down_videomae)
                x_up_videomae = self.up_rec(x_down_videomae)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_videomae = x_loda_videomae + x_up_videomae * self.scale_factor[0, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_videomae = block(x_loda_videomae)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_videomae = self.feature_extraction.ln_post(x_loda_videomae[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_videomae = x_loda_videomae @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_videomae = x_loda_videomae.view(x_size[0], x_size[1], 512)
        # ###############################
        x_videomae = self.feature_extraction(x)
        x_videomae = x_videomae.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae = x_videomae + x_loda_videomae
        x_videomae_with_x = self.base_quality(x_videomae)
        x_videomae_with_x = torch.mean(x_videomae_with_x, dim=1).squeeze(1)
        return x_videomae_with_x




        # 3.lp作为伪影、4.模糊
        lp = self.unify_lp(lp)
        lp_size = lp.shape  # b,feat_len,5*512=2560
        lp = lp.view(-1, lp_size[2])
        lp = lp.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_lp = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_lp = x_loda_lp.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_lp = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_lp.shape[0], 1, 1)
        x_loda_lp = torch.cat([cls_token_lp, x_loda_lp], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_lp = x_loda_lp + self.feature_extraction.positional_embedding
        x_loda_lp = self.feature_extraction.ln_pre(x_loda_lp)  # (b,257,1024)
        # 清理不再需要的张量，节省显存
        del cls_token_lp
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            # with torch.no_grad():
            if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_lp = self.down_rec(x_loda_lp).detach()
                lp_down = self.down_rec(lp).detach()
                x_down_lp = x_loda_down_lp + self.cross_atten(x_loda_down_lp, lp_down)
                x_up_lp = self.up_rec(x_down_lp)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_lp = x_loda_lp + x_up_lp * self.scale_factor[1, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_lp = block(x_loda_lp)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_lp = self.feature_extraction.ln_post(x_loda_lp[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_lp = x_loda_lp @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_lp = x_loda_lp.view(x_size[0], x_size[1], 512)
        # ###############################
        x_lp = self.feature_extraction(x)
        x_lp = x_lp.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_lp = x_lp + x_loda_lp
        lp_with_x = self.base_quality(x_loda_lp)
        lp_with_x = torch.mean(lp_with_x, dim=1).squeeze(1)

        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_lp_2 = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_lp_2 = x_loda_lp_2.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_lp_2 = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_lp_2.shape[0], 1, 1)
        x_loda_lp_2 = torch.cat([cls_token_lp_2, x_loda_lp_2], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_lp_2 = x_loda_lp_2 + self.feature_extraction.positional_embedding
        x_loda_lp_2 = self.feature_extraction.ln_pre(x_loda_lp_2)  # (b,257,1024)
        # 清理不再需要的张量，节省显存
        del cls_token_lp_2
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_lp_2 = self.down_rec(x_loda_lp_2).detach()
                lp_down_2 = self.down_rec(lp).detach()
                x_down_lp_2 = x_loda_down_lp_2 + self.cross_atten(x_loda_down_lp_2, lp_down_2)
                x_up_lp_2 = self.up_rec(x_down_lp_2)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_lp_2 = x_loda_lp_2 + x_up_lp_2 * self.scale_factor[2, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_lp_2 = block(x_loda_lp_2)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_lp_2 = self.feature_extraction.ln_post(x_loda_lp_2[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_lp_2 = x_loda_lp_2 @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_lp_2 = x_loda_lp_2.view(x_size[0], x_size[1], 512)
        # ###############################
        x_lp_2 = self.feature_extraction(x)
        x_lp_2 = x_lp_2.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_lp = x_lp_2 + x_loda_lp_2
        lp_2_with_x = self.base_quality(x_loda_lp_2)
        lp_2_with_x = torch.mean(lp_2_with_x, dim=1).squeeze(1)


        # 2.失真作为噪声 todo
        dist = self.unify_dist(dist)
        dist_size = dist.shape  # b,feat_len, 4096
        dist = dist.view(-1, dist_size[2])
        dist = dist.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_dist = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_dist = x_loda_dist.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_dist = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_dist.shape[0], 1, 1)
        x_loda_dist = torch.cat([cls_token_dist, x_loda_dist], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_dist = x_loda_dist + self.feature_extraction.positional_embedding
        x_loda_dist = self.feature_extraction.ln_pre(x_loda_dist)  # (b,257,1024)
        # 清理不再需要的张量，节省显存
        del cls_token_dist
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_dist = self.down_rec(x_loda_dist).detach()
                dist_down = self.down_rec(dist).detach()
                x_down_dist = x_loda_down_dist + self.cross_atten(x_loda_down_dist, dist_down)
                x_up_dist = self.up_rec(x_down_dist)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_dist = x_loda_dist + x_up_dist * self.scale_factor[3, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_dist = block(x_loda_dist)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_dist = self.feature_extraction.ln_post(x_loda_dist[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_dist = x_loda_dist @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_dist = x_loda_dist.view(x_size[0], x_size[1], 512)
        # ###############################
        x_dist = self.feature_extraction(x)
        x_dist = x_dist.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_dist = x_loda_dist + x_dist
        dist_with_x = self.base_quality(x_loda_dist)
        dist_with_x = torch.mean(dist_with_x, dim=1).squeeze(1)

        # 1. 美学作为颜色
        # aes = (b, 784)
        aes = aes.repeat(1, self.feat_len, 1)
        aes = self.unify_aes(aes)
        aes_size = aes.shape
        aes = aes.view(-1, aes_size[2])
        aes = aes.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda_aes = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda_aes = x_loda_aes.flatten(2).transpose(1, 2)  # Flatten and Transpose
        # Step 2: Add CLS token
        cls_token_aes = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda_aes.shape[0], 1, 1)
        x_loda_aes = torch.cat([cls_token_aes, x_loda_aes], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        # Step 3: Add Positional Embedding and LayerNorm
        x_loda_aes = x_loda_aes + self.feature_extraction.positional_embedding
        x_loda_aes = self.feature_extraction.ln_pre(x_loda_aes)  # (b,257,1024)
        # 清理不再需要的张量，节省显存
        del cls_token_aes
        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down_aes = self.down_rec(x_loda_aes).detach()
                aes_down = self.down_rec(aes).detach()
                x_down_aes = x_loda_down_aes + self.cross_atten(x_loda_down_aes, aes_down)
                x_up_aes = self.up_rec(x_down_aes)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda_aes = x_loda_aes + x_up_aes * self.scale_factor[4, i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()
                x_loda_aes = block(x_loda_aes)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda_aes = self.feature_extraction.ln_post(x_loda_aes[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda_aes = x_loda_aes @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda_aes = x_loda_aes.view(x_size[0], x_size[1], 512)
        # ###############################
        x_aes = self.feature_extraction(x)
        x_aes = x_aes.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_loda_aes = x_loda_aes + x_aes
        aes_with_x = self.base_quality(x_loda_aes)
        aes_with_x = torch.mean(aes_with_x, dim=1).squeeze(1)


        # 6.整体
        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)



        x_whole = torch.cat([x_loda_aes, x_loda_dist, x_loda_lp, x_loda_lp_2, x_loda_3D, x], dim=2)
        x_whole = self.unify_all(x_whole)
        x_whole = self.base_quality(x_whole)

        result = torch.cat([aes_with_x, dist_with_x, lp_with_x, lp_2_with_x, x_3D_features_with_x, x_whole], dim=2)

        # result = torch.mean(result, dim=1).squeeze(1)

        return result
    """




class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_ablation(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_ablation, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_spatial_rec = nn.Linear(5 * 256, 512)
        self.unify_temporal_rec = nn.Linear(256, 512)
        self.unify_distortion_rec = nn.Linear(4096, 512)
        # self.unify_distortion_rec = self.base_quality_regression(4096, 1024, 512)  #

        self.unify_aesthetic_rec = nn.Linear(784, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.down_dim_ablation = nn.Linear(64*2, 64)
        self.down_dim_ablation_5 = nn.Linear(512*2, 512)

        self.unify_distortion_rec_loda = nn.Linear(4096, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)
        # self.scale_factor = nn.Parameter(torch.randn(197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。
        self.scale_factor = nn.Parameter((torch.randn(197, 768).repeat(12, 1, 1)) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用

            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)

                ######ablation 4.##############################
                # loda_dist_cat = torch.cat([x_loda_down, dist_loda_down], dim=-1)
                # x_down = self.down_dim_ablation(loda_dist_cat)
                ###############################################


                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

            ######ablation 1.##############################
            # with torch.no_grad():
            #     if i % 3 == 0:
            #         x_loda_down = self.down_proj(x_loda).detach()
            #         dist_loda_down = self.down_proj(dist_loda).detach()
            #         x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
            #         x_up = self.up_proj(x_down)
            #         x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
            #         x_loda = block(x_loda)


        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)
        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        #
        # dist = self.unify_distortion_rec(dist)
        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        # dist_aes = 0.572 * dist + 0.428 * aes

        # lp = self.unify_spatial_rec(lp)
        # x_3D_features = self.unify_temporal_rec(x_3D_features)

        x_videomae_features = self.unify_videomae_rec(videomae)



        # x_loda = self.cross_atten_2(x_loda, x_videomae_features)

        ######ablation 5.##############################
        semantic_dist_temporal_cat = torch.cat([x_loda, x_videomae_features], dim=-1)
        x_loda = self.down_dim_ablation_5(semantic_dist_temporal_cat)
        ###############################################



        x = x + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_kan_v11(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_kan_v11, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = kan.KAN([512, 128, 1])

        self.unify_spatial_rec = kan.KAN([5 * 256, 512])
        self.unify_temporal_rec = kan.KAN([256, 512])
        self.unify_distortion_rec = kan.KAN([4096, 512])
        self.unify_aesthetic_rec = kan.KAN([784, 512])

        self.unify_distortion_rec_loda = kan.KAN([4096, 768])
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = kan.KAN([768, 64, 64])
        self.up_proj = kan.KAN([64, 768, 768])
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                x_loda = x_loda + x_up * self.scale_factor[i]
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################


        x = self.feature_extraction(x)
        x = x.view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        #
        # dist = self.unify_distortion_rec(dist)
        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        # dist_aes = 0.572 * dist + 0.428 * aes

        # lp = self.unify_spatial_rec(lp)
        x_3D_features = self.unify_temporal_rec(x_3D_features)

        # cross_attention
        # Q = x_loda
        # K = x_3D_features
        # V = x_3D_features
        # d_k = Q.size(-1)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # attention_weights = F.softmax(scores, dim=-1)
        # cross_attention_output = torch.matmul(attention_weights, V)
        x_loda = self.cross_atten_2(x_loda, x_3D_features)
        x = x + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12(torch.nn.Module):
    def __init__(self, feat_len=8):
        super(ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_cam_v12, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 64, 64)
        self.up_proj = self.base_quality_regression(64, 768, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)  # 其中每个元素是从标准正态分布（均值为 0，标准差为 1）中采样的值。  todo 应该初始化block个吧

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # 拆分
        video_flat, dist_flat, videomae_flat = torch.split(inputs,
                                                           [self.feat_len * 3 * 224 * 224, self.feat_len * 4096, self.feat_len * 1408], dim=1)
        # 还原原始的形状
        x = video_flat.contiguous().view(batch_size, self.feat_len, 3, 224, 224)
        dist = dist_flat.contiguous().view(batch_size, self.feat_len, 4096)
        videomae = videomae_flat.contiguous().view(batch_size, self.feat_len, 1408)

        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.contiguous().view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.contiguous().view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.contiguous().view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

######################   xgc   ############################################
# class LoRALinear(nn.Module):
#     def __init__(self, org_module, rank=8, alpha=16):
#         super().__init__()
#         self.org_module = org_module  # 原始 qkv Linear
#         self.rank = rank
#         self.alpha = alpha
#         self.lora_down = nn.Linear(org_module.in_features, rank, bias=False)
#         self.lora_up = nn.Linear(rank, org_module.out_features, bias=False)
#         self.scaling = alpha / rank
#
#     def forward(self, x):
#         return self.org_module(x) + self.scaling * self.lora_up(self.lora_down(x))  # 残差


class xgc_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(torch.nn.Module):
    def __init__(self, feat_len=8):
        super(xgc_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")

        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 6)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_aesthetic_rec = nn.Linear(784, 768)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_aes = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.down_proj_dist = self.base_quality_regression(768, 128, 64)
        self.down_proj_aes = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, dist, videomae, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)

        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)
        aes = aes.view(-1, 768)
        aes_loda = aes.unsqueeze(1).repeat(1, 197, 1)

        """"""
        # 冻结 CLIP 权重
        for param in self.feature_extraction.parameters():
            param.requires_grad = False

        # 插入 LoRA 到 attention 层
        # 应用LoRA到注意力层
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        lora_config = LoraConfig(
            r=8,  # Low-rank adaptation rank
            lora_alpha=32,  # Scaling factor for the LoRA weights
            target_modules=["attn"],  # Layers to apply LoRA
            inference_mode=False,  # Set to True if you want to enable inference mode
        )
        self.feature_extraction = get_peft_model(self.feature_extraction, lora_config)
        # 准备模型进行训练
        self.feature_extraction = prepare_model_for_kbit_training(self.feature_extraction)

        # 下面是你已有的 pipeline  lora
        # ============== 开始 forward 流程 ================
        # 1. Patch Embedding
        x_loda = self.feature_extraction.conv1(x)
        x_loda = x_loda.flatten(2).transpose(1, 2)
        # 2. Add CLS Token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)
        # 3. Positional Embedding & LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)
        del cls_token
        # 4. Transformer + Cross-Attention + Scale
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 降维 + cross-attention (可训练)
            x_loda_down = self.down_proj(x_loda)
            dist_loda_down = self.down_proj_dist(dist_loda)
            aes_down = self.down_proj_aes(aes_loda)
            x_down = (
                    x_loda_down
                    + self.cross_atten_1(x_loda_down, dist_loda_down)
                    + self.cross_atten_aes(x_loda_down, aes_down)
            )
            x_up = self.up_proj(x_down)
            x_loda = x_loda + x_up * self.scale_factor[i]
            # Transformer block (含LoRA, 冻结CLIP本体，只有LoRA参数训练)
            x_loda = block(x_loda)
        # 5. Final LayerNorm and CLS Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])
        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj
        x_loda = x_loda.view(x_size[0], x_size[1], 512)


        # ###############################
        # # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # # Step 1: Patch Embedding
        # x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        # x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose
        #
        # # Step 2: Add CLS token
        # cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        # x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]
        #
        # # Step 3: Add Positional Embedding and LayerNorm
        # x_loda = x_loda + self.feature_extraction.positional_embedding
        # x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)
        #
        # # 清理不再需要的张量，节省显存
        # del cls_token
        #
        # # Step 4: Process through each Transformer Block
        # for i, block in enumerate(self.feature_extraction.transformer.resblocks):
        #     # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
        #     with torch.no_grad():
        #         # if i % 3 == 0:
        #         # 在这里你可以操作每个 Block 的输出，例如：
        #         # x = x + 1  # 示例操作，实际中可以根据需求进行操作
        #         x_loda_down = self.down_proj(x_loda).detach()
        #         dist_loda_down = self.down_proj_dist(dist_loda).detach()
        #         aes_down = self.down_proj_aes(aes_loda).detach()
        #         x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down) + self.cross_atten_aes(x_loda_down, aes_down)
        #         x_up = self.up_proj(x_down)
        #         # x_loda = x_loda + x_up * self.scale_factor
        #         x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
        #         # 清理不再需要的中间变量
        #         # del x_loda_down, dist_loda_down, x_down, x_up
        #         # torch.cuda.empty_cache()
        #
        #         x_loda = block(x_loda)
        #         # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度
        #
        # # Step 5: Final LayerNorm and Class Token output
        # x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])
        #
        # if self.feature_extraction.proj is not None:
        #     x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        # x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # # ###############################

        # x_load_aes = self.cross_atten_2(x_loda, aes)

        x_videomae_features = self.unify_videomae_rec(videomae)
        x_load_aes_videomae = self.cross_atten_2(x_loda, x_videomae_features)

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x = x + x_load_aes_videomae
        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


# LoRA 层实现
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 低秩近似矩阵
        self.W_A = nn.Parameter(torch.randn(input_dim, rank) * 0.02)
        self.W_B = nn.Parameter(torch.randn(rank, output_dim) * 0.02)

    def forward(self, x):
        # 使用低秩近似变换输入
        return x @ self.W_A @ self.W_B
class xgc_color(torch.nn.Module):

    def __init__(self, feat_len=8, rank=32):
        super(xgc_color, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")

        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features

        # 2. 冻结 CLIP 参数
        for param in clip_vit_b_pretrained_features.parameters():
            param.requires_grad = False

        # 在特定的层中添加 LoRA
        self.lora_layer_1 = LoRALayer(512, 512, rank)  # 示例 LoRA 层，适用于注意力块
        self.lora_layer_2 = LoRALayer(768, 768, rank)  # 示例 LoRA 层，适用于投影
        self.lora_layer_3 = LoRALayer(512, 512, rank)  # 示例 LoRA 层，适用于视频特征（VideoMAE）

        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 768)
        self.aes_proj = nn.Linear(768, 768)  # 线性变换


        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_aes = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.down_proj_aes = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, videomae, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(1, self.feat_len, 1)
        aes = aes.view(-1, 768)
        aes_loda = self.aes_proj(aes).unsqueeze(1).repeat(1, 197, 1)

        # 使用 LoRA 层进行特征适应
        # aes_loda = self.lora_layer_2(aes_loda)  # 对美学特征应用 LoRA


        """"""


        # 下面是你已有的 pipeline  lora
        # ============== 开始 forward 流程 ================
        # 1. Patch Embedding
        x_loda = self.feature_extraction.conv1(x)
        x_loda = x_loda.flatten(2).transpose(1, 2)
        # 2. Add CLS Token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)
        # 3. Positional Embedding & LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)
        del cls_token
        # 4. Transformer + Cross-Attention + Scale
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 降维 + cross-attention (可训练)
            x_loda_down = self.down_proj(x_loda)
            aes_down = self.down_proj_aes(aes_loda)
            x_down = (
                    x_loda_down
                    + self.cross_atten_aes(x_loda_down, aes_down)
            )
            x_up = self.up_proj(x_down)
            x_loda = x_loda + x_up * self.scale_factor[i]
            # Transformer block (含LoRA, 冻结CLIP本体，只有LoRA参数训练)
            x_loda = block(x_loda)
        # 5. Final LayerNorm and CLS Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])
        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj
        x_loda = x_loda.view(x_size[0], x_size[1], 512)

        x_videomae_features = self.unify_videomae_rec(videomae)
        # x_videomae_features = self.lora_layer_3(x_videomae_features)  # 对时序特征应用 LoRA
        x_load_aes_videomae = self.cross_atten_2(x_loda, x_videomae_features)

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        # x = self.lora_layer_1(x)  # 对输入特征应用 LoRA
        x = x + x_load_aes_videomae
        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x


class kvq_onLSVQ(torch.nn.Module):
    def __init__(self, feat_len=4):
        super(kvq_onLSVQ, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, dist, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                if i % 2 == 0:
                    # 在这里你可以操作每个 Block 的输出，例如：
                    # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                    x_loda_down = self.down_proj(x_loda).detach()
                    dist_loda_down = self.down_proj(dist_loda).detach()
                    x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                    x_up = self.up_proj(x_down)
                    # x_loda = x_loda + x_up * self.scale_factor
                    x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                    # 清理不再需要的中间变量
                    # del x_loda_down, dist_loda_down, x_down, x_up
                    # torch.cuda.empty_cache()

                    x_loda = block(x_loda)
                    # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x= self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param(torch.nn.Module):
    def __init__(self, feat_len=8):
        super(kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, dist, videomae):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 257, 1)
        ###############################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                # if i % 3 == 0:
                # 在这里你可以操作每个 Block 的输出，例如：
                # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                # x_loda = x_loda + x_up * self.scale_factor
                x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                # 清理不再需要的中间变量
                # del x_loda_down, dist_loda_down, x_down, x_up
                # torch.cuda.empty_cache()

                x_loda = block(x_loda)
                # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x= self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_v2(torch.nn.Module):
    def __init__(self, feat_len=4):
        super(kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_v2, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 768)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.cross_atten_3 = CrossAttention(256)
        self.down_proj_2 = self.base_quality_regression(512, 256, 64)
        self.up_proj_2 = self.base_quality_regression(64, 256, 512)

        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

        self.depth_conv = EfficientImageSplitter(3, 32)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, dist, videomae, aes):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape

        # step0. 调整输入到224X224
        x = self.depth_conv(x)

        # x: batch * frames x 3 x height x width
        # x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = x.view(-1, x_size[2], 224, 224)

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)

        # aes = self.unify_aesthetic_rec(aes)
        # aes = aes.repeat(1, self.feat_len, 1)
        aes = self.unify_aesthetic_rec(aes)
        aes = aes.repeat(self.feat_len, 197, 1)

        dist_aes = 0.572 * dist_loda + 0.428 * aes

        ##############dist#################
        # 1. 首先通过卷积层（Conv2d -> Patch Embedding）
        # Step 1: Patch Embedding
        x_loda = self.feature_extraction.conv1(x)  # Conv2d -> Patch Embedding
        x_loda = x_loda.flatten(2).transpose(1, 2)  # Flatten and Transpose

        # Step 2: Add CLS token
        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)  # [batch_size, num_patches+1, embedding_dim]

        # Step 3: Add Positional Embedding and LayerNorm
        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)  # (b,257,1024)

        # 清理不再需要的张量，节省显存
        del cls_token

        # Step 4: Process through each Transformer Block
        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            # 使用 torch.no_grad() 避免不必要的梯度计算，降低显存占用
            with torch.no_grad():
                if i % 2 == 0:
                    # 在这里你可以操作每个 Block 的输出，例如：
                    # x = x + 1  # 示例操作，实际中可以根据需求进行操作
                    x_loda_down = self.down_proj(x_loda).detach()
                    dist_loda_down = self.down_proj(dist_aes).detach()
                    x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                    x_up = self.up_proj(x_down)
                    # x_loda = x_loda + x_up * self.scale_factor
                    x_loda = x_loda + x_up * self.scale_factor[i]  # 第i个 init scale_factor 矩阵
                    # 清理不再需要的中间变量
                    # del x_loda_down, dist_loda_down, x_down, x_up
                    # torch.cuda.empty_cache()

                    x_loda = block(x_loda)
                    # print(f"Block {i + 1} output shape: {x.shape}")  # 打印每个 Block 的输出维度

        # Step 5: Final LayerNorm and Class Token output
        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj  # Linear projection if it exists
        x_loda = x_loda.view(x_size[0], x_size[1], 512)
        # ###############################

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)
        x_videomae_features = self.unify_videomae_rec(videomae)

        x_loda_aes_videomae = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda_aes_videomae


        # x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x

class kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_v3(torch.nn.Module):
    def __init__(self, feat_len=4):
        super(kvq_ViTbCLIP_SpatialTemporalDistortion_loda_modular_dropout_cross_attention_videomae_v10_param_v3, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        # ViT_L_14, _ = clip.load("ViT-L/14@336px")

        # ViT_L_14, _ = clip.load("ViT-L/14")


        clip_vit_b_pretrained_features = ViT_B_16.visual
        # clip_vit_b_pretrained_features = ViT_L_14.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)
        self.unify_aesthetic_rec = nn.Linear(784, 512)
        self.aes_proj = nn.Linear(512, 512)
        self.unify_lp_rec = self.base_quality_regression(5 * 512, 1024, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 512)

        self.down_proj_lp = self.base_quality_regression(512, 256, 64)
        self.down_proj_dist = self.base_quality_regression(512, 256, 64)
        self.down_proj_aes = self.base_quality_regression(512, 256, 64)
        self.down_proj_videomae = self.base_quality_regression(512, 256, 64)
        self.down_proj_x = self.base_quality_regression(512, 256, 64)

        self.cross_atten_lp = CrossAttention(64)
        self.cross_atten_dist = CrossAttention(64)
        self.cross_atten_aes = CrossAttention(64)
        self.cross_atten_videomae = CrossAttention(64)

        self.up_proj = self.base_quality_regression(64, 256, 512)
        self.scale_factor = nn.Parameter(torch.randn(4, 4, 64) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, dist, videomae, aes, lp):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)  # (b,feat_len,512)

        lp = self.unify_lp_rec(lp)
        dist = self.unify_distortion_rec_loda(dist)
        aes = self.unify_aesthetic_rec(aes)
        aes = self.aes_proj(aes).repeat(1, self.feat_len, 1)
        videomae = self.unify_videomae_rec(videomae)

        lp = self.down_proj_lp(lp)
        dist = self.down_proj_dist(dist)
        aes = self.down_proj_aes(aes)
        videomae = self.down_proj_videomae(videomae)
        x = self.down_proj_x(x)

        x_lp = self.cross_atten_lp(x, lp)
        x_dist = self.cross_atten_dist(x, dist)
        x_aes = self.cross_atten_aes(x, aes)
        x_videomae = self.cross_atten_videomae(x, videomae)

        x = x + self.scale_factor[0]*x_lp + self.scale_factor[1]*x_dist + self.scale_factor[2]*x_aes + self.scale_factor[3]*x_videomae
        x = self.up_proj(x)

        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x