import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, num_mod, dim, t_dim, rank, drank, trank) -> None:
        super().__init__()
        self.n_mod = num_mod
        self.rank = rank
        self.drank = drank
        self.trank = trank
        self.t_dim = t_dim

        self.down = nn.ParameterList(
            [
                nn.Linear(dim, rank)
                for _ in range(self.n_mod)
            ]
        )

        self.factor_dim_list = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.drank, rank, dim))
                for _ in range(self.n_mod)
            ]
        )
        self.factor_t_list = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.trank, t_dim[i], rank))
                for i in range(self.n_mod)
            ]
        )

    def forward(self, x):
        original = torch.cat(x, dim=1)
        for i in range(self.n_mod):
            x[i] = self.down[i](x[i])
            x[i] = nn.functional.gelu(x[i])

        fusion_feat = []
        for i in range(self.n_mod):
            wd = self.factor_dim_list[i].sum(dim=0)
            tmp_feat = torch.matmul(x[i], wd)
            # tmp_feat = tmp_feat.permute(0, 2, 1)
            wt = self.factor_t_list[i].sum(dim=0)
            tmp_feat = torch.matmul(tmp_feat, wt)
            fusion_feat.append(tmp_feat)

        fusion = fusion_feat[0]
        for i in range(1, self.n_mod):
            fusion *= fusion_feat[i]
        # fusion = fusion.permute(0, 2, 1)
        fusion += original[:, : fusion.shape[1], :]
        return fusion

