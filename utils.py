import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torch.nn.functional as F
import torch.nn as nn



def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    print('mos: {}'.format(y_label), 'predict logistic: {}'.format(y_output_logistic))

    # 2. 构造数据框：包含真实值、原始预测值、拟合后预测值
    # 确保输入是np.array，方便按列拼接
    y_label_arr = np.array(y_label)
    y_output_arr = np.array(y_output)
    y_output_logistic_arr = np.array(y_output_logistic)
    data = {
        "真实MOS分数(y_label)": y_label_arr,
        "模型原始预测值(y_output)": y_output_arr,
        "拟合后预测值(y_output_logistic)": y_output_logistic_arr
    }
    df = pd.DataFrame(data)
    save_path = "LBVD_prediction_results.xlsx"

    # 3. 保存到Excel（index=False 不保存行索引）
    try:
        df.to_excel(save_path, index=False)
        print(f"\n数据已成功保存到：{save_path}")
        # 打印数据预览（前5行）
        print("\n数据预览：")
        print(df.head())
    except Exception as e:
        print(f"\n保存Excel失败：{str(e)}")

    return PLCC, SRCC, KRCC, RMSE


def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_label-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE



class L1RankLoss(torch.nn.Module):
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total


def plcc_loss(y, y_pred):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = F.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = F.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def rank_loss(y, y_pred):
    ranking_loss = F.relu((y_pred - y) * torch.sign(y_pred - y))
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def plcc_rank_loss(y_label, y_output):
    if y_label.shape[0] <= 1:
        # batch size = 1，用 MSE 替代
        return F.mse_loss(y_output, y_label)
    plcc = plcc_loss(y_label, y_output)
    rank = rank_loss(y_label, y_output)
    return plcc + rank*0.3


def plcc_l1_loss(y_label, y_output):
    plcc = plcc_loss(y_label, y_output)
    l1_loss = F.l1_loss(y_label, y_output)
    return plcc + 0.0025*l1_loss

# 自定义高斯噪声
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
