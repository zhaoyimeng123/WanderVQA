import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats  # 用于计算线性回归
# ===================== Ubuntu专属：中文乱码终极修复 =====================
# 方案1：使用Ubuntu自带的「文泉驿微米黑」（推荐，无需额外安装）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# 方案2：若方案1无效，改用「Noto Sans CJK SC」（Ubuntu 18.04+ 默认预装）
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
# 强制关闭字体缓存（Ubuntu下解决字体加载异常）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块问题
# ===================== 1. 读取Excel数据 =====================
excel_path = "LSVQ_prediction_results.xlsx"  # 替换为你的Excel路径
try:
    df = pd.read_excel(excel_path)
    true_mos = df["真实MOS分数(y_label)"].values
    pred_mos = df["拟合后预测值(y_output_logistic)"].values
    print(f"数据读取成功，共{len(true_mos)}个样本")
except Exception as e:
    print(f"数据读取失败：{e}")
    exit()

# ===================== 2. 计算线性趋势线 =====================
# 用scipy.stats做线性回归，得到斜率、截距、相关系数等
slope, intercept, r_value, p_value, std_err = stats.linregress(true_mos, pred_mos)
# 生成趋势线的x/y值（覆盖数据范围）
trend_x = np.linspace(min(true_mos), max(true_mos), 100)
trend_y = slope * trend_x + intercept

# 打印趋势线信息（便于分析）
print(f"\n趋势线信息：")
print(f"线性方程：y = {slope:.4f}x + {intercept:.4f}")
print(f"相关系数(R)：{r_value:.4f}")
print(f"R²（拟合优度）：{r_value**2:.4f}")

# ===================== 3. 绘制图表（散点+趋势线+y=x参考线） =====================
plt.figure(figsize=(8, 8))

# 1. 绘制散点（核心数据）
plt.scatter(true_mos, pred_mos,
            c='darkviolet', marker='*', s=60, alpha=0.8, label='Data points')

# 2. 绘制y=x参考线（红色虚线，基准线）
min_val = min(np.min(true_mos), np.min(pred_mos)) - 2
max_val = max(np.max(true_mos), np.max(pred_mos)) + 2
plt.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=1.5, label='y=x (理想拟合曲线)')

# 3. 绘制线性趋势线（蓝色实线，拟合数据）
plt.plot(trend_x, trend_y,
         'g--', linewidth=2.5, label=f'模型预测趋势线y={slope:.2f}x+{intercept:.2f})')

# ===================== 4. 图表美化 =====================
plt.xlabel('True MOS Score', fontsize=12, fontweight='bold')
plt.ylabel('Predicted MOS Score', fontsize=12, fontweight='bold')
plt.title('True MOS vs Predicted MOS (with Trend Line)', fontsize=14, fontweight='bold', pad=15)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, linestyle='--', alpha=0.5)
# 图例包含散点、y=x线、趋势线
plt.legend(loc='upper left', fontsize=9)
plt.tight_layout()

# ===================== 5. 保存/显示 =====================
plt.savefig('mos_pred_scatter_with_trend.png', dpi=300, bbox_inches='tight')
plt.show()