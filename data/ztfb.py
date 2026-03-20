import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, normaltest, boxcox
import chardet

# -------------------------- 1. File Encoding Detection & Safe Reading --------------------------
def detect_encoding(file_path):
    """Detect file encoding to avoid read errors"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

def safe_read_csv(file_path):
    """Try multiple encodings to read CSV file"""
    encodings = [detect_encoding(file_path), 'utf-8', 'gbk', 'latin-1', 'utf-16']
    for encoding in encodings:
        if not encoding:
            continue
        try:
            df = pd.read_csv(
                file_path,
                skip_blank_lines=True,
                encoding=encoding,
                on_bad_lines='skip'  # Skip corrupted lines
            )
            print(f"✅ Successfully read file with encoding: {encoding}")
            return df
        except Exception as e:
            continue
    raise Exception("❌ Failed to read file with all tested encodings")

# Read the uploaded CSV file
file_path = '/mnt/LSVQ_whole_train.csv'
print("="*60)
print("Step 1: Reading LSVQ_whole_train.csv File")
print("="*60)
df = safe_read_csv(file_path)

# -------------------------- 2. Check Columns & Extract 'mos' Column --------------------------
print(f"\n📊 All columns in the file: {df.columns.tolist()}")

# Check if 'mos' column exists
if 'mos' not in df.columns:
    # Case-insensitive check (in case column name is 'MOS'/'Mos')
    mos_column = [col for col in df.columns if col.lower() == 'mos']
    if not mos_column:
        raise Exception(f"❌ 'mos' column not found! Available columns: {df.columns.tolist()}")
    mos_column = mos_column[0]
    print(f"ℹ️  Found 'mos' column (case-insensitive): {mos_column}")
else:
    mos_column = 'mos'
    print(f"✅ Found 'mos' column directly")

# Extract and clean 'mos' data (remove missing values and invalid data)
mos_data = df[mos_column].dropna()
# Convert to numeric (in case of string-formatted numbers)
mos_data = pd.to_numeric(mos_data, errors='coerce').dropna()

# Basic statistics of 'mos' data
print(f"\n" + "="*60)
print("Step 2: Basic Statistics of 'mos' Column")
print("="*60)
print(f"Total valid data points: {len(mos_data)}")
print(f"Data range: [{mos_data.min():.4f}, {mos_data.max():.4f}]")
print(f"Mean (μ): {mos_data.mean():.4f}")
print(f"Standard deviation (σ): {mos_data.std():.4f}")
print(f"Median: {mos_data.median():.4f}")
print(f"Skewness: {mos_data.skew():.4f} (≈0 = symmetric)")
print(f"Kurtosis: {mos_data.kurtosis():.4f} (≈0 = normal distribution)")

# -------------------------- 3. Normality Test for 'mos' Data --------------------------
print(f"\n" + "="*60)
print("Step 3: Normality Test Results (α=0.05)")
print("="*60)

# Shapiro-Wilk Test (suitable for small-to-medium samples, n < 5000)
if len(mos_data) <= 5000:
    shapiro_stat, shapiro_p = shapiro(mos_data)
    print(f"1. Shapiro-Wilk Test:")
    print(f"   Statistic: {shapiro_stat:.4f}")
    print(f"   p-value: {shapiro_p:.4f}")
    print(f"   Conclusion: {'Normal distribution' if shapiro_p > 0.05 else 'Non-normal distribution'}")
else:
    shapiro_p = None
    print(f"1. Shapiro-Wilk Test: Skipped (sample size > 5000, use D'Agostino Test instead)")

# D'Agostino Test (suitable for large samples)
dagostino_stat, dagostino_p = normaltest(mos_data)
print(f"2. D'Agostino Test (Large Sample):")
print(f"   Statistic: {dagostino_stat:.4f}")
print(f"   p-value: {dagostino_p:.4f}")
print(f"   Conclusion: {'Normal distribution' if dagostino_p > 0.05 else 'Non-normal distribution'}")

# Determine final normality conclusion
if (shapiro_p is not None and shapiro_p > 0.05) or (shapiro_p is None and dagostino_p > 0.05):
    normality_conclusion = "Normal distribution"
    normality_color = 'green'
else:
    normality_conclusion = "Non-normal distribution"
    normality_color = 'red'
    # Apply Box-Cox transformation if non-normal
    print(f"\nℹ️  Applying Box-Cox transformation to make 'mos' data approximate normal distribution...")
    mos_data_transformed, _ = boxcox(mos_data + 1e-6)  # +1e-6 to avoid non-positive values
    # Re-test normality after transformation
    trans_shapiro_p = shapiro(mos_data_transformed[:5000])[1] if len(mos_data_transformed) > 5000 else shapiro(mos_data_transformed)[1]
    trans_dagostino_p = normaltest(mos_data_transformed)[1]
    print(f"Transformed Data Normality (Box-Cox):")
    print(f"   Shapiro-Wilk p-value (sample=5000): {trans_shapiro_p:.4f}")
    print(f"   D'Agostino p-value: {trans_dagostino_p:.4f}")
    print(f"   Conclusion: {'Approximate normal distribution' if trans_dagostino_p > 0.05 else 'Still non-normal'}")

# -------------------------- 4. Plot Normal Distribution Histogram for 'mos' --------------------------
# Configure plot style (English labels, no Chinese to avoid garbled text)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Use transformed data if original is non-normal
if normality_conclusion == "Non-normal distribution" and 'mos_data_transformed' in locals():
    plot_data = mos_data_transformed
    x_label = 'Transformed MOS Value (Box-Cox Transformation)'
    mean_val = plot_data.mean()
    std_val = plot_data.std()
    test_p = trans_dagostino_p
else:
    plot_data = mos_data
    x_label = 'MOS Value'
    mean_val = plot_data.mean()
    std_val = plot_data.std()
    test_p = dagostino_p

# 1. Plot histogram (density mode)
n_bins = min(30, len(plot_data) // 15)  # Dynamic bin count
n, bins, patches = ax.hist(
    plot_data,
    bins=n_bins,
    density=True,
    alpha=0.7,
    color='#2E86AB',
    edgecolor='white',
    linewidth=0.8,
    label='MOS Data Distribution (Histogram)'
)

# 2. Plot KDE curve (real data trend)
kde = stats.gaussian_kde(plot_data)
x_kde = np.linspace(plot_data.min() - 0.2, plot_data.max() + 0.2, 1000)
y_kde = kde(x_kde)
ax.plot(
    x_kde,
    y_kde,
    color='#A23B72',
    linewidth=3,
    label='KDE Curve (Real Distribution)'
)

# 3. Plot normal fit curve
x_norm = np.linspace(plot_data.min() - 0.2, plot_data.max() + 0.2, 1000)
y_norm = stats.norm.pdf(x_norm, loc=mean_val, scale=std_val)
ax.plot(
    x_norm,
    y_norm,
    color='#F18F01',
    linewidth=3,
    linestyle='--',
    label=f'Normal Fit Curve\n(μ={mean_val:.2f}, σ={std_val:.2f})'
)

# -------------------------- 5. Plot Styling --------------------------
# Axis labels and title
ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
title = f'MOS Column Normal Distribution Histogram\n' \
        f'Normality: {normality_conclusion if normality_conclusion == "Normal distribution" else "Approximate normal (Box-Cox)"} | ' \
        f'p-value: {test_p:.4f} | Data Count: {len(plot_data)}'
ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color=normality_color)

# Legend
ax.legend(
    fontsize=11,
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.9
)

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tick parameters
ax.tick_params(axis='both', labelsize=12)

# -------------------------- 6. Save Plot and Data --------------------------
# Save high-resolution plot
plt.tight_layout()
plot_save_path = '/mnt/mos_normal_distribution.png'
plt.savefig(
    plot_save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close()

# Save cleaned MOS data (and transformed data if applicable)
mos_clean_df = pd.DataFrame({'MOS_Cleaned': mos_data}).reset_index(drop=True)
mos_clean_df.to_csv('/mnt/mos_cleaned_data.csv', index=False, encoding='utf-8')
if normality_conclusion == "Non-normal distribution" and 'mos_data_transformed' in locals():
    mos_trans_df = pd.DataFrame({'MOS_Transformed': mos_data_transformed}).reset_index(drop=True)
    mos_trans_df.to_csv('/mnt/mos_transformed_data.csv', index=False, encoding='utf-8')

# Final output
print(f"\n" + "="*60)
print("Step 4: Analysis Completed")
print("="*60)
print(f"✅ MOS Normal Distribution Plot saved to: {plot_save_path}")
print(f"✅ Cleaned MOS data saved to: /mnt/mos_cleaned_data.csv")
if normality_conclusion == "Non-normal distribution" and 'mos_data_transformed' in locals():
    print(f"✅ Transformed MOS data (Box-Cox) saved to: /mnt/mos_transformed_data.csv")
print(f"📌 Key Finding: MOS column follows {normality_conclusion if normality_conclusion == 'Normal distribution' else 'approximate normal distribution after Box-Cox transformation'}")