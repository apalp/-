"""
============================================================
核密度估计 (Kernel Density Estimation)
输入: 养老金融高质量发展_TOPSIS评价结果.xlsx → "完整面板数据" sheet
输出: 核密度图
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\标准化处理\养老金融高质量发展_TOPSIS评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\核密度估计"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

REGION_MAP = {
    '东部': ['北京','天津','河北','上海','江苏','浙江','福建','山东','广东','海南'],
    '中部': ['山西','安徽','江西','河南','湖北','湖南'],
    '西部': ['内蒙古','广西','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆'],
    '东北': ['辽宁','吉林','黑龙江'],
}

# ============================================================
# 读取数据
# ============================================================
panel = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')
panel['区域'] = panel['地区'].map({p: r for r, ps in REGION_MAP.items() for p in ps})
print(f"数据: {panel.shape[0]}行")

# ============================================================
# 图6: 全国核密度演进
# ============================================================
print("\n生成图表...")

fig, ax = plt.subplots(figsize=(10, 6))
colors_year = plt.cm.Blues(np.linspace(0.3, 1.0, len(STUDY_YEARS)))

for i, year in enumerate(STUDY_YEARS):
    vals = panel[panel['年份'] == year]['综合得分'].values
    kde = gaussian_kde(vals, bw_method=0.3)
    x_grid = np.linspace(0, 0.85, 300)
    ax.plot(x_grid, kde(x_grid), color=colors_year[i], linewidth=1.8, label=str(year))

ax.set_xlabel('综合得分 S', fontsize=12)
ax.set_ylabel('核密度', fontsize=12)
ax.set_title('养老金融高质量发展综合得分核密度演进 (2016-2023)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图6_全国核密度演进.png'), dpi=300)
print("  ✓ 图6_全国核密度演进.png")
plt.close()

# ============================================================
# 图7: 四大区域核密度 (2x2分面)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
region_list = ['东部', '中部', '西部', '东北']
show_years = [2016, 2019, 2023]
year_colors = {2016: '#2196F3', 2019: '#FF9800', 2023: '#4CAF50'}

for idx, region in enumerate(region_list):
    ax = axes[idx]
    provinces = REGION_MAP[region]
    for year in show_years:
        vals = panel[(panel['年份'] == year) & (panel['地区'].isin(provinces))]['综合得分'].values
        if len(vals) > 2:
            kde = gaussian_kde(vals, bw_method=0.4)
            x_grid = np.linspace(0, 0.85, 300)
            ax.plot(x_grid, kde(x_grid), color=year_colors[year], linewidth=2, label=str(year))
            ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=year_colors[year])
    ax.set_title(f'{region} ({len(provinces)}省)', fontsize=12, fontweight='bold')
    ax.set_xlabel('综合得分 S', fontsize=10)
    ax.set_ylabel('核密度', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('四大区域核密度演进对比', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图7_四大区域核密度.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图7_四大区域核密度.png")
plt.close()

# ============================================================
# 图6b: 三个子维度核密度演进 (分面)
# ============================================================
dim_cols = {
    '养老金金融': '养老金金融_子指数',
    '养老服务金融': '养老服务金融_子指数',
    '养老产业金融': '养老产业金融_子指数',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dim_color_maps = [plt.cm.Greens, plt.cm.Blues, plt.cm.Oranges]

for idx, (dim_name, col_name) in enumerate(dim_cols.items()):
    ax = axes[idx]
    cmap = dim_color_maps[idx]
    year_colors_dim = cmap(np.linspace(0.3, 1.0, len(STUDY_YEARS)))

    for i, year in enumerate(STUDY_YEARS):
        vals = panel[panel['年份'] == year][col_name].values
        kde = gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(0, 1.0, 300)
        ax.plot(x_grid, kde(x_grid), color=year_colors_dim[i], linewidth=1.5, label=str(year))

    ax.set_title(dim_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('子指数', fontsize=10)
    ax.set_ylabel('核密度', fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

plt.suptitle('三维度子指数核密度演进', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图6b_三维度核密度演进.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图6b_三维度核密度演进.png")
plt.close()

# ============================================================
# 核密度特征提取 (论文写作用)
# ============================================================
print("\n" + "=" * 70)
print("核密度特征摘要 (论文写作参考)")
print("=" * 70)

print("\n【全国综合得分分布特征】")
for year in [2016, 2019, 2023]:
    vals = panel[panel['年份'] == year]['综合得分'].values
    kde = gaussian_kde(vals, bw_method=0.3)
    x_grid = np.linspace(0, 0.85, 500)
    density = kde(x_grid)
    peak_x = x_grid[np.argmax(density)]
    peak_y = density.max()

    print(f"\n  {year}年:")
    print(f"    主峰位置: S={peak_x:.3f}, 密度={peak_y:.2f}")
    print(f"    均值={vals.mean():.3f}, 中位数={np.median(vals):.3f}, 标准差={vals.std():.3f}")

    # 检测双峰/多峰
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(density, height=peak_y * 0.3, distance=30)
    if len(peaks) > 1:
        peak_positions = [f"S={x_grid[p]:.3f}" for p in peaks]
        print(f"    ⚠️ 检测到{len(peaks)}个峰: {', '.join(peak_positions)} → 存在极化现象")
    else:
        print(f"    单峰分布, 无明显极化")

print("\n【分布演进趋势判断】")
std_2016 = panel[panel['年份'] == 2016]['综合得分'].std()
std_2023 = panel[panel['年份'] == 2023]['综合得分'].std()
mean_2016 = panel[panel['年份'] == 2016]['综合得分'].mean()
mean_2023 = panel[panel['年份'] == 2023]['综合得分'].mean()

print(f"  标准差: {std_2016:.4f} (2016) → {std_2023:.4f} (2023)")
if std_2023 < std_2016:
    print(f"  → 分布趋于收敛 (离散度下降{(std_2016-std_2023)/std_2016*100:.1f}%)")
else:
    print(f"  → 分布趋于发散 (离散度上升{(std_2023-std_2016)/std_2016*100:.1f}%)")

print(f"  均值: {mean_2016:.4f} (2016) → {mean_2023:.4f} (2023)")
print(f"  → 整体水平{'提升' if mean_2023 > mean_2016 else '下降'} {abs(mean_2023-mean_2016)/mean_2016*100:.1f}%")
print(f"  → 主峰右移 = 整体水平提升, 峰高变化 = 集中/分散程度变化")

print("\n完成!")
