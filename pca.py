"""
============================================================
PCA 主成分分析综合评价
  + 稳健性检验
  + 与TOPSIS结果对比
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CONFIG
# ============================================================
INDICATOR_DIR = r"D:\用户\Desktop\标准化处理"
OUTPUT_DIR = r"D:\用户\Desktop\pca\权重"
TOPSIS_PATH = r"D:\用户\Desktop\标准化处理\养老金融高质量发展_TOPSIS评价结果.xlsx"

STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# 14个指标文件 (文件名, 指标代号, 指标名, 维度)
INDICATORS = [
    ("001基本养老保险覆盖率_标准化结果.xlsx",            "X1",  "基本养老保险覆盖率",       "A"),
    ("002基本养老保险基金收入强度_标准化结果.xlsx",        "X2",  "基本养老保险基金收入强度", "A"),
    ("003基本养老保险基金可支付月数_标准化结果.xlsx",      "X3",  "基本养老保险基金可支付月数","A"),
    ("004城镇职工养老金水平_标准化结果.xlsx",            "X4",  "城镇职工养老金水平",       "A"),
    ("005城乡居民养老金水平_标准化结果.xlsx",            "X5",  "城乡居民养老金水平",       "A"),
    ("006企业年金覆盖率_标准化结果.xlsx",               "X6",  "企业年金覆盖率",           "A"),
    ("007企业年金基金积累强度_标准化结果.xlsx",           "X7",  "企业年金基金积累强度",     "A"),
    ("008健康保险密度_标准化结果.xlsx",                 "X8",  "健康保险密度",             "B"),
    ("009人寿保险密度_标准化结果.xlsx",                 "X9",  "人寿保险密度",             "B"),
    ("010长期护理保险试点覆盖率_标准化结果.xlsx",         "X10", "长期护理保险试点覆盖率",   "B"),
    ("011数字普惠金融综合指数_标准化结果.xlsx",           "X11", "数字普惠金融综合指数",     "B"),
    ("012全国分省每万人银行网点数_标准化结果.xlsx",       "X12", "银行业金融机构网点密度",   "B"),
    ("013newPPP养老产业投资强度_new_标准化结果.xlsx",    "X13", "养老服务类PPP投资强度",    "C"),
    ("014卫生和社会工作固定资产投资增速_标准化结果.xlsx",  "X14", "卫生和社会工作固定资产投资增速","C"),
]

DIM_NAMES = {"A": "养老金金融", "B": "养老服务金融", "C": "养老产业金融"}

# 省份名称统一
NAME_MAP = {
    '北京市': '北京', '天津市': '天津', '河北省': '河北', '山西省': '山西',
    '内蒙古自治区': '内蒙古', '辽宁省': '辽宁', '吉林省': '吉林',
    '黑龙江省': '黑龙江', '上海市': '上海', '江苏省': '江苏',
    '浙江省': '浙江', '安徽省': '安徽', '福建省': '福建', '江西省': '江西',
    '山东省': '山东', '河南省': '河南', '湖北省': '湖北', '湖南省': '湖南',
    '广东省': '广东', '广西壮族自治区': '广西', '海南省': '海南',
    '重庆市': '重庆', '四川省': '四川', '贵州省': '贵州', '云南省': '云南',
    '西藏自治区': '西藏', '陕西省': '陕西', '甘肃省': '甘肃',
    '青海省': '青海', '宁夏回族自治区': '宁夏', '新疆维吾尔自治区': '新疆',
}

def clean_name(name):
    name = str(name).strip()
    return NAME_MAP.get(name, name)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: 读取合并14个指标
# ============================================================
print("=" * 70)
print("STEP 1: 读取合并14个指标")
print("=" * 70)

all_long = []
ind_codes = []
ind_names = {}
ind_dims = {}

for filename, code, cname, dim in INDICATORS:
    filepath = os.path.join(INDICATOR_DIR, filename)
    df = pd.read_excel(filepath, sheet_name="标准化结果")
    df['地区'] = df['地区'].apply(clean_name)
    long = df.melt(id_vars='地区', var_name='年份', value_name=code)
    long['年份'] = long['年份'].astype(int)
    long = long[long['年份'].isin(STUDY_YEARS)]
    all_long.append(long)
    ind_codes.append(code)
    ind_names[code] = cname
    ind_dims[code] = dim
    print(f"  ✓ {code} {cname}")

panel = all_long[0][['地区', '年份', ind_codes[0]]].copy()
for i in range(1, len(all_long)):
    panel = panel.merge(all_long[i][['地区', '年份', ind_codes[i]]], on=['地区', '年份'], how='outer')

print(f"\n合并完成: {panel.shape[0]}行, {panel['地区'].nunique()}省 x {panel['年份'].nunique()}年")

# ============================================================
# STEP 2: PCA主成分分析
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: PCA主成分分析")
print("=" * 70)

X = panel[ind_codes].values

# 标准化 (PCA要求)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA
pca = PCA()
pca.fit(X_std)

# 方差解释率
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n{'主成分':<6} {'方差解释率':>10} {'累计解释率':>10} {'特征值':>8}")
print("-" * 40)
for i in range(len(explained)):
    eigenval = pca.explained_variance_[i]
    marker = " ←" if eigenval >= 1 else ""
    print(f"PC{i+1:<4} {explained[i]:>10.4f} {cumulative[i]:>10.4f} {eigenval:>8.4f}{marker}")

# 按Kaiser准则选取特征值>=1的主成分
n_components = sum(pca.explained_variance_ >= 1)
print(f"\nKaiser准则: 选取{n_components}个主成分 (特征值>=1)")
print(f"累计方差解释率: {cumulative[n_components-1]*100:.2f}%")

# 重新拟合
pca_final = PCA(n_components=n_components)
scores = pca_final.fit_transform(X_std)

# 载荷矩阵
loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)

print(f"\n载荷矩阵:")
header = "".join([f"{'PC'+str(i+1):>8}" for i in range(n_components)])
print(f"{'指标':<28}{header}")
print("-" * (28 + 8 * n_components))
for j, code in enumerate(ind_codes):
    row = "".join([f"{loadings[j, i]:>8.4f}" for i in range(n_components)])
    print(f"{ind_names[code]:<26}{row}")

# ============================================================
# STEP 3: 计算综合得分
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: 计算综合得分")
print("=" * 70)

# 以各主成分方差解释率为权重加权求和
weights_pc = pca_final.explained_variance_ratio_ / pca_final.explained_variance_ratio_.sum()
composite_score = scores @ weights_pc

# 归一化到[0,1]
s_min, s_max = composite_score.min(), composite_score.max()
composite_score_norm = (composite_score - s_min) / (s_max - s_min)

panel['PCA综合得分'] = composite_score_norm

# 各主成分得分也保留
for i in range(n_components):
    panel[f'PC{i+1}'] = scores[:, i]

print(f"PCA综合得分: min={composite_score_norm.min():.4f}, max={composite_score_norm.max():.4f}, "
      f"mean={composite_score_norm.mean():.4f}")

# 2023排名
rank_2023 = panel[panel['年份'] == 2023][['地区', 'PCA综合得分']].sort_values('PCA综合得分', ascending=False)
rank_2023['排名'] = range(1, len(rank_2023) + 1)

print(f"\n2023年PCA排名:")
print(f"\n  Top 10:")
for _, row in rank_2023.head(10).iterrows():
    print(f"    {row['排名']:>2}. {row['地区']:<6} {row['PCA综合得分']:.4f}")
print(f"\n  Bottom 5:")
for _, row in rank_2023.tail(5).iterrows():
    print(f"    {row['排名']:>2}. {row['地区']:<6} {row['PCA综合得分']:.4f}")

# ============================================================
# STEP 4: 稳健性检验 (剔除PPP + 等权重PCA)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: 稳健性检验")
print("=" * 70)

# --- 检验1: 剔除PPP ---
codes_no_ppp = [c for c in ind_codes if c != 'X13']
X_no_ppp = panel[codes_no_ppp].values
X_no_ppp_std = StandardScaler().fit_transform(X_no_ppp)
pca_no_ppp = PCA()
pca_no_ppp.fit(X_no_ppp_std)
n_comp_no_ppp = sum(pca_no_ppp.explained_variance_ >= 1)
pca_no_ppp_final = PCA(n_components=n_comp_no_ppp)
scores_no_ppp = pca_no_ppp_final.fit_transform(X_no_ppp_std)
w_no_ppp = pca_no_ppp_final.explained_variance_ratio_ / pca_no_ppp_final.explained_variance_ratio_.sum()
s_no_ppp = scores_no_ppp @ w_no_ppp
s_no_ppp_norm = (s_no_ppp - s_no_ppp.min()) / (s_no_ppp.max() - s_no_ppp.min())
panel['PCA_剔除PPP'] = s_no_ppp_norm

# --- 检验2: 改变主成分数量 (选n_components-1个) ---
n_comp_alt = n_components - 1  # 少选1个主成分
pca_alt = PCA(n_components=n_comp_alt)
scores_alt = pca_alt.fit_transform(X_std)
w_alt = pca_alt.explained_variance_ratio_ / pca_alt.explained_variance_ratio_.sum()
s_alt = scores_alt @ w_alt
s_alt_norm = (s_alt - s_alt.min()) / (s_alt.max() - s_alt.min())
panel['PCA_少1主成分'] = s_alt_norm

# Spearman相关
data_2023 = panel[panel['年份'] == 2023].copy()

rho1, p1 = stats.spearmanr(data_2023['PCA综合得分'], data_2023['PCA_剔除PPP'])
rho2, p2 = stats.spearmanr(data_2023['PCA综合得分'], data_2023['PCA_少1主成分'])

print(f"\nSpearman秩相关 (2023年):")
print(f"  PCA基准 vs 剔除PPP:  ρ = {rho1:.3f}, p = {p1:.6f}")
print(f"  PCA基准 vs 少1主成分:  ρ = {rho2:.3f}, p = {p2:.6f}")
print(f"\n判断: ", end="")
if rho1 > 0.85 and rho2 > 0.85:
    print(f"✓ 两个检验ρ均>0.85, 结论稳健!")
elif rho1 > 0.85 or rho2 > 0.85:
    print(f"部分稳健")
else:
    print(f"稳健性不足")

# ============================================================
# STEP 5: 与TOPSIS结果对比
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: PCA vs TOPSIS 对比")
print("=" * 70)

topsis = pd.read_excel(TOPSIS_PATH, sheet_name='完整面板数据')
topsis_2023 = topsis[topsis['年份'] == 2023][['地区', '综合得分']].copy()
topsis_2023.columns = ['地区', 'TOPSIS得分']

compare = data_2023[['地区', 'PCA综合得分']].merge(topsis_2023, on='地区')

rho_compare, p_compare = stats.spearmanr(compare['PCA综合得分'], compare['TOPSIS得分'])
print(f"\nPCA vs TOPSIS排名相关: ρ = {rho_compare:.3f}, p = {p_compare:.6f}")

compare['PCA排名'] = compare['PCA综合得分'].rank(ascending=False).astype(int)
compare['TOPSIS排名'] = compare['TOPSIS得分'].rank(ascending=False).astype(int)
compare['排名差'] = abs(compare['PCA排名'] - compare['TOPSIS排名'])
compare = compare.sort_values('PCA排名')

print(f"\n{'省份':<6} {'PCA排名':>6} {'TOPSIS排名':>8} {'排名差':>6}")
print("-" * 30)
for _, row in compare.iterrows():
    flag = " ⚠️" if row['排名差'] >= 10 else ""
    print(f"{row['地区']:<6} {row['PCA排名']:>6} {row['TOPSIS排名']:>8} {row['排名差']:>6.0f}{flag}")

# ============================================================
# STEP 6: 可视化
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 可视化")
print("=" * 70)

# --- 图A: 方差解释率 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(range(1, len(explained)+1), explained*100, color='#4A90D9', alpha=0.7, label='单个')
ax1.plot(range(1, len(explained)+1), cumulative*100, 'ro-', linewidth=2, label='累计')
ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80%线')
ax1.axvline(x=n_components+0.5, color='red', linestyle='--', alpha=0.5, label=f'选取{n_components}个')
ax1.set_xlabel('主成分', fontsize=12)
ax1.set_ylabel('方差解释率 (%)', fontsize=12)
ax1.set_title('主成分方差解释率', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 碎石图
ax2.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, 'bo-', linewidth=2)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='特征值=1 (Kaiser准则)')
ax2.set_xlabel('主成分', fontsize=12)
ax2.set_ylabel('特征值', fontsize=12)
ax2.set_title('碎石图 (Scree Plot)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图A_PCA方差解释率.png'), dpi=300)
print("  ✓ 图A_PCA方差解释率.png")
plt.close()

# --- 图B: 稳健性检验散点图 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA vs 剔除PPP
r1_rank = data_2023['PCA综合得分'].rank(ascending=False)
r1_no_ppp_rank = data_2023['PCA_剔除PPP'].rank(ascending=False)
axes[0].scatter(r1_rank, r1_no_ppp_rank, c='#4A90D9', s=50, zorder=3)
axes[0].plot([1, 31], [1, 31], 'k--', alpha=0.3)
axes[0].set_xlabel('PCA基准排名', fontsize=11)
axes[0].set_ylabel('剔除PPP后排名', fontsize=11)
axes[0].set_title(f'稳健性1: 剔除PPP\nSpearman ρ = {rho1:.3f}', fontsize=12)
axes[0].grid(alpha=0.3)

# PCA vs 简单平均
r1_avg_rank = data_2023['PCA_少1主成分'].rank(ascending=False)
axes[1].scatter(r1_rank, r1_avg_rank, c='#D98A4A', s=50, zorder=3)
axes[1].plot([1, 31], [1, 31], 'k--', alpha=0.3)
axes[1].set_xlabel('PCA基准排名', fontsize=11)
axes[1].set_ylabel('少1主成分排名', fontsize=11)
axes[1].set_title(f'稳健性2: 少1主成分\nSpearman ρ = {rho2:.3f}', fontsize=12)
axes[1].grid(alpha=0.3)

# PCA vs TOPSIS
axes[2].scatter(compare['PCA排名'], compare['TOPSIS排名'], c='#4AA64A', s=50, zorder=3)
axes[2].plot([1, 31], [1, 31], 'k--', alpha=0.3)
axes[2].set_xlabel('PCA排名', fontsize=11)
axes[2].set_ylabel('TOPSIS排名', fontsize=11)
axes[2].set_title(f'PCA vs TOPSIS\nSpearman ρ = {rho_compare:.3f}', fontsize=12)
axes[2].grid(alpha=0.3)

plt.suptitle('稳健性检验与方法对比 (2023年)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图B_稳健性与方法对比.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图B_稳健性与方法对比.png")
plt.close()

# --- 图C: PCA综合得分趋势 ---
fig, ax = plt.subplots(figsize=(10, 6))
means = panel.groupby('年份')['PCA综合得分'].mean()
ax.plot(means.index, means.values, 'ko-', linewidth=2.5, markersize=7)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('PCA综合得分均值', fontsize=12)
ax.set_title('养老金融高质量发展PCA综合得分趋势 (2016-2023)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(STUDY_YEARS)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图C_PCA综合得分趋势.png'), dpi=300)
print("  ✓ 图C_PCA综合得分趋势.png")
plt.close()

# --- 图D: PCA各年箱线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
box_data = [panel[panel['年份'] == y]['PCA综合得分'].values for y in STUDY_YEARS]
bp = ax.boxplot(box_data, labels=[str(y) for y in STUDY_YEARS], patch_artist=True,
                boxprops=dict(facecolor='#E8F0FE', edgecolor='#4A90D9'),
                medianprops=dict(color='#D94A4A', linewidth=2))
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('PCA综合得分', fontsize=12)
ax.set_title('PCA综合得分年度分布', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图D_PCA箱线图.png'), dpi=300)
print("  ✓ 图D_PCA箱线图.png")
plt.close()

# ============================================================
# STEP 7: 输出Excel
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "PCA综合评价结果.xlsx")

# 综合得分宽表
score_wide = panel.pivot(index='地区', columns='年份', values='PCA综合得分').reset_index()
score_wide.columns = ['地区'] + [int(c) for c in score_wide.columns[1:]]

# 载荷矩阵
loading_df = pd.DataFrame(loadings,
                          index=[ind_names[c] for c in ind_codes],
                          columns=[f'PC{i+1}' for i in range(n_components)])

# 方差解释率
var_df = pd.DataFrame({
    '主成分': [f'PC{i+1}' for i in range(len(explained))],
    '特征值': pca.explained_variance_.round(4),
    '方差解释率%': (explained * 100).round(4),
    '累计解释率%': (cumulative * 100).round(4),
})

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    score_wide.to_excel(writer, sheet_name='PCA综合得分', index=False)
    loading_df.to_excel(writer, sheet_name='载荷矩阵')
    var_df.to_excel(writer, sheet_name='方差解释率', index=False)
    compare.to_excel(writer, sheet_name='PCA_vs_TOPSIS', index=False)
    panel.to_excel(writer, sheet_name='完整面板数据', index=False)

print(f"\n  ✓ {output_path}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"""
PCA结果:
  主成分数: {n_components}个 (Kaiser准则)
  累计解释率: {cumulative[n_components-1]*100:.1f}%

稳健性检验:
  剔除PPP:   ρ = {rho1:.3f} {'✓' if rho1 > 0.85 else '✗'}
  少1主成分:   ρ = {rho2:.3f} {'✓' if rho2 > 0.85 else '✗'}

与TOPSIS对比:
  排名相关:   ρ = {rho_compare:.3f}

如果PCA的稳健性检验通过(ρ>0.85), 建议用PCA替代TOPSIS
如果两者排名高度相关(ρ>0.85), 说明两种方法结论一致, 可以互为稳健性检验
""")

print("完成!")
