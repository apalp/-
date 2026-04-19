

import pandas as pd
import numpy as np
import os



# ============================================================
# CONFIG
# ============================================================
INDICATORS = [
    (r"D:\用户\Desktop\标准化处理\001基本养老保险覆盖率_标准化结果.xlsx",       "X1",  "基本养老保险覆盖率",       "A"),
    (r"D:\用户\Desktop\标准化处理\002基本养老保险基金收入强度_标准化结果.xlsx", "X2",  "基本养老保险基金收入强度", "A"),
    (r"D:\用户\Desktop\标准化处理\003基本养老保险基金可支付月数_标准化结果.xlsx","X3",  "基本养老保险基金可支付月数","A"),
    (r"D:\用户\Desktop\标准化处理\004城镇职工养老金水平_标准化结果.xlsx",       "X4",  "城镇职工养老金水平",       "A"),
    (r"D:\用户\Desktop\标准化处理\005城乡居民养老金水平_标准化结果.xlsx",       "X5",  "城乡居民养老金水平",       "A"),
    (r"D:\用户\Desktop\标准化处理\006企业年金覆盖率_标准化结果.xlsx",          "X6",  "企业年金覆盖率",           "A"),
    (r"D:\用户\Desktop\标准化处理\007企业年金基金积累强度_标准化结果.xlsx",     "X7",  "企业年金基金积累强度",     "A"),
    (r"D:\用户\Desktop\标准化处理\008健康保险密度_标准化结果.xlsx",            "X8",  "健康保险密度",             "B"),
    (r"D:\用户\Desktop\标准化处理\009人寿保险密度_标准化结果.xlsx",            "X9",  "人寿保险密度",             "B"),
    (r"D:\用户\Desktop\标准化处理\010长期护理保险试点覆盖率_标准化结果.xlsx",   "X10", "长期护理保险试点覆盖率",   "B"),
    (r"D:\用户\Desktop\标准化处理\011数字普惠金融综合指数_标准化结果.xlsx",     "X11", "数字普惠金融综合指数",     "B"),
    (r"D:\用户\Desktop\标准化处理\012全国分省每万人银行网点数_标准化结果.xlsx", "X12", "银行业金融机构网点密度",   "B"),
    (r"D:\用户\Desktop\标准化处理\013PPP养老产业投资强度_标准化结果.xlsx",     "X13", "养老服务类PPP投资强度",    "C"),
    (r"D:\用户\Desktop\标准化处理\014卫生和社会工作固定资产投资增速_标准化结果.xlsx","X14","卫生和社会工作固定资产投资增速","C"),
]

OUTPUT_DIR = r"D:\用户\Desktop\标准化处理\权重限定百分之12"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

DIM_NAMES = {"A": "养老金金融", "B": "养老服务金融", "C": "养老产业金融"}

# ============================================================
# 省份名称统一
# ============================================================
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

# ============================================================
# 通用函数: 熵值法 + CRITIC + TOPSIS
# ============================================================
def entropy_weight(X):
    """熵值法赋权, X: (n, m) 标准化矩阵"""
    n, m = X.shape
    col_sums = X.sum(axis=0)
    P = X / col_sums
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_P = np.where(P > 0, np.log(P), 0)
    k = 1.0 / np.log(n)
    E = -k * (P * ln_P).sum(axis=0)
    D = 1 - E
    return D / D.sum()

def critic_weight(X):
    """CRITIC法赋权"""
    m = X.shape[1]
    sigma = X.std(axis=0, ddof=1)
    corr = np.corrcoef(X.T)
    C = np.zeros(m)
    for j in range(m):
        C[j] = sigma[j] * sum(1 - corr[j, k] for k in range(m) if k != j)
    return C / C.sum()

def combined_weight(w1, w2, cap=0.12):
    """乘法归一化组合 + 单指标权重上限约束"""
    w = w1 * w2
    w = w / w.sum()
    # 迭代截断
    for _ in range(100):
        exceed = w > cap
        if not exceed.any():
            break
        surplus = w[exceed].sum() - cap * exceed.sum()
        w[exceed] = cap
        remain = ~exceed
        w[remain] = w[remain] + surplus * (w[remain] / w[remain].sum())
    return w

def topsis_score(X, w):
    """TOPSIS计算综合得分"""
    V = X * w
    V_plus = V.max(axis=0)
    V_minus = V.min(axis=0)
    D_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))
    return D_minus / (D_plus + D_minus)

def run_full_pipeline(panel, ind_codes, ind_dims):
    """完整流程: 熵值+CRITIC+组合+TOPSIS, 返回panel带得分"""
    X = panel[ind_codes].values
    w_e = entropy_weight(X)
    w_c = critic_weight(X)
    w = combined_weight(w_e, w_c)
    panel = panel.copy()
    panel['综合得分'] = topsis_score(X, w)

    # 子维度
    for dim_code, dim_name in DIM_NAMES.items():
        cols = [c for c in ind_codes if ind_dims[c] == dim_code]
        if len(cols) == 0:
            continue
        idx = [ind_codes.index(c) for c in cols]
        dim_w = w[idx]
        dim_w = dim_w / dim_w.sum()
        X_dim = X[:, idx]
        panel[f'{dim_name}_子指数'] = topsis_score(X_dim, dim_w)

    return panel, w, w_e, w_c

# ============================================================
# STEP 1: 读取合并
# ============================================================
print("=" * 70)
print("STEP 1: 读取合并14个指标")
print("=" * 70)

all_long = []
for filepath, code, cname, dim in INDICATORS:
    df = pd.read_excel(filepath, sheet_name="标准化结果")
    df['地区'] = df['地区'].apply(clean_name)
    long = df.melt(id_vars='地区', var_name='年份', value_name=code)
    long['年份'] = long['年份'].astype(int)
    long = long[long['年份'].isin(STUDY_YEARS)]
    all_long.append(long)
    print(f"  ✓ {code} {cname}")

panel = all_long[0][['地区', '年份', INDICATORS[0][1]]].copy()
for i in range(1, len(all_long)):
    code = INDICATORS[i][1]
    panel = panel.merge(all_long[i][['地区', '年份', code]], on=['地区', '年份'], how='outer')

ind_codes = [x[1] for x in INDICATORS]       # ['X1','X2',...,'X14']
ind_names = {x[1]: x[2] for x in INDICATORS}  # {'X1':'基本养老保险覆盖率',...}
ind_dims = {x[1]: x[3] for x in INDICATORS}   # {'X1':'A',...}

print(f"\n合并完成: {panel.shape[0]}行 x {panel.shape[1]}列")

# ============================================================
# STEP 2: 基准模型 (14个指标全部)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 基准模型 (全部14个指标)")
print("=" * 70)

panel_base, w_base, w_e_base, w_c_base = run_full_pipeline(panel, ind_codes, ind_dims)

# 2023排名
rank_base = panel_base[panel_base['年份'] == 2023][['地区', '综合得分']].sort_values('综合得分', ascending=False)
rank_base['基准排名'] = range(1, len(rank_base) + 1)

print("\n基准模型 2023年排名:")
for _, row in rank_base.iterrows():
    print(f"  {row['基准排名']:>2}. {row['地区']:<6} {row['综合得分']:.4f}")

# ============================================================
# STEP 3: 稳健性检验1 — 剔除PPP指标 (13个指标)
# ============================================================
print("\n" + "=" * 70)
print("稳健性检验1: 剔除PPP投资强度 (X13)")
print("=" * 70)

codes_no_ppp = [c for c in ind_codes if c != 'X13']
panel_r1, w_r1, _, _ = run_full_pipeline(panel, codes_no_ppp, ind_dims)

rank_r1 = panel_r1[panel_r1['年份'] == 2023][['地区', '综合得分']].sort_values('综合得分', ascending=False)
rank_r1.columns = ['地区', '得分_剔除PPP']
rank_r1['排名_剔除PPP'] = range(1, len(rank_r1) + 1)

# ============================================================
# STEP 4: 稳健性检验2 — 等权重法 (14个指标各1/14)
# ============================================================
print("\n" + "=" * 70)
print("稳健性检验2: 等权重法 (各指标权重 = 1/14)")
print("=" * 70)

w_equal = np.ones(len(ind_codes)) / len(ind_codes)
panel_r2 = panel.copy()
panel_r2['综合得分'] = topsis_score(panel[ind_codes].values, w_equal)

rank_r2 = panel_r2[panel_r2['年份'] == 2023][['地区', '综合得分']].sort_values('综合得分', ascending=False)
rank_r2.columns = ['地区', '得分_等权重']
rank_r2['排名_等权重'] = range(1, len(rank_r2) + 1)

# ============================================================
# STEP 5: 稳健性对比 — Spearman秩相关
# ============================================================
print("\n" + "=" * 70)
print("稳健性对比")
print("=" * 70)

comparison = rank_base[['地区', '基准排名']].merge(
    rank_r1[['地区', '排名_剔除PPP']], on='地区'
).merge(
    rank_r2[['地区', '排名_等权重']], on='地区'
)

from scipy import stats
rho1, p1 = stats.spearmanr(comparison['基准排名'], comparison['排名_剔除PPP'])
rho2, p2 = stats.spearmanr(comparison['基准排名'], comparison['排名_等权重'])

print(f"\nSpearman秩相关系数:")
print(f"  基准 vs 剔除PPP:  ρ = {rho1:.4f}, p = {p1:.6f} {'✓ 显著' if p1 < 0.01 else ''}")
print(f"  基准 vs 等权重:   ρ = {rho2:.4f}, p = {p2:.6f} {'✓ 显著' if p2 < 0.01 else ''}")
print(f"\n判断: ", end="")
if rho1 > 0.85 and rho2 > 0.85:
    print("两个检验ρ均>0.85, 结论稳健!")
elif rho1 > 0.85 or rho2 > 0.85:
    print("部分稳健, 需在论文中讨论差异原因")
else:
    print("稳健性不足, 需要重新审视指标体系")

# 排名对比表
print(f"\n{'省份':<6} {'基准':>4} {'剔除PPP':>8} {'等权重':>6} {'最大排名变动':>10}")
print("-" * 40)
for _, row in comparison.sort_values('基准排名').iterrows():
    max_change = max(abs(row['基准排名'] - row['排名_剔除PPP']),
                     abs(row['基准排名'] - row['排名_等权重']))
    flag = " ⚠️" if max_change >= 10 else ""
    print(f"{row['地区']:<6} {row['基准排名']:>4} {row['排名_剔除PPP']:>8} {row['排名_等权重']:>6} {max_change:>10.0f}{flag}")

# ============================================================
# STEP 6: 描述性可视化
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 生成可视化图表")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 图1: 各年份综合得分箱线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
box_data = [panel_base[panel_base['年份'] == y]['综合得分'].values for y in STUDY_YEARS]
bp = ax.boxplot(box_data, labels=[str(y) for y in STUDY_YEARS], patch_artist=True,
                boxprops=dict(facecolor='#E8F0FE', edgecolor='#4A90D9'),
                medianprops=dict(color='#D94A4A', linewidth=2),
                whiskerprops=dict(color='#4A90D9'),
                capprops=dict(color='#4A90D9'),
                flierprops=dict(marker='o', markerfacecolor='#D94A4A', markersize=5))
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('综合得分 (S)', fontsize=12)
ax.set_title('养老金融高质量发展综合得分年度分布', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图1_综合得分箱线图.png'), dpi=300)
print("  ✓ 图1_综合得分箱线图.png")

# --- 图2: 三个子维度得分箱线图 (分面) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
dim_list = ['养老金金融', '养老服务金融', '养老产业金融']
colors = ['#E8F4E8', '#E8F0FE', '#FEF0E8']
edge_colors = ['#4AA64A', '#4A90D9', '#D98A4A']

for idx, (dim, color, ec) in enumerate(zip(dim_list, colors, edge_colors)):
    col = f'{dim}_子指数'
    data = [panel_base[panel_base['年份'] == y][col].values for y in STUDY_YEARS]
    axes[idx].boxplot(data, labels=[str(y) for y in STUDY_YEARS], patch_artist=True,
                      boxprops=dict(facecolor=color, edgecolor=ec),
                      medianprops=dict(color='#D94A4A', linewidth=2),
                      whiskerprops=dict(color=ec),
                      capprops=dict(color=ec),
                      flierprops=dict(marker='o', markerfacecolor='#D94A4A', markersize=4))
    axes[idx].set_title(dim, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('年份', fontsize=10)
    axes[idx].grid(axis='y', alpha=0.3)

axes[0].set_ylabel('子指数', fontsize=11)
plt.suptitle('养老金融三维度子指数年度分布', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图2_三维度子指数箱线图.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图2_三维度子指数箱线图.png")

# --- 图3: 全国均值趋势折线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
means = panel_base.groupby('年份').agg({
    '综合得分': 'mean',
    '养老金金融_子指数': 'mean',
    '养老服务金融_子指数': 'mean',
    '养老产业金融_子指数': 'mean',
}).reset_index()

ax.plot(means['年份'], means['综合得分'], 'ko-', linewidth=2.5, markersize=7, label='综合得分 S')
ax.plot(means['年份'], means['养老金金融_子指数'], 's--', color='#4AA64A', linewidth=1.5, markersize=5, label='养老金金融 $S_A$')
ax.plot(means['年份'], means['养老服务金融_子指数'], '^--', color='#4A90D9', linewidth=1.5, markersize=5, label='养老服务金融 $S_B$')
ax.plot(means['年份'], means['养老产业金融_子指数'], 'D--', color='#D98A4A', linewidth=1.5, markersize=5, label='养老产业金融 $S_C$')

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('均值', fontsize=12)
ax.set_title('养老金融高质量发展全国均值趋势 (2016-2023)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xticks(STUDY_YEARS)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图3_全国均值趋势.png'), dpi=300)
print("  ✓ 图3_全国均值趋势.png")

# --- 图4: 四大区域均值对比 ---
region_map = {
    '东部': ['北京','天津','河北','上海','江苏','浙江','福建','山东','广东','海南'],
    '中部': ['山西','安徽','江西','河南','湖北','湖南'],
    '西部': ['内蒙古','广西','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆'],
    '东北': ['辽宁','吉林','黑龙江'],
}

panel_base_copy = panel_base.copy()
panel_base_copy['区域'] = panel_base_copy['地区'].map(
    {p: r for r, ps in region_map.items() for p in ps}
)

fig, ax = plt.subplots(figsize=(10, 6))
region_colors = {'东部': '#4A90D9', '中部': '#4AA64A', '西部': '#D98A4A', '东北': '#9B59B6'}
for region in ['东部', '中部', '西部', '东北']:
    rmeans = panel_base_copy[panel_base_copy['区域'] == region].groupby('年份')['综合得分'].mean()
    ax.plot(rmeans.index, rmeans.values, 'o-', color=region_colors[region],
            linewidth=2, markersize=6, label=region)

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('综合得分均值', fontsize=12)
ax.set_title('四大区域养老金融高质量发展趋势对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(STUDY_YEARS)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图4_四大区域趋势对比.png'), dpi=300)
print("  ✓ 图4_四大区域趋势对比.png")

# --- 图5: 稳健性检验排名对比散点图 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(comparison['基准排名'], comparison['排名_剔除PPP'], c='#4A90D9', s=50, zorder=3)
axes[0].plot([1, 31], [1, 31], 'k--', alpha=0.3)
axes[0].set_xlabel('基准模型排名', fontsize=11)
axes[0].set_ylabel('剔除PPP后排名', fontsize=11)
axes[0].set_title(f'稳健性检验1: 剔除PPP\nSpearman ρ = {rho1:.3f}', fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].scatter(comparison['基准排名'], comparison['排名_等权重'], c='#D98A4A', s=50, zorder=3)
axes[1].plot([1, 31], [1, 31], 'k--', alpha=0.3)
axes[1].set_xlabel('基准模型排名', fontsize=11)
axes[1].set_ylabel('等权重法排名', fontsize=11)
axes[1].set_title(f'稳健性检验2: 等权重法\nSpearman ρ = {rho2:.3f}', fontsize=12)
axes[1].grid(alpha=0.3)

plt.suptitle('稳健性检验: 省份排名对比 (2023年)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图5_稳健性检验排名对比.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图5_稳健性检验排名对比.png")

# ============================================================
# STEP 7: 保存所有数值结果
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "稳健性检验与描述统计.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 指标对照表
    ind_table = pd.DataFrame({
        '编号': [x[1] for x in INDICATORS],
        '指标名称': [x[2] for x in INDICATORS],
        '维度代号': [x[3] for x in INDICATORS],
        '维度名称': [DIM_NAMES[x[3]] for x in INDICATORS],
        '基准组合权重': w_base,
    })
    ind_table.to_excel(writer, sheet_name='指标对照表', index=False)

    # 稳健性排名对比
    comparison.to_excel(writer, sheet_name='排名对比_2023', index=False)

    # 各年描述统计
    desc_stats = panel_base.groupby('年份').agg({
        '综合得分': ['mean', 'std', 'min', 'max', 'median'],
    }).round(4)
    desc_stats.columns = ['均值', '标准差', '最小值', '最大值', '中位数']
    desc_stats.to_excel(writer, sheet_name='各年描述统计')

    # 区域均值
    region_means = panel_base_copy.groupby(['区域', '年份'])['综合得分'].mean().unstack().round(4)
    region_means.to_excel(writer, sheet_name='区域均值')

print(f"\n  ✓ 数值结果: {output_path}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("全部完成! 输出文件汇总:")
print("=" * 70)
print(f"  1. 图1_综合得分箱线图.png")
print(f"  2. 图2_三维度子指数箱线图.png")
print(f"  3. 图3_全国均值趋势.png")
print(f"  4. 图4_四大区域趋势对比.png")
print(f"  5. 图5_稳健性检验排名对比.png")
print(f"  6. 稳健性检验与描述统计.xlsx")

print("\n" + "=" * 70)
print("后续研究路径:")
print("=" * 70)
