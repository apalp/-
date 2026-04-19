"""
============================================================
Dagum基尼系数分解
输入: 养老金融高质量发展_TOPSIS评价结果.xlsx → "完整面板数据" sheet
输出: Dagum分解结果表 + 图表
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\标准化处理\养老金融高质量发展_TOPSIS评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\dagum基尼系数分析"
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
print(f"数据: {panel.shape[0]}行, {panel['地区'].nunique()}省, {panel['年份'].nunique()}年")

# ============================================================
# Dagum基尼系数分解函数
# ============================================================
def dagum_gini(data_dict):
    """
    data_dict: {'区域名': np.array(该区域各省的综合得分)}
    返回分解结果字典
    """
    regions = list(data_dict.keys())
    k = len(regions)
    all_data = np.concatenate(list(data_dict.values()))
    n = len(all_data)
    mu = all_data.mean()

    # 总体基尼系数
    G_total = 0
    for i in range(n):
        for j in range(n):
            G_total += abs(all_data[i] - all_data[j])
    G_total = G_total / (2 * n * n * mu)

    # 各区域参数
    n_r = {r: len(data_dict[r]) for r in regions}
    mu_r = {r: data_dict[r].mean() for r in regions}

    # 区域内基尼系数 G_rr
    G_rr = {}
    for r in regions:
        vals = data_dict[r]
        nr = len(vals)
        s = sum(abs(vals[i] - vals[j]) for i in range(nr) for j in range(nr))
        G_rr[r] = s / (2 * nr * nr * mu_r[r]) if mu_r[r] > 0 else 0

    # 区域间基尼系数 G_rs
    G_rs = {}
    for i_idx in range(k):
        for j_idx in range(i_idx + 1, k):
            r1, r2 = regions[i_idx], regions[j_idx]
            vals1, vals2 = data_dict[r1], data_dict[r2]
            s = sum(abs(a - b) for a in vals1 for b in vals2)
            G_rs[(r1, r2)] = s / (len(vals1) * len(vals2) * (mu_r[r1] + mu_r[r2]))

    # 区域内贡献 Gw
    Gw = 0
    for r in regions:
        Gw += G_rr[r] * (n_r[r] / n) ** 2 * (mu_r[r] / mu)

    # 区域间净贡献 Gnb 和 超变密度 Gt
    Gnb = 0
    Gt = 0
    for i_idx in range(k):
        for j_idx in range(i_idx + 1, k):
            r1, r2 = regions[i_idx], regions[j_idx]
            if mu_r[r1] < mu_r[r2]:
                r1, r2 = r2, r1

            vals1, vals2 = data_dict[r1], data_dict[r2]
            p1 = n_r[r1] / n
            p2 = n_r[r2] / n
            s1 = mu_r[r1] / mu
            s2 = mu_r[r2] / mu

            key = (regions[i_idx], regions[j_idx])
            G_jh = G_rs.get(key, G_rs.get((key[1], key[0]), 0))

            d_jh = sum(a - b for a in vals1 for b in vals2 if a > b)
            p_jh = sum(b - a for a in vals1 for b in vals2 if a < b)

            n1n2 = len(vals1) * len(vals2)
            D_jh = (d_jh - p_jh) / (n1n2 * (mu_r[r1] + mu_r[r2])) if n1n2 > 0 else 0

            if G_jh > 0:
                Gnb += G_jh * p1 * p2 * (s1 + s2) * D_jh / G_jh
                Gt += G_jh * p1 * p2 * (s1 + s2) * (1 - D_jh / G_jh)

    G_sum = Gw + Gnb + Gt

    return {
        'G_total': G_total,
        'Gw': Gw, 'Gnb': Gnb, 'Gt': Gt,
        'pct_w': Gw / G_sum * 100 if G_sum > 0 else 0,
        'pct_nb': Gnb / G_sum * 100 if G_sum > 0 else 0,
        'pct_t': Gt / G_sum * 100 if G_sum > 0 else 0,
        'G_rr': G_rr, 'G_rs': G_rs,
    }

# ============================================================
# 逐年计算
# ============================================================
print("\n" + "=" * 70)
print("Dagum基尼系数分解结果")
print("=" * 70)

results = []
for year in STUDY_YEARS:
    year_data = panel[panel['年份'] == year]
    data_dict = {r: year_data[year_data['地区'].isin(ps)]['综合得分'].values
                 for r, ps in REGION_MAP.items()}
    res = dagum_gini(data_dict)
    res['年份'] = year
    results.append(res)

    print(f"\n{year}年: G={res['G_total']:.4f}")
    print(f"  区域内 Gw ={res['Gw']:.4f} ({res['pct_w']:.1f}%)")
    print(f"  区域间 Gnb={res['Gnb']:.4f} ({res['pct_nb']:.1f}%)")
    print(f"  超变密度  ={res['Gt']:.4f} ({res['pct_t']:.1f}%)")
    print(f"  东部={res['G_rr']['东部']:.4f} 中部={res['G_rr']['中部']:.4f} "
          f"西部={res['G_rr']['西部']:.4f} 东北={res['G_rr']['东北']:.4f}")

# 整理结果表
dagum_df = pd.DataFrame([{
    '年份': r['年份'],
    '总体基尼系数G': round(r['G_total'], 4),
    '区域内贡献Gw': round(r['Gw'], 4),
    '区域间贡献Gnb': round(r['Gnb'], 4),
    '超变密度Gt': round(r['Gt'], 4),
    '区域内贡献率%': round(r['pct_w'], 2),
    '区域间贡献率%': round(r['pct_nb'], 2),
    '超变密度贡献率%': round(r['pct_t'], 2),
    '东部_区域内基尼': round(r['G_rr']['东部'], 4),
    '中部_区域内基尼': round(r['G_rr']['中部'], 4),
    '西部_区域内基尼': round(r['G_rr']['西部'], 4),
    '东北_区域内基尼': round(r['G_rr']['东北'], 4),
} for r in results])

# 区域间基尼系数表
inter_rows = []
for r in results:
    row = {'年份': r['年份']}
    for (r1, r2), g in r['G_rs'].items():
        row[f'{r1}-{r2}'] = round(g, 4)
    inter_rows.append(row)
inter_df = pd.DataFrame(inter_rows)

# ============================================================
# 可视化
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 图8: 总体基尼 + 三项分解趋势 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(dagum_df['年份'], dagum_df['总体基尼系数G'], 'ko-', linewidth=2.5, markersize=7, label='总体 G')
ax1.plot(dagum_df['年份'], dagum_df['区域内贡献Gw'], 's--', color='#4A90D9', linewidth=1.5, label='区域内 Gw')
ax1.plot(dagum_df['年份'], dagum_df['区域间贡献Gnb'], '^--', color='#4AA64A', linewidth=1.5, label='区域间 Gnb')
ax1.plot(dagum_df['年份'], dagum_df['超变密度Gt'], 'D--', color='#D98A4A', linewidth=1.5, label='超变密度 Gt')
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('基尼系数', fontsize=12)
ax1.set_title('Dagum基尼系数及分解趋势', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_xticks(STUDY_YEARS)

ax2.stackplot(dagum_df['年份'],
              dagum_df['区域内贡献率%'],
              dagum_df['区域间贡献率%'],
              dagum_df['超变密度贡献率%'],
              labels=['区域内', '区域间', '超变密度'],
              colors=['#4A90D9', '#4AA64A', '#D98A4A'], alpha=0.7)
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('贡献率 (%)', fontsize=12)
ax2.set_title('差异来源贡献率构成', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_ylim(0, 100)
ax2.grid(alpha=0.3)
ax2.set_xticks(STUDY_YEARS)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图8_Dagum基尼系数.png'), dpi=300)
print("\n  ✓ 图8_Dagum基尼系数.png")

# --- 图9: 区域内基尼系数对比 ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'东部': '#4A90D9', '中部': '#4AA64A', '西部': '#D98A4A', '东北': '#9B59B6'}
for region in ['东部', '中部', '西部', '东北']:
    ax.plot(dagum_df['年份'], dagum_df[f'{region}_区域内基尼'],
            'o-', color=colors[region], linewidth=2, markersize=6, label=region)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('区域内基尼系数', fontsize=12)
ax.set_title('四大区域内部差异演变', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(STUDY_YEARS)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图9_区域内基尼系数.png'), dpi=300)
print("  ✓ 图9_区域内基尼系数.png")

# ============================================================
# 保存Excel
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "Dagum基尼系数分解结果.xlsx")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    dagum_df.to_excel(writer, sheet_name='总体分解', index=False)
    inter_df.to_excel(writer, sheet_name='区域间基尼系数', index=False)

print(f"\n  ✓ {output_path}")
print("\n完成!")
