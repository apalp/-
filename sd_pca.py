"""
============================================================
SD 系统动力学仿真_PCA
  - 使用PCA版基准FE回归系数 (无倒U型)
  - 三情景模拟 2024-2035
  - 加入"十五五"政策目标对照
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BG_COLOR = '#FAFBFC'

# ============================================================
# CONFIG
# ============================================================
PCA_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
REG_PATH = r"D:\用户\Desktop\pca\回归\回归优化结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\SD"

os.makedirs(OUTPUT_DIR, exist_ok=True)

STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
SIM_YEARS = list(range(2023, 2036))

# "十五五"政策目标 
POLICY_TARGET_2030 = 0.60  # 2030年目标
POLICY_TARGET_2035 = 0.75  # 2035年目标

# ============================================================
# STEP 1: 读取数据
# ============================================================
print("=" * 70)
print("STEP 1: 读取历史数据")
print("=" * 70)

# 读取PCA综合得分
pca_panel = pd.read_excel(PCA_PATH, sheet_name='完整面板数据')
print(f"PCA面板: {pca_panel.shape[0]}行")

# 读取回归面板 (含解释变量)
try:
    reg_panel = pd.read_excel(REG_PATH, sheet_name='完整数据')
    print(f"回归面板: {reg_panel.shape[0]}行")
    HAS_REG = True
except:
    print("  ⚠️ 未找到回归面板, 使用默认参数")
    HAS_REG = False

# 2023年初始值
data_2023 = pca_panel[pca_panel['年份'] == 2023]
S_2023 = data_2023['PCA综合得分'].mean()
print(f"\n2023年全国均值: S = {S_2023:.4f}")

# 历史趋势
national_means = pca_panel.groupby('年份')['PCA综合得分'].mean()
print(f"\n历史年均增长率:")
for yr in STUDY_YEARS[1:]:
    prev = national_means[yr - 1]
    curr = national_means[yr]
    gr = (curr - prev) / prev * 100
    print(f"  {yr}: {gr:+.2f}%")

avg_growth = national_means.pct_change().dropna().mean()
print(f"\n2017-2023平均年增长率: {avg_growth*100:.2f}%")

# 2023年各解释变量均值 (仿真起点)
if HAS_REG:
    reg_2023 = reg_panel[reg_panel['年份'] == 2023]
    init_vars = {
        '老龄化率': reg_2023['老龄化率'].mean(),
        'ln人均GDP': np.log(reg_2023['人均GDP'].mean()) if '人均GDP' in reg_2023.columns else reg_2023['ln人均GDP'].mean(),
        '城镇化率': reg_2023['城镇化率'].mean(),
        '信贷深度': reg_2023['信贷深度'].mean(),
        '金融业占比': reg_2023['金融业占比'].mean(),
        '第三产业占比': reg_2023['第三产业占比'].mean(),
        '养老床位密度': reg_2023['养老床位密度'].mean(),
    }
else:
    init_vars = {
        '老龄化率': 15.0,
        'ln人均GDP': 11.3,
        '城镇化率': 67.0,
        '信贷深度': 3.4,
        '金融业占比': 8.0,
        '第三产业占比': 56.0,
        '养老床位密度': 5.8,
    }

print(f"\n2023年解释变量初始值:")
for k, v in init_vars.items():
    print(f"  {k}: {v:.2f}")

# ============================================================
# STEP 2: PCA版FE回归系数 (无倒U型)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 模型参数")
print("=" * 70)

# PCA版基准FE回归系数
COEFS = {
    '老龄化率': 0.0086,
    'ln人均GDP': 0.4666,
    '城镇化率': 0.0046,
    '信贷深度': 0.0425,
    '金融业占比': 0.0123,
    '第三产业占比': 0.0061,
    '养老床位密度': -0.0343,
}

print("回归系数 (PCA版基准FE):")
for k, v in COEFS.items():
    print(f"  {k}: {v:+.4f}")

# ============================================================
# STEP 3: 定义三个情景
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: 定义情景")
print("=" * 70)

SCENARIOS = {
    '基准情景': {
        'desc': '维持当前趋势',
        'color': '#3498DB',
        'linestyle': '-',
        'deltas': {
            '老龄化率': 0.40,
            'ln人均GDP': 0.04,
            '城镇化率': 0.70,
            '信贷深度': 0.05,
            '金融业占比': 0.10,
            '第三产业占比': 0.50,
            '养老床位密度': 0.20,
        }
    },
    '乐观情景': {
        'desc': '积极政策 + 经济稳健',
        'color': '#2ECC71',
        'linestyle': '--',
        'deltas': {
            '老龄化率': 0.35,
            'ln人均GDP': 0.05,
            '城镇化率': 1.00,
            '信贷深度': 0.10,
            '金融业占比': 0.20,
            '第三产业占比': 0.80,
            '养老床位密度': 0.10,
        }
    },
    '悲观情景': {
        'desc': '经济下行 + 深度老龄化',
        'color': '#E74C3C',
        'linestyle': '-.',
        'deltas': {
            '老龄化率': 0.50,
            'ln人均GDP': 0.02,
            '城镇化率': 0.40,
            '信贷深度': 0.02,
            '金融业占比': 0.00,
            '第三产业占比': 0.30,
            '养老床位密度': 0.30,
        }
    },
}

for name, sc in SCENARIOS.items():
    print(f"\n{name}: {sc['desc']}")
    for k, v in sc['deltas'].items():
        print(f"  {k}_年增: {v:+.2f}")

# ============================================================
# STEP 4: 运行仿真
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: 运行仿真 (2024-2035)")
print("=" * 70)

results = {}

for sc_name, sc in SCENARIOS.items():
    scores = [S_2023]
    vars_trajectory = {k: [v] for k, v in init_vars.items()}

    for t in range(1, len(SIM_YEARS)):
        # 更新各变量
        current_vars = {}
        for var_name in init_vars:
            prev_val = vars_trajectory[var_name][-1]
            delta = sc['deltas'][var_name]
            new_val = prev_val + delta
            current_vars[var_name] = new_val
            vars_trajectory[var_name].append(new_val)

        # 计算dS (回归系数 × 变量变化量)
        dS = sum(COEFS[var] * sc['deltas'][var] for var in COEFS)

        # 综合得分更新
        new_S = scores[-1] + dS
        new_S = max(0, min(1, new_S))  # 限制在[0,1]
        scores.append(new_S)

    results[sc_name] = {
        'scores': scores,
        'vars': vars_trajectory,
    }

    # 找到2030和2035的值
    idx_2030 = SIM_YEARS.index(2030)
    idx_2035 = SIM_YEARS.index(2035)

    print(f"\n{sc_name}:")
    print(f"  2023: S = {scores[0]:.4f}")
    print(f"  2030: S = {scores[idx_2030]:.4f}", end="")
    if scores[idx_2030] >= POLICY_TARGET_2030:
        print(f" ✓ 达到政策目标({POLICY_TARGET_2030})")
    else:
        gap = POLICY_TARGET_2030 - scores[idx_2030]
        print(f" ✗ 距目标差 {gap:.4f}")
    print(f"  2035: S = {scores[idx_2035]:.4f}", end="")
    if scores[idx_2035] >= POLICY_TARGET_2035:
        print(f" ✓ 达到政策目标({POLICY_TARGET_2035})")
    else:
        gap = POLICY_TARGET_2035 - scores[idx_2035]
        print(f" ✗ 距目标差 {gap:.4f}")

# ============================================================
# STEP 5: 敏感性分析
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: 敏感性分析")
print("=" * 70)

sensitivity = {}
base_deltas = SCENARIOS['基准情景']['deltas']

for var in COEFS:
    contribution = COEFS[var] * base_deltas[var]
    sensitivity[var] = contribution

sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n各驱动因素年贡献量 (基准情景):")
total_dS = sum(sensitivity.values())
for rank, (var, contrib) in enumerate(sorted_sens, 1):
    direction = "促进" if contrib > 0 else "抑制"
    pct = abs(contrib) / abs(total_dS) * 100
    print(f"  {rank}. {var}: {direction} ({contrib:+.4f}/年, 占{pct:.1f}%)")
print(f"  合计年变化: {total_dS:+.4f}")

# ============================================================
# STEP 6: 可视化
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 生成图表")
print("=" * 70)

# --- 情景模拟主图 ---
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# 历史值
hist_means = [national_means[y] for y in STUDY_YEARS]
ax.plot(STUDY_YEARS, hist_means, 'ko-', linewidth=3, markersize=8,
        label='历史值', zorder=5)

# 三情景
for sc_name, sc in SCENARIOS.items():
    scores = results[sc_name]['scores']
    ax.plot(SIM_YEARS, scores,
            color=sc['color'], linestyle=sc['linestyle'],
            linewidth=2.5, markersize=5, marker='o',
            label=f"{sc_name}: {sc['desc']}", zorder=4)

# 政策目标线
ax.axhline(y=POLICY_TARGET_2030, color='#95A5A6', linestyle=':', linewidth=1.5, alpha=0.8)
ax.annotate(f'2030目标: {POLICY_TARGET_2030}',
           xy=(2030, POLICY_TARGET_2030), xytext=(2025.5, POLICY_TARGET_2030 + 0.02),
           fontsize=10, color='#7F8C8D', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='#95A5A6'))

ax.axhline(y=POLICY_TARGET_2035, color='#95A5A6', linestyle=':', linewidth=1.5, alpha=0.8)
ax.annotate(f'2035目标: {POLICY_TARGET_2035}',
           xy=(2035, POLICY_TARGET_2035), xytext=(2031, POLICY_TARGET_2035 + 0.02),
           fontsize=10, color='#7F8C8D', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='#95A5A6'))

# 分隔线
ax.axvline(x=2023.5, color='gray', linestyle='--', alpha=0.3)
ax.text(2020, ax.get_ylim()[1] * 0.95, '历史', fontsize=12, color='gray', ha='center')
ax.text(2029, ax.get_ylim()[1] * 0.95, '预测', fontsize=12, color='gray', ha='center')

# 标注2030和2035的值
for sc_name in SCENARIOS:
    scores = results[sc_name]['scores']
    color = SCENARIOS[sc_name]['color']
    idx_2030 = SIM_YEARS.index(2030)
    idx_2035 = SIM_YEARS.index(2035)
    ax.annotate(f'{scores[idx_2030]:.3f}',
               xy=(2030, scores[idx_2030]),
               xytext=(5, 10), textcoords='offset points',
               fontsize=9, color=color, fontweight='bold')
    ax.annotate(f'{scores[idx_2035]:.3f}',
               xy=(2035, scores[idx_2035]),
               xytext=(5, -15), textcoords='offset points',
               fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('年份', fontsize=13)
ax.set_ylabel('PCA综合得分', fontsize=13)
ax.set_title('养老金融高质量发展情景模拟 (2016-2035)', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(alpha=0.2, linestyle='--')
ax.set_xticks(list(range(2016, 2036, 2)))
ax.tick_params(labelsize=11)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '情景模拟.png'), dpi=300,
            bbox_inches='tight', facecolor=BG_COLOR)
print("  ✓ 情景模拟.png")
plt.close()

# --- 驱动因素轨迹 ---
var_names_cn = {
    '老龄化率': '老龄化率 (%)',
    'ln人均GDP': 'ln人均GDP',
    '城镇化率': '城镇化率 (%)',
    '信贷深度': '信贷深度',
    '金融业占比': '金融业占比 (%)',
    '第三产业占比': '第三产业占比 (%)',
    '养老床位密度': '养老床位密度',
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=BG_COLOR)
axes = axes.flatten()

for idx, var in enumerate(COEFS.keys()):
    ax = axes[idx]
    ax.set_facecolor(BG_COLOR)
    for sc_name, sc in SCENARIOS.items():
        vals = results[sc_name]['vars'][var]
        ax.plot(SIM_YEARS, vals,
                color=sc['color'], linestyle=sc['linestyle'],
                linewidth=2, label=sc_name.replace('情景', ''))
    ax.set_title(var_names_cn.get(var, var), fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)

# 最后一个子图放图例
axes[-1].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=11,
          bbox_to_anchor=(0.95, 0.08), framealpha=0.9)

plt.suptitle('驱动因素变化轨迹 (2023-2035)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '驱动因素轨迹.png'), dpi=300,
            bbox_inches='tight', facecolor=BG_COLOR)
print("  ✓ 驱动因素轨迹.png")
plt.close()

# --- 图14: 敏感性分析 (水平条形图) ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

vars_sorted = [x[0] for x in sorted_sens]
vals_sorted = [x[1] for x in sorted_sens]
colors_bar = ['#2ECC71' if v > 0 else '#E74C3C' for v in vals_sorted]

bars = ax.barh(range(len(vars_sorted)), vals_sorted, color=colors_bar, alpha=0.85, height=0.6)

for i, (var, val) in enumerate(zip(vars_sorted, vals_sorted)):
    pct = abs(val) / abs(total_dS) * 100
    offset = 0.0002 if val > 0 else -0.0002
    ax.text(val + offset, i, f'{val:+.4f} ({pct:.0f}%)',
            va='center', fontsize=10, fontweight='bold',
            color=colors_bar[i])

ax.set_yticks(range(len(vars_sorted)))
ax.set_yticklabels(vars_sorted, fontsize=11)
ax.set_xlabel('年贡献量 (基准情景下对PCA综合得分的年变化贡献)', fontsize=11)
ax.set_title('各驱动因素敏感性分析', fontsize=15, fontweight='bold', pad=15)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.2, linestyle='--')
ax.tick_params(labelsize=11)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '敏感性分析.png'), dpi=300,
            bbox_inches='tight', facecolor=BG_COLOR)
print("  ✓ 敏感性分析.png")
plt.close()

# --- 图15 : 政策目标差距分析 ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

idx_2030 = SIM_YEARS.index(2030)
idx_2035 = SIM_YEARS.index(2035)

sc_names = list(SCENARIOS.keys())
x_pos = np.arange(len(sc_names))
width = 0.35

vals_2030 = [results[sc]['scores'][idx_2030] for sc in sc_names]
vals_2035 = [results[sc]['scores'][idx_2035] for sc in sc_names]
sc_colors = [SCENARIOS[sc]['color'] for sc in sc_names]

bars1 = ax.bar(x_pos - width/2, vals_2030, width, label='2030年预测',
               color=sc_colors, alpha=0.7, edgecolor='white')
bars2 = ax.bar(x_pos + width/2, vals_2035, width, label='2035年预测',
               color=sc_colors, alpha=1.0, edgecolor='white')

# 目标线
ax.axhline(y=POLICY_TARGET_2030, color='#E67E22', linestyle='--', linewidth=2,
           label=f'2030目标 ({POLICY_TARGET_2030})')
ax.axhline(y=POLICY_TARGET_2035, color='#8E44AD', linestyle='--', linewidth=2,
           label=f'2035目标 ({POLICY_TARGET_2035})')

# 标注数值
for i, (v30, v35) in enumerate(zip(vals_2030, vals_2035)):
    ax.text(i - width/2, v30 + 0.01, f'{v30:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, v35 + 0.01, f'{v35:.3f}', ha='center', fontsize=10, fontweight='bold')

    # 差距标注
    if v30 < POLICY_TARGET_2030:
        gap = POLICY_TARGET_2030 - v30
        ax.annotate(f'差{gap:.3f}', xy=(i - width/2, POLICY_TARGET_2030),
                   xytext=(i - width/2, POLICY_TARGET_2030 + 0.02),
                   fontsize=8, color='red', ha='center')

ax.set_xticks(x_pos)
ax.set_xticklabels(sc_names, fontsize=12)
ax.set_ylabel('PCA综合得分', fontsize=12)
ax.set_title('情景预测与政策目标对照', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.2, linestyle='--')
ax.set_ylim(0, max(POLICY_TARGET_2035, max(vals_2035)) + 0.1)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '政策目标差距.png'), dpi=300,
            bbox_inches='tight', facecolor=BG_COLOR)
print("  ✓ 政策目标差距.png")
plt.close()

# ============================================================
# STEP 7: 输出Excel
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "SD仿真结果_PCA.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 情景预测值
    sim_df = pd.DataFrame({'年份': SIM_YEARS})
    for sc_name in SCENARIOS:
        sim_df[sc_name] = results[sc_name]['scores']
    sim_df['政策目标_2030'] = POLICY_TARGET_2030
    sim_df['政策目标_2035'] = POLICY_TARGET_2035
    sim_df.to_excel(writer, sheet_name='情景预测', index=False)

    # 敏感性分析
    sens_df = pd.DataFrame(sorted_sens, columns=['变量', '年贡献量'])
    sens_df['贡献占比%'] = (sens_df['年贡献量'].abs() / sens_df['年贡献量'].abs().sum() * 100).round(1)
    sens_df.to_excel(writer, sheet_name='敏感性分析', index=False)

    # 回归系数
    coef_df = pd.DataFrame(list(COEFS.items()), columns=['变量', '系数'])
    coef_df.to_excel(writer, sheet_name='回归系数', index=False)

print(f"\n  ✓ {output_path}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("SD仿真结论")
print("=" * 70)

idx_2030 = SIM_YEARS.index(2030)
idx_2035 = SIM_YEARS.index(2035)

print(f"""
1. 情景预测:
   基准: S = {results['基准情景']['scores'][idx_2030]:.3f}(2030) → {results['基准情景']['scores'][idx_2035]:.3f}(2035)
   乐观: S = {results['乐观情景']['scores'][idx_2030]:.3f}(2030) → {results['乐观情景']['scores'][idx_2035]:.3f}(2035)
   悲观: S = {results['悲观情景']['scores'][idx_2030]:.3f}(2030) → {results['悲观情景']['scores'][idx_2035]:.3f}(2035)

2. 政策目标可达性:
   2030目标({POLICY_TARGET_2030}): 基准{'✓可达' if results['基准情景']['scores'][idx_2030] >= POLICY_TARGET_2030 else '✗不可达'}  乐观{'✓可达' if results['乐观情景']['scores'][idx_2030] >= POLICY_TARGET_2030 else '✗不可达'}
   2035目标({POLICY_TARGET_2035}): 基准{'✓可达' if results['基准情景']['scores'][idx_2035] >= POLICY_TARGET_2035 else '✗不可达'}  乐观{'✓可达' if results['乐观情景']['scores'][idx_2035] >= POLICY_TARGET_2035 else '✗不可达'}

3. 敏感性排序:""")

for rank, (var, contrib) in enumerate(sorted_sens, 1):
    direction = "促进" if contrib > 0 else "抑制"
    print(f"   {rank}. {var}: {direction} ({contrib:+.4f}/年)")

print(f"""
4. 政策建议:
   - ln人均GDP贡献最大({COEFS['ln人均GDP']:+.4f}), 保持经济增长是根本
   - 第三产业占比有稳定正效应, 推动服务业转型升级
   - 信贷深度显著促进, 应扩大养老金融产品供给
   - 养老床位密度为负效应, 应注重服务质量而非数量扩张
""")

print("全部完成!")
