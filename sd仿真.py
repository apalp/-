"""
============================================================
SD系统动力学仿真
  基于回归结果构建存量流量模型
  三情景模拟: 基准 / 乐观 / 悲观
  预测 2024-2035 年养老金融高质量发展指数
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

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\标准化处理\权重限定百分之12\养老金融高质量发展_TOPSIS评价结果.xlsx"
REG_DATA_PATH = r"D:\用户\Desktop\回归数据处理\老龄化率加平方\回归优化结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\SD仿真"

# 仿真参数
SIM_START = 2024
SIM_END = 2035
DT = 1  # 步长=1年

# ============================================================
# STEP 1: 读取历史数据, 提取2023年各省初始值
# ============================================================
print("=" * 70)
print("STEP 1: 读取历史数据")
print("=" * 70)

# 读取回归面板 
try:
    reg_panel = pd.read_excel(REG_DATA_PATH, sheet_name='完整数据')
except:
    print("  ⚠️ 未找到回归面板数据, 请确认路径")
    print("  尝试从TOPSIS结果读取...")
    reg_panel = None

# 读取TOPSIS结果
topsis = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')

# 提取2023年各省数据作为仿真初始值
data_2023 = topsis[topsis['年份'] == 2023][['地区', '综合得分']].copy()
print(f"2023年初始值: {len(data_2023)}省, 均值={data_2023['综合得分'].mean():.4f}")

# 计算历史趋势 (设定情景参数)
national_means = topsis.groupby('年份')['综合得分'].mean()
growth_rate_history = national_means.pct_change().dropna()

print(f"\n历史年均增长率:")
for yr, gr in growth_rate_history.items():
    print(f"  {int(yr)}: {gr*100:+.2f}%")

avg_growth = growth_rate_history.mean()
print(f"\n2017-2023平均年增长率: {avg_growth*100:.2f}%")

# ============================================================
# STEP 2: 从回归结果提取参数 (存量流量模型的系数)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 构建SD模型参数")
print("=" * 70)

# 基于固定效应倒U型回归结果 (模型2)
# S = f(老龄化率, 老龄化率², ln人均GDP, 城镇化率, 信贷深度, 金融业占比, 第三产业占比, 养老床位密度)

# 回归系数 
BETA = {
    '老龄化率':     0.0236,
    '老龄化率²':   -0.0007,
    'ln人均GDP':    0.0952,
    '城镇化率':     0.0059,
    '信贷深度':     0.0371,
    '金融业占比':    0.0069,
    '第三产业占比':   0.0053,
    '养老床位密度':  -0.0113,
}

print("回归系数 :")
for k, v in BETA.items():
    print(f"  {k}: {v:+.4f}")

# ============================================================
# STEP 3: 定义三个情景
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: 定义情景")
print("=" * 70)

"""
存量流量模型核心方程:
  
  存量: S(t) = 养老金融高质量发展指数
  流量: dS/dt = 增量, 由各驱动因素变化决定
  
  dS = Σ β_i * dX_i
     = β_老龄化 * d老龄化率 + β_老龄化² * d(老龄化率²) 
       + β_GDP * d(lnGDP) + β_城镇化 * d城镇化率 + ...
  
  各因素的年变化量由情景假设决定
"""

# 2023年全国均值 (作为基准点)
INIT_VALUES = {
    '老龄化率':    15.0,     # 约15%
    'ln人均GDP':   11.3,     # 约8万元
    '城镇化率':    66.0,     # 约66%
    '信贷深度':     3.2,     # 贷款/GDP
    '金融业占比':    7.5,     # %
    '第三产业占比':  55.0,     # %
    '养老床位密度':   5.3,     # 床/千名老人
}

# 三个情景下各因素的年均变化量
SCENARIOS = {
    '基准情景': {
        '描述': '维持当前趋势',
        '老龄化率_年增':     0.4,    # 每年增0.4个百分点
        'ln人均GDP_年增':    0.04,   # GDP年增约4%
        '城镇化率_年增':     0.7,    # 每年增0.7个百分点
        '信贷深度_年增':     0.05,   # 温和增长
        '金融业占比_年增':    0.1,    # 温和增长
        '第三产业占比_年增':   0.5,    # 每年增0.5个百分点
        '养老床位密度_年增':   0.2,    # 温和增长
    },
    '乐观情景': {
        '描述': '积极养老金融政策 + 经济稳健增长',
        '老龄化率_年增':     0.35,   # 老龄化略慢 (健康老龄化)
        'ln人均GDP_年增':    0.05,   # GDP年增约5%
        '城镇化率_年增':     1.0,    # 加速城镇化
        '信贷深度_年增':     0.10,   # 金融深化加速
        '金融业占比_年增':    0.2,    # 金融业快速发展
        '第三产业占比_年增':   0.8,    # 服务业加速
        '养老床位密度_年增':   0.1,    # 增速放缓(质量>数量)
    },
    '悲观情景': {
        '描述': '经济下行 + 深度老龄化',
        '老龄化率_年增':     0.5,    # 加速老龄化
        'ln人均GDP_年增':    0.02,   # GDP年增约2%
        '城镇化率_年增':     0.4,    # 城镇化放缓
        '信贷深度_年增':     0.02,   # 信贷收缩
        '金融业占比_年增':    0.0,    # 金融业停滞
        '第三产业占比_年增':   0.3,    # 服务业放缓
        '养老床位密度_年增':   0.3,    # 大量建设但效率低
    },
}

for name, params in SCENARIOS.items():
    print(f"\n{name}: {params['描述']}")
    for k, v in params.items():
        if k != '描述':
            print(f"  {k}: {v:+.2f}")

# ============================================================
# STEP 4: 运行仿真
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: 运行仿真 (2024-2035)")
print("=" * 70)

sim_years = list(range(SIM_START, SIM_END + 1))
results = {}

for scenario_name, params in SCENARIOS.items():
    # 初始状态
    S = float(national_means.iloc[-1])  # 2023年全国均值
    values = INIT_VALUES.copy()
    
    trajectory = {'年份': [2023], '综合得分': [S]}
    
    # 记录各变量轨迹
    var_trajectories = {var: [val] for var, val in values.items()}
    
    for year in sim_years:
        # 更新各驱动因素
        values['老龄化率'] += params['老龄化率_年增']
        values['ln人均GDP'] += params['ln人均GDP_年增']
        values['城镇化率'] += params['城镇化率_年增']
        values['信贷深度'] += params['信贷深度_年增']
        values['金融业占比'] += params['金融业占比_年增']
        values['第三产业占比'] += params['第三产业占比_年增']
        values['养老床位密度'] += params['养老床位密度_年增']
        
        # 计算增量 dS = Σ β_i * dX_i
        dS = 0
        dS += BETA['老龄化率'] * params['老龄化率_年增']
        dS += BETA['老龄化率²'] * (values['老龄化率']**2 - (values['老龄化率'] - params['老龄化率_年增'])**2)
        dS += BETA['ln人均GDP'] * params['ln人均GDP_年增']
        dS += BETA['城镇化率'] * params['城镇化率_年增']
        dS += BETA['信贷深度'] * params['信贷深度_年增']
        dS += BETA['金融业占比'] * params['金融业占比_年增']
        dS += BETA['第三产业占比'] * params['第三产业占比_年增']
        dS += BETA['养老床位密度'] * params['养老床位密度_年增']
        
        # 更新存量 (加上界约束, S不能超过1也不能低于0)
        S = max(0, min(1, S + dS))
        
        trajectory['年份'].append(year)
        trajectory['综合得分'].append(S)
        
        for var, val in values.items():
            var_trajectories[var].append(val)
    
    results[scenario_name] = pd.DataFrame(trajectory)
    results[scenario_name + '_vars'] = pd.DataFrame({
        '年份': [2023] + sim_years,
        **var_trajectories
    })
    
    print(f"\n{scenario_name}:")
    print(f"  2023: S={trajectory['综合得分'][0]:.4f}")
    print(f"  2030: S={trajectory['综合得分'][7]:.4f}")
    print(f"  2035: S={trajectory['综合得分'][-1]:.4f}")
    
    # 检测倒U型拐点
    age_at_end = values['老龄化率']
    turning_point = -BETA['老龄化率'] / (2 * BETA['老龄化率²'])
    if age_at_end > turning_point:
        cross_year = 2023 + int((turning_point - INIT_VALUES['老龄化率']) / params['老龄化率_年增'])
        print(f"  ⚠️ 老龄化率在{cross_year}年前后越过拐点({turning_point:.1f}%)")

# ============================================================
# STEP 5: 可视化
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: 生成图表")
print("=" * 70)

# 历史数据
hist_years = list(national_means.index.astype(int))
hist_values = list(national_means.values)

# --- 图12: 三情景综合得分预测 ---
fig, ax = plt.subplots(figsize=(12, 6))

# 历史
ax.plot(hist_years, hist_values, 'ko-', linewidth=2.5, markersize=7, label='历史值', zorder=5)

# 连接线 (历史到预测)
colors = {'基准情景': '#4A90D9', '乐观情景': '#4AA64A', '悲观情景': '#D94A4A'}
linestyles = {'基准情景': '-', '乐观情景': '--', '悲观情景': '-.'}

for name, df in results.items():
    if '_vars' in name:
        continue
    ax.plot(df['年份'], df['综合得分'], color=colors[name], linestyle=linestyles[name],
            linewidth=2, markersize=5, marker='s', label=name)

# 分隔线
ax.axvline(x=2023.5, color='gray', linestyle=':', alpha=0.5)
ax.text(2021, ax.get_ylim()[1] * 0.95, '历史', fontsize=10, ha='center', color='gray')
ax.text(2026, ax.get_ylim()[1] * 0.95, '预测', fontsize=10, ha='center', color='gray')

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('综合得分 S', fontsize=12)
ax.set_title('养老金融高质量发展情景模拟 (2016-2035)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xticks(list(range(2016, 2036, 2)))
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图12_情景模拟.png'), dpi=300)
print("  ✓ 图12_情景模拟.png")
plt.close()

# --- 图13: 驱动因素变化轨迹 (基准情景) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
var_names_plot = list(INIT_VALUES.keys())
var_colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1', '#6D4C41']

base_vars = results['基准情景_vars']

for i, var in enumerate(var_names_plot):
    ax = axes[i]
    for scenario_name in ['基准情景', '乐观情景', '悲观情景']:
        df = results[scenario_name + '_vars']
        ax.plot(df['年份'], df[var], color=colors[scenario_name],
                linestyle=linestyles[scenario_name], linewidth=1.5)
    
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # 标注拐点 (仅对老龄化率)
    if var == '老龄化率':
        turning_point = -BETA['老龄化率'] / (2 * BETA['老龄化率²'])
        ax.axhline(y=turning_point, color='red', linestyle=':', alpha=0.5)
        ax.text(2035, turning_point, f'拐点{turning_point:.1f}%', fontsize=8, color='red', va='bottom')

# 删除多余子图
if len(var_names_plot) < len(axes):
    for j in range(len(var_names_plot), len(axes)):
        fig.delaxes(axes[j])

# 统一图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors['基准情景'], linestyle='-', label='基准'),
    Line2D([0], [0], color=colors['乐观情景'], linestyle='--', label='乐观'),
    Line2D([0], [0], color=colors['悲观情景'], linestyle='-.', label='悲观'),
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=10, ncol=3)

plt.suptitle('驱动因素变化轨迹 (2023-2035)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图13_驱动因素轨迹.png'), dpi=300, bbox_inches='tight')
print("  ✓ 图13_驱动因素轨迹.png")
plt.close()

# --- 图14: 敏感性分析 (单因素变化对S的边际影响) ---
fig, ax = plt.subplots(figsize=(10, 6))

# 计算每个因素变化1单位对S的边际贡献
marginal = {}
for var in ['老龄化率', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']:
    if var == '老龄化率':
        # 在当前老龄化率水平(15%)的边际效应: β + 2β²*X
        marginal[var] = BETA['老龄化率'] + 2 * BETA['老龄化率²'] * INIT_VALUES['老龄化率']
    else:
        marginal[var] = BETA[var]

# 乘以基准情景下的年变化量 = 年贡献
annual_contrib = {}
base_params = SCENARIOS['基准情景']
param_map = {
    '老龄化率': '老龄化率_年增', 'ln人均GDP': 'ln人均GDP_年增',
    '城镇化率': '城镇化率_年增', '信贷深度': '信贷深度_年增',
    '金融业占比': '金融业占比_年增', '第三产业占比': '第三产业占比_年增',
    '养老床位密度': '养老床位密度_年增'
}

for var in marginal:
    annual_contrib[var] = marginal[var] * base_params[param_map[var]]

# 排序画图
sorted_vars = sorted(annual_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
var_labels = [x[0] for x in sorted_vars]
var_values = [x[1] for x in sorted_vars]
bar_colors = ['#4AA64A' if v > 0 else '#D94A4A' for v in var_values]

bars = ax.barh(range(len(var_labels)), var_values, color=bar_colors, height=0.6)
ax.set_yticks(range(len(var_labels)))
ax.set_yticklabels(var_labels, fontsize=11)
ax.set_xlabel('年贡献量 (基准情景下对综合得分S的年变化贡献)', fontsize=11)
ax.set_title('各驱动因素敏感性分析', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

# 标注数值
for bar, val in zip(bars, var_values):
    ax.text(val + (0.0002 if val > 0 else -0.0002), bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图14_敏感性分析.png'), dpi=300)
print("  ✓ 图14_敏感性分析.png")
plt.close()

# ============================================================
# STEP 6: 输出
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 输出结果")
print("=" * 70)

output_path = os.path.join(OUTPUT_DIR, "SD仿真结果.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 三情景预测值
    for name in ['基准情景', '乐观情景', '悲观情景']:
        results[name].to_excel(writer, sheet_name=name, index=False)
    
    # 三情景驱动因素轨迹
    for name in ['基准情景', '乐观情景', '悲观情景']:
        results[name + '_vars'].to_excel(writer, sheet_name=f'{name}_变量', index=False)
    
    # 情景假设汇总
    scenario_df = pd.DataFrame(SCENARIOS).T
    scenario_df.to_excel(writer, sheet_name='情景假设')
    
    # 敏感性分析
    sens_df = pd.DataFrame({
        '变量': var_labels,
        '边际效应': [marginal[v] for v in var_labels],
        '年变化量(基准)': [base_params[param_map[v]] for v in var_labels],
        '年贡献': var_values,
    })
    sens_df.to_excel(writer, sheet_name='敏感性分析', index=False)

print(f"  ✓ {output_path}")

# ============================================================
# STEP 7: 结论要点
# ============================================================
print("\n" + "=" * 70)



print(f"""
SD模型核心结论:

1. 情景预测:
   基准情景: S从{results['基准情景']['综合得分'].iloc[0]:.3f}(2023)升至{results['基准情景']['综合得分'].iloc[-1]:.3f}(2035)
   乐观情景: S升至{results['乐观情景']['综合得分'].iloc[-1]:.3f}(2035)
   悲观情景: S变为{results['悲观情景']['综合得分'].iloc[-1]:.3f}(2035)

2. 倒U型拐点:
   老龄化率拐点={-BETA['老龄化率']/(2*BETA['老龄化率²']):.1f}%
   基准情景下约在{2023 + int((-BETA['老龄化率']/(2*BETA['老龄化率²']) - INIT_VALUES['老龄化率']) / SCENARIOS['基准情景']['老龄化率_年增'])}年越过拐点

3. 敏感性排序 (对综合得分年贡献):
""")

for i, (var, val) in enumerate(sorted_vars):
    direction = "促进" if val > 0 else "抑制"
    print(f"   {i+1}. {var}: {direction} ({val:+.4f}/年)")

print(f"""
4. 政策建议方向:
   - 经济发展(人均GDP)是最大正向驱动力, 保持经济增长是基础
   - 金融深化(信贷深度)效果显著, 应扩大养老金融产品供给
   - 服务业发展(第三产业占比)有稳定正效应
   - 老龄化存在拐点, 深度老龄化省份需重点制度创新
   - 养老床位{'"' if BETA['养老床位密度'] < 0 else ''}建设应注重质量而非数量
""")

print("全部完成!")
