"""
============================================================
空间马尔科夫链分析
  1. 传统马尔科夫转移概率矩阵
  2. 空间马尔科夫转移概率矩阵 (条件转移)
输入: 养老金融高质量发展_TOPSIS评价结果.xlsx → "完整面板数据" sheet
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
OUTPUT_DIR = r"D:\用户\Desktop\马尔科夫链"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# 31省份邻接关系 (地理相邻 = 1)
ADJACENCY = {
    '北京': ['天津','河北'],
    '天津': ['北京','河北'],
    '河北': ['北京','天津','山西','河南','山东','内蒙古','辽宁'],
    '山西': ['河北','内蒙古','陕西','河南'],
    '内蒙古': ['河北','山西','陕西','宁夏','甘肃','黑龙江','吉林','辽宁'],
    '辽宁': ['河北','内蒙古','吉林'],
    '吉林': ['辽宁','内蒙古','黑龙江'],
    '黑龙江': ['吉林','内蒙古'],
    '上海': ['江苏','浙江'],
    '江苏': ['上海','浙江','安徽','山东'],
    '浙江': ['上海','江苏','安徽','江西','福建'],
    '安徽': ['江苏','浙江','江西','湖北','河南','山东'],
    '福建': ['浙江','江西','广东'],
    '江西': ['浙江','安徽','福建','湖北','湖南','广东'],
    '山东': ['河北','河南','安徽','江苏'],
    '河南': ['河北','山西','陕西','湖北','安徽','山东'],
    '湖北': ['河南','陕西','重庆','湖南','江西','安徽'],
    '湖南': ['湖北','重庆','贵州','广西','广东','江西'],
    '广东': ['福建','江西','湖南','广西','海南'],
    '广西': ['广东','湖南','贵州','云南'],
    '海南': ['广东'],
    '重庆': ['湖北','陕西','四川','贵州','湖南'],
    '四川': ['重庆','陕西','甘肃','青海','西藏','云南','贵州'],
    '贵州': ['重庆','四川','云南','广西','湖南'],
    '云南': ['四川','西藏','广西','贵州'],
    '西藏': ['新疆','青海','四川','云南'],
    '陕西': ['山西','内蒙古','宁夏','甘肃','四川','重庆','湖北','河南'],
    '甘肃': ['内蒙古','宁夏','青海','新疆','陕西','四川'],
    '青海': ['甘肃','新疆','西藏','四川'],
    '宁夏': ['内蒙古','陕西','甘肃'],
    '新疆': ['西藏','青海','甘肃','内蒙古'],
}

# ============================================================
# 读取数据
# ============================================================
panel = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')
print(f"数据: {panel.shape[0]}行, {panel['地区'].nunique()}省, {panel['年份'].nunique()}年")

# ============================================================
# 1. 状态划分 (四分位数)
# ============================================================
all_scores = panel['综合得分'].values
q25, q50, q75 = np.percentile(all_scores, [25, 50, 75])

def classify(val):
    if val <= q25: return 1
    elif val <= q50: return 2
    elif val <= q75: return 3
    else: return 4

panel['状态'] = panel['综合得分'].apply(classify)
state_labels = {1: '低(L)', 2: '中低(ML)', 3: '中高(MH)', 4: '高(H)'}

print(f"\n状态划分阈值: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")
for s in [1, 2, 3, 4]:
    n = (panel['状态'] == s).sum()
    print(f"  状态{s} {state_labels[s]}: {n}个观测 ({n/len(panel)*100:.1f}%)")

# ============================================================
# 2. 计算空间滞后状态
# ============================================================
def get_spatial_lag_state(province, year, panel_data):
    """计算省份在某年的邻居平均得分, 再分类为状态"""
    neighbors = ADJACENCY.get(province, [])
    if not neighbors:
        return None
    neighbor_scores = panel_data[
        (panel_data['年份'] == year) & (panel_data['地区'].isin(neighbors))
    ]['综合得分'].values
    if len(neighbor_scores) == 0:
        return None
    return classify(neighbor_scores.mean())

# 为每个观测计算邻居状态
panel['邻居状态'] = None
for idx, row in panel.iterrows():
    panel.at[idx, '邻居状态'] = get_spatial_lag_state(row['地区'], row['年份'], panel)

panel['邻居状态'] = panel['邻居状态'].astype('Int64')

print(f"\n邻居状态计算完成, 缺失={panel['邻居状态'].isna().sum()}个")

# ============================================================
# 3. 传统马尔科夫转移概率矩阵
# ============================================================
def compute_transition(panel_data, n_states=4):
    """计算转移概率矩阵"""
    count = np.zeros((n_states, n_states))
    for prov in panel_data['地区'].unique():
        prov_data = panel_data[panel_data['地区'] == prov].sort_values('年份')
        states = prov_data['状态'].values
        for t in range(len(states) - 1):
            count[states[t]-1, states[t+1]-1] += 1
    row_sums = count.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob = count / row_sums
    return count, prob

count_trad, prob_trad = compute_transition(panel)

print("\n" + "=" * 70)
print("传统马尔科夫转移概率矩阵")
print("=" * 70)
print(f"{'t/t+1':>8} {'低(L)':>8} {'中低(ML)':>8} {'中高(MH)':>8} {'高(H)':>8} {'样本':>6}")
print("-" * 48)
for i in range(4):
    n = int(count_trad[i].sum())
    print(f"{state_labels[i+1]:>8} {prob_trad[i,0]:>8.3f} {prob_trad[i,1]:>8.3f} "
          f"{prob_trad[i,2]:>8.3f} {prob_trad[i,3]:>8.3f} {n:>6}")

# 对角线稳定性
diag = [prob_trad[i,i] for i in range(4)]
print(f"\n对角线(维持原状态概率): {[f'{d:.3f}' for d in diag]}")
print(f"平均维持概率: {np.mean(diag):.3f}")

# ============================================================
# 4. 空间马尔科夫转移概率矩阵
# ============================================================
print("\n" + "=" * 70)
print("空间马尔科夫转移概率矩阵 (条件转移)")
print("=" * 70)

spatial_matrices = {}  # {邻居状态: (count, prob)}

for lag_state in [1, 2, 3, 4]:
    count = np.zeros((4, 4))
    for prov in panel['地区'].unique():
        prov_data = panel[panel['地区'] == prov].sort_values('年份')
        states = prov_data['状态'].values
        lag_states = prov_data['邻居状态'].values
        for t in range(len(states) - 1):
            if pd.notna(lag_states[t]) and int(lag_states[t]) == lag_state:
                count[states[t]-1, states[t+1]-1] += 1

    row_sums = count.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob = count / row_sums

    spatial_matrices[lag_state] = (count, prob)

    total_n = int(count.sum())
    print(f"\n邻居状态 = {state_labels[lag_state]} (共{total_n}次转移)")
    print(f"{'t/t+1':>8} {'低(L)':>8} {'中低(ML)':>8} {'中高(MH)':>8} {'高(H)':>8} {'样本':>6}")
    print("-" * 48)
    for i in range(4):
        n = int(count[i].sum())
        if n > 0:
            print(f"{state_labels[i+1]:>8} {prob[i,0]:>8.3f} {prob[i,1]:>8.3f} "
                  f"{prob[i,2]:>8.3f} {prob[i,3]:>8.3f} {n:>6}")
        else:
            print(f"{state_labels[i+1]:>8} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {0:>6}")

# ============================================================
# 5. 空间效应分析
# ============================================================
print("\n" + "=" * 70)
print("空间效应分析")
print("=" * 70)

print("\n【向上跃迁概率对比: 邻居好 vs 邻居差】")
print("(省份处于低水平时, 邻居状态对其向上跃迁的影响)")
print()

for i in range(4):
    print(f"省份处于 {state_labels[i+1]} 时:")
    for lag in [1, 2, 3, 4]:
        _, prob = spatial_matrices[lag]
        n = int(spatial_matrices[lag][0][i].sum())
        if n > 0:
            up_prob = sum(prob[i, j] for j in range(i+1, 4))
            stay = prob[i, i]
            down_prob = sum(prob[i, j] for j in range(0, i))
            print(f"  邻居={state_labels[lag]}: 上升={up_prob:.3f}, "
                  f"维持={stay:.3f}, 下降={down_prob:.3f} (n={n})")
    print()

# ============================================================
# 6. 保存结果
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "马尔科夫链分析结果.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 传统转移矩阵
    trad_df = pd.DataFrame(prob_trad,
                           index=['低(L)', '中低(ML)', '中高(MH)', '高(H)'],
                           columns=['低(L)', '中低(ML)', '中高(MH)', '高(H)'])
    trad_df['样本量'] = count_trad.sum(axis=1).astype(int)
    trad_df.to_excel(writer, sheet_name='传统转移矩阵')

    # 空间转移矩阵
    for lag in [1, 2, 3, 4]:
        count, prob = spatial_matrices[lag]
        sp_df = pd.DataFrame(prob,
                             index=['低(L)', '中低(ML)', '中高(MH)', '高(H)'],
                             columns=['低(L)', '中低(ML)', '中高(MH)', '高(H)'])
        sp_df['样本量'] = count.sum(axis=1).astype(int)
        sp_df.to_excel(writer, sheet_name=f'邻居={state_labels[lag]}')

    # 完整面板(含状态)
    panel[['地区', '年份', '综合得分', '状态', '邻居状态']].to_excel(
        writer, sheet_name='省份状态', index=False)

print(f"\n  ✓ {output_path}")

# ============================================================
# 7. 可视化
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 图10: 传统 + 空间马尔科夫转移矩阵热力图 ---
fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
labels_short = ['L', 'ML', 'MH', 'H']
titles = ['传统转移矩阵', '邻居=低(L)', '邻居=中低(ML)', '邻居=中高(MH)', '邻居=高(H)']

# 传统矩阵
im = axes[0].imshow(prob_trad, cmap='Blues', vmin=0, vmax=1, aspect='equal')
for i in range(4):
    for j in range(4):
        n = int(count_trad[i].sum())
        text = f'{prob_trad[i,j]:.3f}\n({int(count_trad[i,j])})'
        axes[0].text(j, i, text, ha='center', va='center', fontsize=8,
                    color='white' if prob_trad[i,j] > 0.5 else 'black')
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(labels_short)
axes[0].set_yticklabels(labels_short)
axes[0].set_xlabel('t+1', fontsize=10)
axes[0].set_ylabel('t', fontsize=10)
axes[0].set_title(titles[0], fontsize=11, fontweight='bold')

# 空间条件矩阵
for k, lag in enumerate([1, 2, 3, 4]):
    ax = axes[k + 1]
    _, prob = spatial_matrices[lag]
    count_k = spatial_matrices[lag][0]
    im = ax.imshow(prob, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(4):
        for j in range(4):
            n = int(count_k[i].sum())
            if n > 0:
                text = f'{prob[i,j]:.3f}\n({int(count_k[i,j])})'
            else:
                text = '—'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color='white' if prob[i,j] > 0.5 else 'black')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels_short)
    ax.set_yticklabels(labels_short)
    ax.set_xlabel('t+1', fontsize=10)
    ax.set_title(titles[k+1], fontsize=11, fontweight='bold')

plt.suptitle('传统与空间马尔科夫转移概率矩阵', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图10_马尔科夫转移矩阵.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ 图10_马尔科夫转移矩阵.png")

# --- 图11: 各省份状态演变热力图 ---
provinces_order = []
for region in ['东部', '中部', '西部', '东北']:
    region_provs = {
        '东部': ['北京','天津','河北','上海','江苏','浙江','福建','山东','广东','海南'],
        '中部': ['山西','安徽','江西','河南','湖北','湖南'],
        '西部': ['内蒙古','广西','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆'],
        '东北': ['辽宁','吉林','黑龙江'],
    }[region]
    provinces_order.extend(region_provs)

state_matrix = np.zeros((len(provinces_order), len(STUDY_YEARS)))
for i, prov in enumerate(provinces_order):
    for j, year in enumerate(STUDY_YEARS):
        row = panel[(panel['地区'] == prov) & (panel['年份'] == year)]
        if len(row) > 0:
            state_matrix[i, j] = row['状态'].values[0]

fig, ax = plt.subplots(figsize=(10, 12))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#EF5350', '#FF9800', '#42A5F5', '#66BB6A'])
im = ax.imshow(state_matrix, cmap=cmap, aspect='auto', vmin=0.5, vmax=4.5)

ax.set_xticks(range(len(STUDY_YEARS)))
ax.set_xticklabels([str(y) for y in STUDY_YEARS])
ax.set_yticks(range(len(provinces_order)))
ax.set_yticklabels(provinces_order, fontsize=9)

# 区域分隔线
sep_positions = [10, 16, 28]  # 东部10, 中部6, 西部12, 东北3
for pos in sep_positions:
    ax.axhline(y=pos - 0.5, color='black', linewidth=1.5)

# 区域标签
ax.text(-1.5, 4.5, '东部', fontsize=10, fontweight='bold', rotation=90, va='center')
ax.text(-1.5, 13, '中部', fontsize=10, fontweight='bold', rotation=90, va='center')
ax.text(-1.5, 22, '西部', fontsize=10, fontweight='bold', rotation=90, va='center')
ax.text(-1.5, 29, '东北', fontsize=10, fontweight='bold', rotation=90, va='center')

# 在每个格子里写状态
for i in range(len(provinces_order)):
    for j in range(len(STUDY_YEARS)):
        s = int(state_matrix[i, j])
        ax.text(j, i, str(s), ha='center', va='center', fontsize=8,
               color='white' if s in [1, 3] else 'black')

cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], shrink=0.6)
cbar.ax.set_yticklabels(['1-低', '2-中低', '3-中高', '4-高'])

ax.set_title('各省份养老金融发展状态演变 (2016-2023)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图11_省份状态演变热力图.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ 图11_省份状态演变热力图.png")

print("\n完成!")
