"""
============================================================
养老金融高质量发展评价 全流程代码
合并14个指标 → CRITIC-熵值法组合赋权 → TOPSIS综合评价
============================================================
使用方法:
  1. 修改下方 CONFIG 区的文件路径和输出路径
  2. 运行: python3 full_pipeline.py
============================================================
"""

import pandas as pd
import numpy as np
import os

# ============================================================

# ============================================================

# 14个指标文件路径 (只读Sheet1"标准化结果")
# 格式: (文件路径, 指标名称, 一级维度)
INDICATORS = [
    # ---- 养老金金融 (7个) ----
    (r"D:\用户\Desktop\标准化处理\001基本养老保险覆盖率_标准化结果.xlsx",    "基本养老保险覆盖率",       "养老金金融"),
    (r"D:\用户\Desktop\标准化处理\002基本养老保险基金收入强度_标准化结果.xlsx","基本养老保险基金收入强度", "养老金金融"),
    (r"D:\用户\Desktop\标准化处理\003基本养老保险基金可支付月数_标准化结果.xlsx","基本养老保险基金可支付月数","养老金金融"),
    (r"D:\用户\Desktop\标准化处理\004城镇职工养老金水平_标准化结果.xlsx",    "城镇职工养老金水平",       "养老金金融"),
    (r"D:\用户\Desktop\标准化处理\005城乡居民养老金水平_标准化结果.xlsx",    "城乡居民养老金水平",       "养老金金融"),
    (r"D:\用户\Desktop\标准化处理\006企业年金覆盖率_标准化结果.xlsx",       "企业年金覆盖率",           "养老金金融"),
    (r"D:\用户\Desktop\标准化处理\007企业年金基金积累强度_标准化结果.xlsx",   "企业年金基金积累强度",     "养老金金融"),
    # ---- 养老服务金融 (5个) ----
    (r"D:\用户\Desktop\标准化处理\008健康保险密度_标准化结果.xlsx",         "健康保险密度",             "养老服务金融"),
    (r"D:\用户\Desktop\标准化处理\009人寿保险密度_标准化结果.xlsx",         "人寿保险密度",             "养老服务金融"),
    (r"D:\用户\Desktop\标准化处理\010长期护理保险试点覆盖率_标准化结果.xlsx", "长期护理保险试点覆盖率",   "养老服务金融"),
    (r"D:\用户\Desktop\标准化处理\011数字普惠金融综合指数_标准化结果.xlsx",   "数字普惠金融综合指数",     "养老服务金融"),
    (r"D:\用户\Desktop\标准化处理\012全国分省每万人银行网点数_标准化结果.xlsx","银行业金融机构网点密度",   "养老服务金融"),
    # ---- 养老产业金融 (2个) ----
    (r"D:\用户\Desktop\标准化处理\013PPP养老产业投资强度_标准化结果.xlsx",      "养老服务类PPP投资强度",    "养老产业金融"),
    (r"D:\用户\Desktop\标准化处理\014卫生和社会工作固定资产投资增速_标准化结果.xlsx","卫生和社会工作固定资产投资增速","养老产业金融"),
]

# 输出路径
OUTPUT_DIR = r"D:\用户\Desktop\标准化处理\权重限定百分之12"


STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

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

def clean_province_name(name):
    """统一省份名称: 先查映射表, 没有就原样返回"""
    name = str(name).strip()
    if name in NAME_MAP:
        return NAME_MAP[name]
    return name

# ============================================================
# STEP 1: 读取并合并14个指标为长面板
# ============================================================
print("=" * 70)
print("STEP 1: 读取并合并14个指标")
print("=" * 70)

all_long = []  # 收集所有长面板数据

for filepath, ind_name, dimension in INDICATORS:
    df = pd.read_excel(filepath, sheet_name="标准化结果")
    df['地区'] = df['地区'].apply(clean_province_name)

    # 年份列可能是int或str, 统一处理
    year_cols = [c for c in df.columns if c != '地区']
    
    long = df.melt(id_vars='地区', var_name='年份', value_name=ind_name)
    long['年份'] = long['年份'].astype(int)
    long = long[long['年份'].isin(STUDY_YEARS)]

    all_long.append((ind_name, dimension, long))
    print(f"  ✓ {ind_name} ({dimension}): {len(long)}行, "
          f"范围[{long[ind_name].min():.4f}, {long[ind_name].max():.4f}]")

# 合并: 以 地区+年份 为key, 逐个join
panel = all_long[0][2][['地区', '年份', all_long[0][0]]].copy()
for ind_name, _, long_df in all_long[1:]:
    panel = panel.merge(long_df[['地区', '年份', ind_name]], on=['地区', '年份'], how='outer')

# 指标名列表和维度映射
indicator_names = [x[1] for x in INDICATORS]
indicator_dims = {x[1]: x[2] for x in INDICATORS}

print(f"\n合并完成: {panel.shape[0]}行 x {panel.shape[1]}列")
print(f"省份数: {panel['地区'].nunique()}, 年份数: {panel['年份'].nunique()}")

# 检查缺失值
missing = panel[indicator_names].isnull().sum()
if missing.sum() > 0:
    print("\n⚠️ 缺失值:")
    print(missing[missing > 0])
else:
    print("✓ 无缺失值")

# ============================================================
# STEP 2: 熵值法赋权
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 熵值法赋权")
print("=" * 70)

X = panel[indicator_names].values  # (n, m) 矩阵
n, m = X.shape

# 计算比重 p_ij
col_sums = X.sum(axis=0)
P = X / col_sums  # (n, m)

# 计算信息熵 e_j
# 处理 p*ln(p) 中 p=0 的情况
with np.errstate(divide='ignore', invalid='ignore'):
    ln_P = np.where(P > 0, np.log(P), 0)

k = 1.0 / np.log(n)
E = -k * (P * ln_P).sum(axis=0)  # (m,)

# 信息效用值
D = 1 - E

# 熵值法权重
w_entropy = D / D.sum()

print(f"{'指标':<28} {'信息熵e':>8} {'效用值d':>8} {'熵值法权重':>10}")
print("-" * 60)
for j in range(m):
    print(f"{indicator_names[j]:<26} {E[j]:>8.4f} {D[j]:>8.4f} {w_entropy[j]:>10.4f}")

# ============================================================
# STEP 3: CRITIC法赋权
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: CRITIC法赋权")
print("=" * 70)

# 标准差 sigma_j
sigma = X.std(axis=0, ddof=1)  # (m,)

# 相关系数矩阵
corr_matrix = np.corrcoef(X.T)  # (m, m)

# 冲突性: sum(1 - r_jk) for k != j
C = np.zeros(m)
for j in range(m):
    conflict = sum(1 - corr_matrix[j, k] for k in range(m) if k != j)
    C[j] = sigma[j] * conflict

# CRITIC权重
w_critic = C / C.sum()

print(f"{'指标':<28} {'标准差σ':>8} {'信息量C':>10} {'CRITIC权重':>10}")
print("-" * 62)
for j in range(m):
    print(f"{indicator_names[j]:<26} {sigma[j]:>8.4f} {C[j]:>10.4f} {w_critic[j]:>10.4f}")

# ============================================================
# STEP 4: 组合赋权 (乘法归一化)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: 组合赋权 (熵值法 × CRITIC)")
print("=" * 70)

w_combined_raw = w_entropy * w_critic
w_combined = w_combined_raw / w_combined_raw.sum()

# ============================================================
# 权重上限约束 (单个指标不超过15%)
# ============================================================
CAP = 0.12 # 上限

w_capped = w_combined.copy()
for _ in range(100):  # 最多迭代100次，实际几轮就收敛
    exceed = w_capped > CAP
    if not exceed.any():
        break
    surplus = w_capped[exceed].sum() - CAP * exceed.sum()
    w_capped[exceed] = CAP
    # 剩余权重按原比例分配多出来的部分
    remain = ~exceed
    w_capped[remain] = w_capped[remain] + surplus * (w_capped[remain] / w_capped[remain].sum())

w_combined = w_capped

print(f"{'指标':<28} {'熵值法':>8} {'CRITIC':>8} {'组合权重':>8} {'维度':<12}")
print("-" * 72)
for j in range(m):
    dim = indicator_dims[indicator_names[j]]
    print(f"{indicator_names[j]:<26} {w_entropy[j]:>8.4f} {w_critic[j]:>8.4f} "
          f"{w_combined[j]:>8.4f} {dim:<12}")

# 各维度权重汇总
print("\n各维度权重汇总:")
dim_weights = {}
for j in range(m):
    dim = indicator_dims[indicator_names[j]]
    dim_weights[dim] = dim_weights.get(dim, 0) + w_combined[j]
for dim, w in dim_weights.items():
    print(f"  {dim}: {w:.4f} ({w*100:.1f}%)")

# ============================================================
# STEP 5: TOPSIS综合评价
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: TOPSIS综合评价")
print("=" * 70)

# 加权标准化矩阵
V = X * w_combined  # (n, m)

# 正理想解和负理想解 (所有指标都是正向)
V_plus = V.max(axis=0)   # (m,)
V_minus = V.min(axis=0)  # (m,)

# 到正/负理想解的距离
D_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))   # (n,)
D_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))  # (n,)

# 相对贴近度 (综合得分)
S = D_minus / (D_plus + D_minus)  # (n,)

panel['综合得分'] = S

# ============================================================
# STEP 5b: 三个子维度的TOPSIS子指数
# ============================================================
dimensions = ['养老金金融', '养老服务金融', '养老产业金融']

for dim in dimensions:
    dim_indicators = [indicator_names[j] for j in range(m) if indicator_dims[indicator_names[j]] == dim]
    dim_indices = [j for j in range(m) if indicator_dims[indicator_names[j]] == dim]

    # 子维度内权重重新归一化
    dim_w = w_combined[dim_indices]
    dim_w_norm = dim_w / dim_w.sum()

    X_dim = X[:, dim_indices]
    V_dim = X_dim * dim_w_norm

    Vp = V_dim.max(axis=0)
    Vm = V_dim.min(axis=0)

    Dp = np.sqrt(((V_dim - Vp) ** 2).sum(axis=1))
    Dm = np.sqrt(((V_dim - Vm) ** 2).sum(axis=1))

    S_dim = Dm / (Dp + Dm)
    panel[f'{dim}_子指数'] = S_dim

# ============================================================
# STEP 6: 输出结果
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 输出结果")
print("=" * 70)

# --- 结果1: 综合得分宽表 (31省 x 8年) ---
score_wide = panel.pivot(index='地区', columns='年份', values='综合得分').reset_index()
score_wide.columns = ['地区'] + [int(c) for c in score_wide.columns[1:]]

# --- 结果2: 子维度得分宽表 ---
sub_scores = {}
for dim in dimensions:
    sw = panel.pivot(index='地区', columns='年份', values=f'{dim}_子指数').reset_index()
    sw.columns = ['地区'] + [int(c) for c in sw.columns[1:]]
    sub_scores[dim] = sw

# --- 结果3: 权重表 ---
weight_df = pd.DataFrame({
    '指标': indicator_names,
    '维度': [indicator_dims[ind] for ind in indicator_names],
    '熵值法权重': w_entropy,
    'CRITIC权重': w_critic,
    '组合权重': w_combined,
})

# --- 结果4: 完整长面板 ---
panel_sorted = panel.sort_values(['年份', '地区']).reset_index(drop=True)

# 写入Excel
output_path = os.path.join(OUTPUT_DIR, "养老金融高质量发展_TOPSIS评价结果.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    score_wide.to_excel(writer, sheet_name='综合得分', index=False)
    for dim in dimensions:
        sheet_name = dim[:10]  # Excel sheet名最长31字符
        sub_scores[dim].to_excel(writer, sheet_name=f'{sheet_name}_子指数', index=False)
    weight_df.to_excel(writer, sheet_name='权重表', index=False)
    panel_sorted.to_excel(writer, sheet_name='完整面板数据', index=False)

print(f"\n文件已保存: {output_path}")
print(f"  Sheet1: 综合得分 (31省 x 8年)")
print(f"  Sheet2-4: 三个子维度子指数")
print(f"  Sheet5: 权重表 (熵值法/CRITIC/组合)")
print(f"  Sheet6: 完整面板数据 (含所有指标和得分)")

# ============================================================
# STEP 7: 结果预览
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: 结果预览")
print("=" * 70)

# 各年全国均值
print("\n各年全国均值:")
for y in STUDY_YEARS:
    year_data = panel[panel['年份'] == y]
    print(f"  {y}: 综合={year_data['综合得分'].mean():.4f}, "
          f"金金融={year_data['养老金金融_子指数'].mean():.4f}, "
          f"服务金融={year_data['养老服务金融_子指数'].mean():.4f}, "
          f"产业金融={year_data['养老产业金融_子指数'].mean():.4f}")

# 2023年排名 Top10 和 Bottom5
print("\n2023年综合得分排名:")
rank_2023 = panel[panel['年份'] == 2023][['地区', '综合得分']].sort_values('综合得分', ascending=False)
rank_2023['排名'] = range(1, len(rank_2023) + 1)

print("\n  Top 10:")
for _, row in rank_2023.head(10).iterrows():
    print(f"    {row['排名']:>2}. {row['地区']:<6} {row['综合得分']:.4f}")

print("\n  Bottom 5:")
for _, row in rank_2023.tail(5).iterrows():
    print(f"    {row['排名']:>2}. {row['地区']:<6} {row['综合得分']:.4f}")

print("\n" + "=" * 70)
print("全流程完成!")
print("=" * 70)
