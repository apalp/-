"""
============================================================
回归优化
  Part 1: 加老龄化率平方项 (检验倒U型关系)
  Part 2: 空间杜宾模型 SDM (空间溢出效应)
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\标准化处理\权重限定百分之12\养老金融高质量发展_TOPSIS评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\回归数据处理\老龄化率加平方"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# 解释变量文件 
EXPLANATORY_VARS = [
    # ---- 人口老龄化 ----
    (r"D:\用户\Desktop\回归数据处理\01六十五岁以上人口占比.xlsx",   "老龄化率"),
    
    # ---- 经济基础 ----
    (r"D:\用户\Desktop\回归数据处理\03人均地区生产总值.xlsx",              "人均GDP"),
    
    (r"D:\用户\Desktop\回归数据处理\05城镇化率.xlsx",             "城镇化率"),
    # ---- 金融生态 ----
    (r"D:\用户\Desktop\回归数据处理\06金融机构人民币贷款余额除以地区生产总值.xlsx",         "信贷深度"),
    (r"D:\用户\Desktop\回归数据处理\07金融业增加值占GDP比重.xlsx",      "金融业占比"),
    # ---- 服务产业 ----
    (r"D:\用户\Desktop\回归数据处理\08第三产业增加值占GDP比重.xlsx",         "第三产业占比"),
    (r"D:\用户\Desktop\回归数据处理\09每千位老人床位数.xlsx",     "养老床位密度"),
]


# 31省邻接矩阵
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

def read_wide_table(filepath, var_name):
    df = pd.read_excel(filepath, sheet_name=0)
    region_col = df.columns[0]
    df = df.rename(columns={region_col: '地区'})
    df['地区'] = df['地区'].apply(clean_name)
    year_cols = {}
    for col in df.columns[1:]:
        col_str = str(col).strip().replace('年', '').replace('.0', '')
        try:
            year = int(float(col_str))
            if 2000 <= year <= 2030:
                year_cols[col] = year
        except:
            continue
    keep_cols = {orig: yr for orig, yr in year_cols.items() if yr in STUDY_YEARS}
    df_subset = df[['地区'] + list(keep_cols.keys())].copy()
    df_subset = df_subset.rename(columns=keep_cols)
    long = df_subset.melt(id_vars='地区', var_name='年份', value_name=var_name)
    long['年份'] = long['年份'].astype(int)
    return long

# ============================================================
# 数据准备
# ============================================================
print("=" * 70)
print("数据准备")
print("=" * 70)

# 读取解释变量
all_vars = []
var_names = []
for filepath, var_name in EXPLANATORY_VARS:
    long = read_wide_table(filepath, var_name)
    all_vars.append(long)
    var_names.append(var_name)
    print(f"  ✓ {var_name}")

exog = all_vars[0][['地区', '年份', var_names[0]]].copy()
for i in range(1, len(all_vars)):
    exog = exog.merge(all_vars[i][['地区', '年份', var_names[i]]], on=['地区', '年份'], how='outer')

# 读取被解释变量
topsis = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')
panel = topsis[['地区', '年份', '综合得分']].merge(exog, on=['地区', '年份'], how='inner')

# 预处理
panel['ln人均GDP'] = np.log(panel['人均GDP'])
panel['老龄化率²'] = panel['老龄化率'] ** 2

print(f"面板: {panel.shape[0]}行, {panel['地区'].nunique()}省 x {panel['年份'].nunique()}年")

# ============================================================
# PART 1: 固定效应 + 老龄化率平方项 (倒U型检验)
# ============================================================
print("\n" + "=" * 70)
print("PART 1: 固定效应模型 + 老龄化率平方项")
print("=" * 70)

from linearmodels.panel import PanelOLS

# 模型1: 基准 (无平方项)
var_list_base = ['老龄化率', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']

# 模型2: 加平方项
var_list_sq = ['老龄化率', '老龄化率²', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']

panel_idx = panel.set_index(['地区', '年份'])

print("\n--- 模型1: 基准固定效应 ---")
fe1 = PanelOLS(panel_idx['综合得分'], panel_idx[var_list_base],
               entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(fe1.summary.tables[1])
print(f"R²(within) = {fe1.rsquared_within:.4f}")

print("\n--- 模型2: 加老龄化率² (倒U型检验) ---")
fe2 = PanelOLS(panel_idx['综合得分'], panel_idx[var_list_sq],
               entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(fe2.summary.tables[1])
print(f"R²(within) = {fe2.rsquared_within:.4f}")

# 判断倒U型
coef_age = fe2.params.get('老龄化率', 0)
coef_age2 = fe2.params.get('老龄化率²', 0)
p_age = fe2.pvalues.get('老龄化率', 1)
p_age2 = fe2.pvalues.get('老龄化率²', 1)

print(f"\n倒U型检验:")
print(f"  老龄化率:  β={coef_age:.6f}, p={p_age:.4f}")
print(f"  老龄化率²: β={coef_age2:.6f}, p={p_age2:.4f}")

if coef_age > 0 and coef_age2 < 0 and p_age < 0.1 and p_age2 < 0.1:
    turning_point = -coef_age / (2 * coef_age2)
    print(f"  → ✓ 存在倒U型关系!")
    print(f"  → 拐点: 老龄化率 = {turning_point:.2f}%")
    print(f"  → 含义: 老龄化率低于{turning_point:.1f}%时促进养老金融, 超过后抑制")
elif coef_age < 0 and coef_age2 > 0 and p_age < 0.1 and p_age2 < 0.1:
    turning_point = -coef_age / (2 * coef_age2)
    print(f"  → ✓ 存在U型关系!")
    print(f"  → 拐点: 老龄化率 = {turning_point:.2f}%")
else:
    print(f"  → 未检测到显著的倒U型/U型关系")
    print(f"  → 老龄化率与养老金融的关系更可能是线性的")

# ============================================================
# PART 2: 空间杜宾模型 (SDM)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: 空间杜宾模型 (SDM)")
print("=" * 70)

# --- 2.1 构建空间权重矩阵 W ---
provinces = sorted(panel['地区'].unique())
n_prov = len(provinces)
prov_idx = {p: i for i, p in enumerate(provinces)}

W = np.zeros((n_prov, n_prov))
for prov, neighbors in ADJACENCY.items():
    if prov in prov_idx:
        i = prov_idx[prov]
        for nb in neighbors:
            if nb in prov_idx:
                j = prov_idx[nb]
                W[i, j] = 1

# 行标准化
row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
W_norm = W / row_sums

print(f"空间权重矩阵: {n_prov}x{n_prov}, 非零元素={int(W.sum())}")

# --- 2.2 计算空间滞后变量 WX 和 Wy ---
var_list_sdm = ['老龄化率', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']

for year in STUDY_YEARS:
    year_mask = panel['年份'] == year
    year_data = panel[year_mask].copy()

    # 按省份顺序排列
    year_data = year_data.set_index('地区').loc[provinces].reset_index()

    # Wy: 被解释变量的空间滞后
    y_vec = year_data['综合得分'].values
    Wy = W_norm @ y_vec
    panel.loc[year_mask, 'W_综合得分'] = panel.loc[year_mask, '地区'].map(
        dict(zip(provinces, Wy))
    )

    # WX: 解释变量的空间滞后
    for var in var_list_sdm:
        x_vec = year_data[var].values
        Wx = W_norm @ x_vec
        panel.loc[year_mask, f'W_{var}'] = panel.loc[year_mask, '地区'].map(
            dict(zip(provinces, Wx))
        )

print("空间滞后变量计算完成")

# --- 2.3 估计SDM (用ML近似: 两阶段法) ---
# SDM: y = ρWy + Xβ + WXθ + ε
# 由于Python没有成熟的空间面板ML估计, 我们用以下方法:
# 方法A: 固定效应 + 空间滞后项作为额外解释变量 (S2SLS近似)
# 这在文献中也被广泛使用

print("\n--- SDM估计 (固定效应 + 空间滞后) ---")

# 构建完整变量列表: X + WX + Wy
sdm_vars = var_list_sdm.copy()
sdm_vars.append('W_综合得分')  # ρ: 空间自回归项
for var in var_list_sdm:
    sdm_vars.append(f'W_{var}')  # θ: 空间滞后解释变量

panel_sdm = panel.set_index(['地区', '年份'])

# 检查缺失
if panel_sdm[sdm_vars].isnull().any().any():
    print("  ⚠️ 存在缺失值, 删除缺失行")
    panel_sdm = panel_sdm.dropna(subset=sdm_vars + ['综合得分'])

sdm_result = PanelOLS(panel_sdm['综合得分'], panel_sdm[sdm_vars],
                      entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(sdm_result.summary.tables[1])
print(f"\nR²(within) = {sdm_result.rsquared_within:.4f}")

# --- 2.4 解读SDM结果 ---
print("\n" + "=" * 70)
print("SDM结果解读")
print("=" * 70)

rho = sdm_result.params.get('W_综合得分', 0)
rho_p = sdm_result.pvalues.get('W_综合得分', 1)

print(f"\n空间自回归系数 ρ = {rho:.4f}, p = {rho_p:.4f}", end="")
if rho_p < 0.05:
    print(f" ***")
    if rho > 0:
        print(f"  → 正向空间溢出: 邻居省份养老金融发展水平提高, 本省也会提高")
    else:
        print(f"  → 负向空间效应: 存在竞争/虹吸效应")
else:
    print(f" (不显著)")

print(f"\n{'变量':<16} {'直接效应β':>10} {'空间溢出θ':>10} {'β显著':>6} {'θ显著':>6}")
print("-" * 56)
for var in var_list_sdm:
    beta = sdm_result.params.get(var, 0)
    theta = sdm_result.params.get(f'W_{var}', 0)
    p_beta = sdm_result.pvalues.get(var, 1)
    p_theta = sdm_result.pvalues.get(f'W_{var}', 1)
    sig_b = '***' if p_beta < 0.01 else '**' if p_beta < 0.05 else '*' if p_beta < 0.1 else ''
    sig_t = '***' if p_theta < 0.01 else '**' if p_theta < 0.05 else '*' if p_theta < 0.1 else ''
    print(f"{var:<14} {beta:>+10.4f} {theta:>+10.4f} {sig_b:>6} {sig_t:>6}")

# ============================================================
# 输出
# ============================================================
print("\n" + "=" * 70)
print("输出结果")
print("=" * 70)

output_path = os.path.join(OUTPUT_DIR, "回归优化结果.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 倒U型检验
    utest = pd.DataFrame({
        '变量': fe2.params.index,
        '系数': fe2.params.values.round(6),
        'p值': fe2.pvalues.values.round(4),
    })
    utest.to_excel(writer, sheet_name='倒U型检验(FE)', index=False)

    # SDM结果
    sdm_df = pd.DataFrame({
        '变量': sdm_result.params.index,
        '系数': sdm_result.params.values.round(6),
        '标准误': sdm_result.std_errors.values.round(6),
        't值': sdm_result.tstats.values.round(4),
        'p值': sdm_result.pvalues.values.round(4),
    })
    sdm_df.to_excel(writer, sheet_name='SDM结果', index=False)

    # 空间权重矩阵
    w_df = pd.DataFrame(W_norm, index=provinces, columns=provinces).round(4)
    w_df.to_excel(writer, sheet_name='空间权重矩阵W')

    # 完整面板(含空间滞后)
    panel.to_excel(writer, sheet_name='完整数据', index=False)

print(f"  ✓ {output_path}")


print("完成!")
