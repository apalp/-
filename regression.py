

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================

# ============================================================

# 被解释变量来源
TOPSIS_PATH = r"D:\用户\Desktop\标准化处理\权重限定百分之12\养老金融高质量发展_TOPSIS评价结果.xlsx"
# 解释变量文件列表: 

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

OUTPUT_DIR = r"D:\用户\Desktop\回归数据处理\删掉高度共线的变量"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

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

def read_wide_table(filepath, var_name):
    """
    通用读取: 自动识别第一个sheet, 自动处理年份格式("2016年"/2016/2016.0)
    返回长面板: 地区, 年份, var_name
    """
    df = pd.read_excel(filepath, sheet_name=0)

    # 第一列当作地区
    region_col = df.columns[0]
    df = df.rename(columns={region_col: '地区'})
    df['地区'] = df['地区'].apply(clean_name)

    # 其余列当作年份, 清洗列名
    year_cols = {}
    for col in df.columns[1:]:
        col_str = str(col).strip().replace('年', '').replace('.0', '')
        try:
            year = int(float(col_str))
            if 2000 <= year <= 2030:
                year_cols[col] = year
        except:
            continue

    if not year_cols:
        print(f"  ⚠️ {filepath}: 未识别到年份列!")
        return None

    # 只保留研究期内的年份
    keep_cols = {orig: yr for orig, yr in year_cols.items() if yr in STUDY_YEARS}
    df_subset = df[['地区'] + list(keep_cols.keys())].copy()
    df_subset = df_subset.rename(columns=keep_cols)

    # 转长面板
    long = df_subset.melt(id_vars='地区', var_name='年份', value_name=var_name)
    long['年份'] = long['年份'].astype(int)

    return long

# ============================================================
# STEP 1: 读取并合并解释变量
# ============================================================
print("=" * 70)
print("STEP 1: 读取解释变量")
print("=" * 70)

all_vars = []
var_names = []

for filepath, var_name in EXPLANATORY_VARS:
    long = read_wide_table(filepath, var_name)
    if long is not None:
        all_vars.append(long)
        var_names.append(var_name)
        print(f"  ✓ {var_name}: {len(long)}行, "
              f"范围[{long[var_name].min():.4f}, {long[var_name].max():.4f}]")
    else:
        print(f"  ✗ {var_name}: 读取失败!")

# 合并所有解释变量
exog_panel = all_vars[0][['地区', '年份', var_names[0]]].copy()
for i in range(1, len(all_vars)):
    exog_panel = exog_panel.merge(
        all_vars[i][['地区', '年份', var_names[i]]],
        on=['地区', '年份'], how='outer'
    )

print(f"\n解释变量面板: {exog_panel.shape[0]}行 x {exog_panel.shape[1]}列")

# ============================================================
# STEP 2: 合并被解释变量 (综合得分)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 合并被解释变量")
print("=" * 70)

topsis = pd.read_excel(TOPSIS_PATH, sheet_name='完整面板数据')
topsis = topsis[['地区', '年份', '综合得分']].copy()

reg_panel = topsis.merge(exog_panel, on=['地区', '年份'], how='inner')

print(f"回归面板: {reg_panel.shape[0]}行 x {reg_panel.shape[1]}列")
print(f"省份: {reg_panel['地区'].nunique()}, 年份: {reg_panel['年份'].nunique()}")

# 检查缺失
missing = reg_panel.isnull().sum()
if missing.sum() > 0:
    print(f"\n⚠️ 缺失值:")
    print(missing[missing > 0])
    print(f"\n缺失行数: {reg_panel.isnull().any(axis=1).sum()}")
    # 删除缺失行
    reg_panel = reg_panel.dropna()
    print(f"删除后: {reg_panel.shape[0]}行")
else:
    print("✓ 无缺失值")

# ============================================================
# STEP 3: 变量预处理
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: 变量预处理")
print("=" * 70)

# 人均GDP取对数
if '人均GDP' in var_names:
    reg_panel['ln人均GDP'] = np.log(reg_panel['人均GDP'])
    var_names = [v if v != '人均GDP' else 'ln人均GDP' for v in var_names]
    print("  ✓ 人均GDP → ln人均GDP (取对数)")

# 人均可支配收入取对数
if '人均可支配收入' in var_names:
    reg_panel['ln人均可支配收入'] = np.log(reg_panel['人均可支配收入'])
    var_names = [v if v != '人均可支配收入' else 'ln人均可支配收入' for v in var_names]
    print("  ✓ 人均可支配收入 → ln人均可支配收入 (取对数)")

# 描述性统计
print("\n描述性统计:")
desc_cols = ['综合得分'] + var_names
print(reg_panel[desc_cols].describe().round(4).to_string())

# 相关系数矩阵
print("\n相关系数矩阵 (与综合得分):")
for v in var_names:
    corr = reg_panel['综合得分'].corr(reg_panel[v])
    print(f"  {v:<16}: r = {corr:>+.4f}")

# ============================================================
# STEP 4: 回归分析
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: 回归分析")
print("=" * 70)

try:
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("  ⚠️ 未安装 linearmodels, 尝试: pip install linearmodels")
    print("  将使用 statsmodels 替代 (功能有限)")

import statsmodels.api as sm

# --- 4.1 混合OLS ---
print("\n--- 混合OLS回归 ---")
X = reg_panel[var_names].copy()
X = sm.add_constant(X)
y = reg_panel['综合得分']

ols_result = sm.OLS(y, X).fit(cov_type='HC1')  # 异方差稳健标准误
print(ols_result.summary().tables[1])
print(f"\nR² = {ols_result.rsquared:.4f}, Adj-R² = {ols_result.rsquared_adj:.4f}")

if HAS_LINEARMODELS:
    # 设置面板索引
    reg_panel = reg_panel.set_index(['地区', '年份'])

    # --- 4.2 固定效应模型 (FE) ---
    print("\n--- 固定效应模型 (FE) ---")
    fe_model = PanelOLS(reg_panel['综合得分'],
                        reg_panel[var_names],
                        entity_effects=True,
                        time_effects=False)
    fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)
    print(fe_result.summary.tables[1])
    print(f"\nR² (within) = {fe_result.rsquared_within:.4f}")

    # --- 4.3 随机效应模型 (RE) ---
    print("\n--- 随机效应模型 (RE) ---")
    re_model = RandomEffects(reg_panel['综合得分'],
                             sm.add_constant(reg_panel[var_names]))
    re_result = re_model.fit(cov_type='clustered', cluster_entity=True)
    print(re_result.summary.tables[1])
    print(f"\nR² = {re_result.rsquared:.4f}")

    # --- 4.4 Hausman检验 ---
    print("\n--- Hausman检验 (FE vs RE) ---")
    # 手动Hausman检验
    b_fe = fe_result.params
    b_re = re_result.params

    # 取共同变量
    common_vars = [v for v in b_fe.index if v in b_re.index]
    b_diff = b_fe[common_vars] - b_re[common_vars]

    cov_fe = fe_result.cov[common_vars].loc[common_vars]
    cov_re = re_result.cov[common_vars].loc[common_vars]
    cov_diff = cov_fe - cov_re

    try:
        chi2 = float(b_diff.T @ np.linalg.inv(cov_diff) @ b_diff)
        df = len(common_vars)
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi2, df)
        print(f"  χ² = {chi2:.4f}, df = {df}, p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → p < 0.05, 拒绝H0, 应选择【固定效应模型】")
            preferred = "FE"
        else:
            print(f"  → p > 0.05, 不拒绝H0, 应选择【随机效应模型】")
            preferred = "RE"
    except:
        print("  Hausman检验计算失败 (可能因协方差矩阵奇异)")
        preferred = "FE"

    reg_panel = reg_panel.reset_index()

else:
    # 没有linearmodels时, 用statsmodels做简易固定效应
    print("\n--- 简易固定效应 (LSDV, 加省份虚拟变量) ---")
    dummies = pd.get_dummies(reg_panel['地区'], prefix='D', drop_first=True)
    X_fe = pd.concat([reg_panel[var_names], dummies], axis=1)
    X_fe = sm.add_constant(X_fe)
    fe_result = sm.OLS(y, X_fe).fit(cov_type='HC1')
    print("解释变量系数:")
    for v in var_names:
        coef = fe_result.params[v]
        pval = fe_result.pvalues[v]
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {v:<16}: β = {coef:>+.6f}, p = {pval:.4f} {sig}")
    print(f"\nR² = {fe_result.rsquared:.4f}")
    preferred = "FE"

# ============================================================
# STEP 5: 输出结果
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: 输出结果")
print("=" * 70)

output_path = os.path.join(OUTPUT_DIR, "回归分析结果.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 回归面板数据
    if isinstance(reg_panel.index, pd.MultiIndex):
        reg_panel.reset_index().to_excel(writer, sheet_name='回归面板数据', index=False)
    else:
        reg_panel.to_excel(writer, sheet_name='回归面板数据', index=False)

    # 描述性统计
    desc = reg_panel[['综合得分'] + var_names].describe().round(4)
    desc.to_excel(writer, sheet_name='描述性统计')

    # 相关系数矩阵
    corr_matrix = reg_panel[['综合得分'] + var_names].corr().round(4)
    corr_matrix.to_excel(writer, sheet_name='相关系数矩阵')

    # OLS回归结果
    ols_df = pd.DataFrame({
        '变量': ols_result.params.index,
        '系数': ols_result.params.values.round(6),
        '标准误': ols_result.bse.values.round(6),
        't值': ols_result.tvalues.values.round(4),
        'p值': ols_result.pvalues.values.round(4),
    })
    ols_df.to_excel(writer, sheet_name='混合OLS结果', index=False)

print(f"  ✓ {output_path}")

# ============================================================
# STEP 6: 结果摘要
# ============================================================
print("\n" + "=" * 70)
print("结果摘要 ")
print("=" * 70)

print(f"\n样本: {reg_panel['地区'].nunique()}省 × {reg_panel['年份'].nunique()}年 = {len(reg_panel)}个观测")
print(f"被解释变量: 综合得分S")
print(f"解释变量: {len(var_names)}个")

print(f"\n混合OLS主要发现:")
for v in var_names:
    if v in ols_result.params.index:
        coef = ols_result.params[v]
        pval = ols_result.pvalues[v]
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '不显著'
        direction = '正向' if coef > 0 else '负向'
        print(f"  {v:<16}: {direction} ({coef:>+.6f}) {sig}")

print("\n完成!")
