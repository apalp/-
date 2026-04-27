import pandas as pd
import numpy as np
import os

from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. 配置与路径
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\新增回归变量\熵权赋值"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

DIM_MAP = {
    'Aging': [
        (r"D:\用户\Desktop\topsis\回归数据处理\01六十五岁以上人口占比.xlsx", '老龄化率'),
        (r"D:\用户\Desktop\topsis\回归数据处理\02老年人口抚养比.xlsx",       '抚养比'),
    ],
    'Econ': [
        (r"D:\用户\Desktop\topsis\回归数据处理\03人均地区生产总值.xlsx",     '人均GDP'),
        (r"D:\用户\Desktop\topsis\回归数据处理\05城镇化率.xlsx",             '城镇化率'),
    ],
    'Finance': [
        (r"D:\用户\Desktop\topsis\回归数据处理\06金融机构人民币贷款余额除以地区生产总值.xlsx", '信贷深度'),
        (r"D:\用户\Desktop\topsis\回归数据处理\07金融业增加值占GDP比重.xlsx",                 '金融业占比'),
    ],
    'Service': [
        (r"D:\用户\Desktop\topsis\回归数据处理\08第三产业增加值占GDP比重.xlsx", '第三产业占比'),
        (r"D:\用户\Desktop\topsis\回归数据处理\09每千位老人床位数.xlsx",        '养老床位密度'),
    ],
}

NAME_MAP = {
    '北京市': '北京', '天津市': '天津', '河北省': '河北', '山西省': '山西',
    '内蒙古自治区': '内蒙古', '辽宁省': '辽宁', '吉林省': '吉林', '黑龙江省': '黑龙江',
    '上海市': '上海', '江苏省': '江苏', '浙江省': '浙江', '安徽省': '安徽',
    '福建省': '福建', '江西省': '江西', '山东省': '山东', '河南省': '河南',
    '湖北省': '湖北', '湖南省': '湖南', '广东省': '广东', '广西壮族自治区': '广西',
    '海南省': '海南', '重庆市': '重庆', '四川省': '四川', '贵州省': '贵州',
    '云南省': '云南', '西藏自治区': '西藏', '陕西省': '陕西', '甘肃省': '甘肃',
    '青海省': '青海', '宁夏回族自治区': '宁夏', '新疆维吾尔自治区': '新疆',
}

# ============================================================
# 2. 数据读取
# ============================================================
def clean_name(name):
    return NAME_MAP.get(str(name).strip(), str(name).strip())

def read_var(file_path, var_name):
    df = pd.read_excel(file_path)
    df = df.rename(columns={df.columns[0]: '地区'})
    df['地区'] = df['地区'].apply(clean_name)
    cols = {c: int(float(str(c).replace('年','').replace('.0',''))) for c in df.columns[1:]}
    df = df[['地区'] + list(cols.keys())].rename(columns=cols)
    return df.melt(id_vars='地区', var_name='年份', value_name=var_name)

# ============================================================
# 3. 数据整合
# ============================================================
print("=" * 60)
print("【步骤1】读取面板数据")
print("=" * 60)

panel = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')[['地区', '年份', 'PCA综合得分']]

for dim_name, vars_info in DIM_MAP.items():
    for file, v_name in vars_info:
        v_df = read_var(file, v_name)
        panel = panel.merge(v_df, on=['地区', '年份'], how='inner')

# 人均GDP取对数
panel['ln人均GDP'] = np.log(panel['人均GDP'].clip(lower=1))

print(f"  面板: {len(panel)} 行, {panel['地区'].nunique()} 省, {panel['年份'].nunique()} 年")

# ============================================================
# 4. 双向固定效应回归（7个指标）
# ============================================================
print("\n" + "=" * 60)
print("【步骤2】双向固定效应回归")
print("=" * 60)

dep_var = 'PCA综合得分'
indep_vars = ['老龄化率', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']
# 注意：抚养比和老龄化率高度相关，放一起会有共线性问题
#
#indep_vars = ['老龄化率', '抚养比', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']

panel_idx = panel.set_index(['地区', '年份'])

fe2_res = PanelOLS(
    panel_idx[dep_var],
    panel_idx[indep_vars],
    entity_effects=True,
    time_effects=True
).fit(cov_type='clustered', cluster_entity=True)

print(f"\n  R²(overall)  = {fe2_res.rsquared:.4f}")
print(f"  R²(within)   = {fe2_res.rsquared_within:.4f}")
print(f"  观测数        = {fe2_res.nobs}")

print(f"\n  {'变量':12s}  {'系数':>10s}  {'标准误':>10s}  {'t值':>8s}  {'p值':>8s}  显著性")
print(f"  {'-'*65}")
for v in indep_vars:
    sig = '***' if fe2_res.pvalues[v]<0.01 else ('**' if fe2_res.pvalues[v]<0.05 else ('*' if fe2_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}  {fe2_res.params[v]:>+10.4f}  {fe2_res.std_errors[v]:>10.4f}  {fe2_res.tstats[v]:>8.2f}  {fe2_res.pvalues[v]:>8.4f}  {sig}")


# ============================================================
# 输出Excel
# ============================================================
print("\n" + "=" * 60)
print("【步骤3】输出Excel")
print("=" * 60)

out_file = os.path.join(OUTPUT_DIR, "7指标_双向固定效应.xlsx")

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:

    # Sheet1: 双向FE回归结果
    reg_df = pd.DataFrame({
        '变量': fe2_res.params.index,
        '系数': fe2_res.params.values,
        '标准误': fe2_res.std_errors.values,
        't值': fe2_res.tstats.values,
        'p值': fe2_res.pvalues.values,
    })
    reg_df['显著性'] = reg_df['p值'].apply(
        lambda p: '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))
    reg_df.to_excel(writer, sheet_name='双向FE回归结果', index=False)

    
    # Sheet3: 模型摘要
    summary = [
        ['',  '双向FE'],
        ['R²(within)', fe2_res.rsquared_within],
        ['R²(overall)',  fe2_res.rsquared],
        ['观测数',  fe2_res.nobs],
        ['个体固定效应',  '是'],
        ['时间固定效应',  '是'],
        ['标准误',  '聚类稳健'],
    ]
    pd.DataFrame(summary[1:], columns=summary[0]).to_excel(writer, sheet_name='模型对比', index=False)

    # Sheet4: 面板数据
    out_cols = ['地区', '年份', 'PCA综合得分'] + indep_vars
    panel[out_cols].to_excel(writer, sheet_name='面板数据', index=False)

print(f"  ✓ 输出: {out_file}")
print("\n完成！")
