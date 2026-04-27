import pandas as pd
import numpy as np
import os

from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. 配置与路径
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\新增回归变量\64分"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

DIM_MAP = {
    'Aging': [
        (r"D:\用户\Desktop\topsis\回归数据处理\01六十五岁以上人口占比.xlsx", '占比65'),
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
        (r"D:\用户\Desktop\topsis\回归数据处理\08第三产业增加值占GDP比重.xlsx", '三产占比'),
        (r"D:\用户\Desktop\topsis\回归数据处理\09每千位老人床位数.xlsx",        '养老床位密度'),
    ],
}

BASE_PATH = r"D:\用户\Desktop\pca\新增回归变量\等权重"

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
# 2. 数据读取与清洗
# ============================================================
def clean_name(name):
    return NAME_MAP.get(str(name).strip(), str(name).strip())

def read_var(file_name, var_name):
    path = file_name  # DIM_MAP里已经是完整路径
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: '地区'})
    df['地区'] = df['地区'].apply(clean_name)
    cols = {c: int(float(str(c).replace('年','').replace('.0',''))) for c in df.columns[1:]}
    df = df[['地区'] + list(cols.keys())].rename(columns=cols)
    return df.melt(id_vars='地区', var_name='年份', value_name=var_name)

# ============================================================
# 3. 面板数据整合
# ============================================================
print("=" * 60)
print("【步骤1】读取面板数据并合并各维度原始指标")
print("=" * 60)

panel = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')[['地区', '年份', 'PCA综合得分']]
print(f"  基础面板行数: {len(panel)}")

for dim_name, vars_info in DIM_MAP.items():
    for file, v_name in vars_info:
        v_df = read_var(file, v_name)
        before = len(panel)
        panel = panel.merge(v_df, on=['地区', '年份'], how='inner')
        after = len(panel)
        print(f"  合并 {v_name}: {before} → {after} 行")

print(f"\n  最终面板: {len(panel)} 行, {panel['地区'].nunique()} 省, {panel['年份'].nunique()} 年")

# ============================================================
# 4. 维度合成（标准化 + 等权平均）
# ============================================================
print("\n" + "=" * 60)
print("【步骤2】四维度合成（标准化 + 等权平均）")
print("=" * 60)

scaler = StandardScaler()

# Aging
aging_raw = ['占比65', '抚养比']
panel[['z_占比65', 'z_抚养比']] = scaler.fit_transform(panel[aging_raw])
panel['Aging'] = 0.6 * panel['z_占比65'] + 0.4 * panel['z_抚养比']
print("  ✓ Aging = mean(z(占比65), z(抚养比))")

# Econ - 人均GDP取对数
panel['ln_人均GDP'] = np.log(panel['人均GDP'].clip(lower=1))
econ_raw = ['ln_人均GDP', '城镇化率']
panel[['z_ln人均GDP', 'z_城镇化率']] = scaler.fit_transform(panel[econ_raw])
panel['Econ'] = 0.6 * panel['z_ln人均GDP'] + 0.4 * panel['z_城镇化率']
print("  ✓ Econ = mean(z(ln(人均GDP)), z(城镇化率))")

# Finance
finance_raw = ['信贷深度', '金融业占比']
panel[['z_信贷深度', 'z_金融业占比']] = scaler.fit_transform(panel[finance_raw])
panel['Finance'] =  0.6 * panel['z_信贷深度'] + 0.4 * panel['z_金融业占比']
print("  ✓ Finance = mean(z(信贷深度), z(金融业占比))")

# Service
service_raw = ['三产占比', '养老床位密度']
panel[['z_三产占比', 'z_养老床位']] = scaler.fit_transform(panel[service_raw])
panel['Service'] = 0.6 * panel['z_三产占比'] + 0.4 * panel['z_养老床位']
print("  ✓ Service = mean(z(三产占比), z(养老床位密度))")

# ============================================================
# 先看数据再跑回归
# ============================================================
print("\n" + "=" * 60)
print("【步骤3】数据诊断（定位R²为负的原因）")
print("=" * 60)

dep_var = 'PCA综合得分'
indep_vars = ['Aging', 'Econ', 'Finance', 'Service']

# 5a. 描述性统计
print("\n--- 描述性统计 ---")
print(panel[[dep_var] + indep_vars].describe().round(4).to_string())

# 5b. 相关系数矩阵
print("\n--- 相关系数（Pearson）---")
corr = panel[[dep_var] + indep_vars].corr().round(4)
print(corr.to_string())

# 5c. 组内变异诊断（这是关键！）
print("\n--- 组内变异（去均值后的标准差）---")
print("  说明：双向FE只利用组内变异，如果去均值后变异很小，模型就无法拟合")
demean = panel.copy()
for v in [dep_var] + indep_vars:
    demean[v] = demean.groupby('地区')[v].transform(lambda x: x - x.mean())
    demean[v] = demean.groupby('年份')[v].transform(lambda x: x - x.mean())

for v in [dep_var] + indep_vars:
    print(f"  {v:20s}: 原始std={panel[v].std():.4f}, 组内std={demean[v].std():.4f}, "
          f"比例={demean[v].std()/max(panel[v].std(),1e-10)*100:.1f}%")

# 5d. 组内相关系数
print("\n--- 组内相关系数（去均值后）---")
corr_within = demean[[dep_var] + indep_vars].corr().round(4)
print(corr_within.to_string())

# ============================================================
# 6. 多模型对比（找到能用的模型）
# ============================================================
print("\n" + "=" * 60)
print("【步骤4】多模型对比")
print("=" * 60)

panel_idx = panel.set_index(['地区', '年份'])
results_summary = []

# 模型1: 混合OLS（无固定效应）
print("\n--- 模型1: 混合OLS（基准） ---")
from linearmodels.panel import PooledOLS
import statsmodels.api as sm

pool_X = sm.add_constant(panel_idx[indep_vars])
pool_res = PooledOLS(panel_idx[dep_var], pool_X).fit(cov_type='clustered', cluster_entity=True)
print(f"  R² = {pool_res.rsquared:.4f}")
for v in indep_vars:
    sig = '***' if pool_res.pvalues[v]<0.01 else ('**' if pool_res.pvalues[v]<0.05 else ('*' if pool_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}: β={pool_res.params[v]:+.4f}, t={pool_res.tstats[v]:.2f}, p={pool_res.pvalues[v]:.4f} {sig}")
results_summary.append({
    '模型': '混合OLS', 'R²': pool_res.rsquared,
    **{f'β_{v}': pool_res.params[v] for v in indep_vars},
    **{f'p_{v}': pool_res.pvalues[v] for v in indep_vars},
})

# 模型2: 仅个体固定效应
print("\n--- 模型2: 仅个体固定效应 ---")
fe1_res = PanelOLS(panel_idx[dep_var], panel_idx[indep_vars],
                   entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"  R²={fe1_res.rsquared:.4f}, R²(within)={fe1_res.rsquared_within:.4f}")
for v in indep_vars:
    sig = '***' if fe1_res.pvalues[v]<0.01 else ('**' if fe1_res.pvalues[v]<0.05 else ('*' if fe1_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}: β={fe1_res.params[v]:+.4f}, t={fe1_res.tstats[v]:.2f}, p={fe1_res.pvalues[v]:.4f} {sig}")
results_summary.append({
    '模型': '个体FE', 'R²': fe1_res.rsquared, 'R²(within)': fe1_res.rsquared_within,
    **{f'β_{v}': fe1_res.params[v] for v in indep_vars},
    **{f'p_{v}': fe1_res.pvalues[v] for v in indep_vars},
})

# 模型3: 双向固定效应（你原来跑的）
print("\n--- 模型3: 双向固定效应（个体+时间） ---")
fe2_res = PanelOLS(panel_idx[dep_var], panel_idx[indep_vars],
                   entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"  R²={fe2_res.rsquared:.4f}, R²(within)={fe2_res.rsquared_within:.4f}")
for v in indep_vars:
    sig = '***' if fe2_res.pvalues[v]<0.01 else ('**' if fe2_res.pvalues[v]<0.05 else ('*' if fe2_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}: β={fe2_res.params[v]:+.4f}, t={fe2_res.tstats[v]:.2f}, p={fe2_res.pvalues[v]:.4f} {sig}")
results_summary.append({
    '模型': '双向FE', 'R²': fe2_res.rsquared, 'R²(within)': fe2_res.rsquared_within,
    **{f'β_{v}': fe2_res.params[v] for v in indep_vars},
    **{f'p_{v}': fe2_res.pvalues[v] for v in indep_vars},
})

# 模型4: 个体FE + 年份虚拟变量作为控制变量（更灵活）
print("\n--- 模型4: 个体FE + 年份虚拟变量 ---")
year_dummies = pd.get_dummies(panel['年份'], prefix='Y', drop_first=True).astype(float)
panel_yd = pd.concat([panel, year_dummies], axis=1)
yd_cols = year_dummies.columns.tolist()
panel_yd_idx = panel_yd.set_index(['地区', '年份'])

fe4_res = PanelOLS(panel_yd_idx[dep_var], panel_yd_idx[indep_vars + yd_cols],
                   entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"  R²={fe4_res.rsquared:.4f}, R²(within)={fe4_res.rsquared_within:.4f}")
for v in indep_vars:
    sig = '***' if fe4_res.pvalues[v]<0.01 else ('**' if fe4_res.pvalues[v]<0.05 else ('*' if fe4_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}: β={fe4_res.params[v]:+.4f}, t={fe4_res.tstats[v]:.2f}, p={fe4_res.pvalues[v]:.4f} {sig}")
results_summary.append({
    '模型': '个体FE+年份dummy', 'R²': fe4_res.rsquared, 'R²(within)': fe4_res.rsquared_within,
    **{f'β_{v}': fe4_res.params[v] for v in indep_vars},
    **{f'p_{v}': fe4_res.pvalues[v] for v in indep_vars},
})

# ============================================================
# 7. 选择最佳模型并输出
# ============================================================
print("\n" + "=" * 60)
print("【步骤5】模型对比汇总")
print("=" * 60)
df_compare = pd.DataFrame(results_summary)
print(df_compare[['模型', 'R²', 'R²(within)'] + [f'β_{v}' for v in indep_vars]].to_string(index=False))

# 自动选择：取R²(within)最高且为正的模型
best_idx = 0
best_label = '混合OLS'
for i, row in enumerate(results_summary):
    rw = row.get('R²(within)', row.get('R²', -999))
    if rw > results_summary[best_idx].get('R²(within)', results_summary[best_idx].get('R²', -999)):
        best_idx = i
        best_label = row['模型']
print(f"\n  ★ 推荐模型: {best_label}")

# ============================================================
# 8. 输出 Excel
# ============================================================
print("\n" + "=" * 60)
print("【步骤6】输出Excel报表")
print("=" * 60)

out_file = os.path.join(OUTPUT_DIR, "四维度等权合成回归_诊断版.xlsx")

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:

    # Sheet1: 维度处理思路
    notes_data = [
        ['人口老龄化', 'Aging', '65岁及以上人口占比、老年人口抚养比',
         '两指标Z-score标准化后等权平均'],
        ['经济基础', 'Econ', '人均GDP（取ln）、城镇化率',
         '人均GDP取对数后，两指标Z-score标准化后等权平均'],
        ['金融生态', 'Finance', '金融机构人民币贷款余额/GDP、金融业增加值占GDP比重',
         '两指标Z-score标准化后等权平均'],
        ['服务产业', 'Service', '第三产业增加值占GDP比重、每千名老年人床位数',
         '两指标Z-score标准化后等权平均'],
    ]
    pd.DataFrame(notes_data, columns=['维度','英文','包含指标','处理方法']).to_excel(
        writer, sheet_name='维度处理思路', index=False)

    # Sheet2: 模型对比
    df_compare.to_excel(writer, sheet_name='模型对比', index=False)

    # Sheet3: 各模型详细回归结果
    all_reg = []
    for label, res in [('混合OLS', pool_res), ('个体FE', fe1_res),
                        ('双向FE', fe2_res), ('个体FE+年份dummy', fe4_res)]:
        for v in indep_vars:
            if v in res.params.index:
                all_reg.append({
                    '模型': label, '变量': v,
                    '系数': res.params[v], '标准误': res.std_errors[v],
                    't值': res.tstats[v], 'p值': res.pvalues[v],
                    '显著性': '***' if res.pvalues[v]<0.01 else ('**' if res.pvalues[v]<0.05 else ('*' if res.pvalues[v]<0.1 else ''))
                })
    pd.DataFrame(all_reg).to_excel(writer, sheet_name='各模型回归系数', index=False)

    # Sheet4: 相关系数
    corr.to_excel(writer, sheet_name='相关系数_总体')

    # Sheet5: 组内相关系数
    corr_within.to_excel(writer, sheet_name='相关系数_组内')

    # Sheet6: 描述性统计
    panel[[dep_var] + indep_vars].describe().to_excel(writer, sheet_name='描述性统计')

    # Sheet7: 合成后面板数据
    out_cols = ['地区', '年份', 'PCA综合得分',
                '占比65', '抚养比', 'Aging',
                '人均GDP', 'ln_人均GDP', '城镇化率', 'Econ',
                '信贷深度', '金融业占比', 'Finance',
                '三产占比', '养老床位密度', 'Service']
    panel[out_cols].to_excel(writer, sheet_name='合成后面板数据', index=False)

print(f"  ✓ 输出完成: {out_file}")
print("\n" + "=" * 60)
print("【诊断建议】")
print("=" * 60)
print("""
如果你看到：
  1. 混合OLS的R²很高但个体FE的R²(within)很低或为负
     → 说明维度变量主要解释的是省份间差异（截面差异），
       加入个体FE后截面差异被吸收，组内时序变化解释力不足。
     → 这在学术上是合理的，论文中可报告个体FE结果并讨论。

  2. 个体FE正常但双向FE变负
     → 说明时间FE吸收了太多公共趋势，可改用个体FE即可。

  3. 所有模型R²都很低
     → 可能需要加入更多控制变量，或考虑维度合成方式是否合理。

请根据输出的Excel对比表选择合适的模型！
""")
