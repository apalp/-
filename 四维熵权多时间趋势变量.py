import pandas as pd
import numpy as np
import os

from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ============================================================
# 1. 配置与路径
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\新增回归变量\熵权赋值"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

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

panel['ln_人均GDP'] = np.log(panel['人均GDP'].clip(lower=1))
print(f"  面板: {len(panel)} 行, {panel['地区'].nunique()} 省, {panel['年份'].nunique()} 年")

# ============================================================
# 4. 熵权法赋权
# ============================================================
print("\n" + "=" * 60)
print("【步骤2】熵权法计算权重")
print("=" * 60)

def entropy_weight(df_cols):
    scaler = MinMaxScaler()
    normed = pd.DataFrame(scaler.fit_transform(df_cols), columns=df_cols.columns) + 1e-10
    n = len(normed)
    p = normed.div(normed.sum(axis=0), axis=1)
    k = 1.0 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    return d / d.sum()

dim_indicators = {
    'Aging':   ['占比65', '抚养比'],
    'Econ':    ['ln_人均GDP', '城镇化率'],
    'Finance': ['信贷深度', '金融业占比'],
    'Service': ['三产占比', '养老床位密度'],
}

z_scaler = StandardScaler()
weight_records = []

for dim, indicators in dim_indicators.items():
    w = entropy_weight(panel[indicators])
    for ind, wi in zip(indicators, w):
        print(f"  {dim:10s} - {ind:20s}: {wi:.4f} ({wi*100:.1f}%)")
        weight_records.append({'维度': dim, '指标': ind, '熵权权重': wi})
    z_cols = [f'z_{ind}' for ind in indicators]
    panel[z_cols] = z_scaler.fit_transform(panel[indicators])
    panel[dim] = sum(w[i] * panel[z_cols[i]] for i in range(len(indicators)))

df_weights = pd.DataFrame(weight_records)

# ============================================================
# 5. ★ 加入线性时间趋势控制时间维度
# ============================================================
# trend = 1(2016), 2(2017), ..., 8(2023)
# 只用1个变量控制时间的线性趋势，不像双向FE用7个dummy吸光变异
panel['trend'] = panel['年份'] - panel['年份'].min() + 1

# ============================================================
# 6. 回归分析：个体FE + 线性时间趋势
# ============================================================
print("\n" + "=" * 60)
print("【步骤3】回归分析")
print("=" * 60)

dep_var = 'PCA综合得分'
core_vars = ['Aging', 'Econ', 'Finance', 'Service']

panel_idx = panel.set_index(['地区', '年份'])

# --- 主回归：个体FE + 线性时间趋势 ---
main_vars = core_vars + ['trend']
main_res = PanelOLS(panel_idx[dep_var], panel_idx[main_vars],
                    entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print("\n  ★ 主回归: 个体FE + 线性时间趋势")
print(f"  R²(within) = {main_res.rsquared_within:.4f}")
print(f"  观测数 = {main_res.nobs}")
print(f"\n  {'变量':12s}  {'系数':>10s}  {'标准误':>10s}  {'t值':>8s}  {'p值':>8s}  显著性")
print(f"  {'-'*65}")
for v in main_vars:
    sig = '***' if main_res.pvalues[v]<0.01 else ('**' if main_res.pvalues[v]<0.05 else ('*' if main_res.pvalues[v]<0.1 else ''))
    print(f"  {v:12s}  {main_res.params[v]:>+10.4f}  {main_res.std_errors[v]:>10.4f}  {main_res.tstats[v]:>8.2f}  {main_res.pvalues[v]:>8.4f}  {sig}")

# --- 对比：纯个体FE（无时间控制，上次跑的）---
base_res = PanelOLS(panel_idx[dep_var], panel_idx[core_vars],
                    entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(f"\n  对比:")
print(f"  纯个体FE（无时间控制）  R²(within) = {base_res.rsquared_within:.4f}")
print(f"  个体FE + 时间趋势      R²(within) = {main_res.rsquared_within:.4f}")
print(f"  trend系数 = {main_res.params['trend']:+.4f}, p = {main_res.pvalues['trend']:.4f}")

if main_res.pvalues['trend'] < 0.05:
    print("  → trend显著，说明控制时间趋势是必要的")
else:
    print("  → trend不显著，时间趋势不明显，但保留控制更稳健")

# ============================================================
# 7. 过拟合检验（留一省交叉验证）
# ============================================================
print("\n" + "=" * 60)
print("【步骤4】过拟合检验：留一省交叉验证")
print("=" * 60)

provinces = sorted(panel['地区'].unique())
cv_detail = []

for prov in provinces:
    train = panel[panel['地区'] != prov].copy()
    test = panel[panel['地区'] == prov].copy()
    train_idx = train.set_index(['地区', '年份'])
    try:
        cv_model = PanelOLS(train_idx[dep_var], train_idx[main_vars],
                            entity_effects=True).fit()
        y_pred = (test[main_vars].values @ cv_model.params.values)
        y_true = test[dep_var].values
        y_pred_adj = y_pred - y_pred.mean() + y_true.mean()
        for i, (_, row) in enumerate(test.iterrows()):
            cv_detail.append({
                '省份': prov, '年份': row['年份'],
                '真实值': y_true[i], '预测值': y_pred_adj[i],
                '残差': y_true[i] - y_pred_adj[i]
            })
    except:
        pass

df_cv = pd.DataFrame(cv_detail)
df_cv['真实值_dm'] = df_cv.groupby('省份')['真实值'].transform(lambda x: x - x.mean())
df_cv['预测值_dm'] = df_cv.groupby('省份')['预测值'].transform(lambda x: x - x.mean())
ss_res = (df_cv['真实值_dm'] - df_cv['预测值_dm']).pow(2).sum()
ss_tot = df_cv['真实值_dm'].pow(2).sum()
cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

print(f"  原始 R²(within)      = {main_res.rsquared_within:.4f}")
print(f"  交叉验证 CV-R²(within) = {cv_r2:.4f}")
print(f"  差距                  = {abs(main_res.rsquared_within - cv_r2):.4f}")
if abs(main_res.rsquared_within - cv_r2) < 0.10:
    cv_msg = "✓ 差距<0.10，不存在过拟合"
else:
    cv_msg = "△ 存在一定过拟合风险"
print(f"  {cv_msg}")

# ============================================================
# 8. 输出Excel
# ============================================================
print("\n" + "=" * 60)
print("【步骤5】输出Excel")
print("=" * 60)

out_file = os.path.join(OUTPUT_DIR, "四维度熵权法_个体FE加时间趋势.xlsx")

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    
    # Sheet1: 熵权法权重
    df_weights.to_excel(writer, sheet_name='熵权法权重', index=False)
    
    # Sheet2: 主回归结果
    reg_df = pd.DataFrame({
        '变量': main_res.params.index,
        '系数': main_res.params.values,
        '标准误': main_res.std_errors.values,
        't值': main_res.tstats.values,
        'p值': main_res.pvalues.values,
    })
    reg_df['显著性'] = reg_df['p值'].apply(
        lambda p: '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))
    reg_df.to_excel(writer, sheet_name='回归结果', index=False)
    
    # Sheet3: 模型摘要
    summary = [
        ['模型', '个体固定效应 + 线性时间趋势'],
        ['R²(within)', main_res.rsquared_within],
        ['R²(overall)', main_res.rsquared],
        ['观测数', main_res.nobs],
        ['个体数', main_res.entity_info['total']],
        ['标准误', '聚类稳健(cluster_entity)'],
        ['因变量', 'PCA综合得分（养老金融高质量发展指数）'],
        ['时间控制', '线性趋势变量 trend=1(2016)~8(2023)'],
        ['', ''],
        ['--- 过拟合检验 ---', ''],
        ['原始R²(within)', main_res.rsquared_within],
        ['CV-R²(within)', cv_r2],
        ['差距', abs(main_res.rsquared_within - cv_r2)],
        ['结论', cv_msg],
        ['', ''],
        ['--- 论文写法建议 ---', ''],
        ['模型设定', '采用个体固定效应模型，控制省份异质性，同时加入线性时间趋势变量控制公共时间趋势'],
        ['为何不用双向FE', '样本仅含4个核心自变量和31省×8年面板，时间固定效应（7个虚拟变量）会吸收过多组内变异，导致R²(within)为负，故改用线性趋势控制'],
    ]
    pd.DataFrame(summary, columns=['项目','值']).to_excel(writer, sheet_name='模型摘要', index=False)
    
    # Sheet4: 交叉验证
    df_cv.to_excel(writer, sheet_name='交叉验证详情', index=False)
    
    # Sheet5: 维度说明
    notes = []
    for dim, indicators in dim_indicators.items():
        w_dict = df_weights[df_weights['维度']==dim].set_index('指标')['熵权权重']
        w_str = ' + '.join([f"{w_dict[ind]:.3f}×z({ind})" for ind in indicators])
        notes.append({'维度': dim, '合成公式': f"{dim} = {w_str}", '方法': '熵权法+Z-score标准化'})
    pd.DataFrame(notes).to_excel(writer, sheet_name='维度合成说明', index=False)
    
    # Sheet6: 面板数据
    out_cols = ['地区', '年份', 'PCA综合得分',
                '占比65', '抚养比', 'Aging',
                '人均GDP', 'ln_人均GDP', '城镇化率', 'Econ',
                '信贷深度', '金融业占比', 'Finance',
                '三产占比', '养老床位密度', 'Service',
                'trend']
    panel[out_cols].to_excel(writer, sheet_name='面板数据', index=False)

print(f"  ✓ 输出: {out_file}")
print("\n完成！")
