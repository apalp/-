import pandas as pd
import numpy as np
import os

from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm

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

# 人均GDP取对数
panel['ln_人均GDP'] = np.log(panel['人均GDP'].clip(lower=1))

print(f"  面板: {len(panel)} 行, {panel['地区'].nunique()} 省, {panel['年份'].nunique()} 年")

# ============================================================
# 4. ★ 熵权法赋权
# ============================================================
print("\n" + "=" * 60)
print("【步骤2】熵权法计算各维度内指标权重")
print("=" * 60)

def entropy_weight(df_cols):
    """
    熵权法计算权重
    输入: DataFrame，每列为一个指标（已标准化到0-1）
    输出: 各指标权重（numpy数组），权重之和=1
    """
    # Step1: Min-Max归一化到(0,1)区间（避免log(0)）
    scaler = MinMaxScaler()
    normed = pd.DataFrame(scaler.fit_transform(df_cols), columns=df_cols.columns)
    # 平移避免0值
    normed = normed + 1e-10
    
    n = len(normed)
    
    # Step2: 计算各指标的比重 p_ij
    p = normed.div(normed.sum(axis=0), axis=1)
    
    # Step3: 计算信息熵 e_j
    k = 1.0 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    
    # Step4: 计算信息效用值 d_j = 1 - e_j
    d = 1 - e
    
    # Step5: 归一化得到权重
    w = d / d.sum()
    
    return w

# 定义各维度的原始指标（用于熵权法计算）
dim_indicators = {
    'Aging':   ['占比65', '抚养比'],
    'Econ':    ['ln_人均GDP', '城镇化率'],
    'Finance': ['信贷深度', '金融业占比'],
    'Service': ['三产占比', '养老床位密度'],
}

# 计算权重并合成
weight_records = []
z_scaler = StandardScaler()

for dim, indicators in dim_indicators.items():
    # 计算熵权
    w = entropy_weight(panel[indicators])
    
    print(f"\n  {dim} 维度熵权法权重:")
    for ind, wi in zip(indicators, w):
        print(f"    {ind:20s}: {wi:.4f} ({wi*100:.1f}%)")
        weight_records.append({'维度': dim, '指标': ind, '熵权权重': wi})
    
    # Z-score标准化
    z_cols = [f'z_{ind}' for ind in indicators]
    panel[z_cols] = z_scaler.fit_transform(panel[indicators])
    
    # ★ 熵权加权合成
    panel[dim] = sum(w[i] * panel[z_cols[i]] for i in range(len(indicators)))
    print(f"  ✓ {dim} = " + " + ".join([f"{w[i]:.3f}*z({ind})" for i, ind in enumerate(indicators)]))

df_weights = pd.DataFrame(weight_records)

# ============================================================
# 5. 个体固定效应回归
# ============================================================
print("\n" + "=" * 60)
print("【步骤3】个体固定效应回归")
print("=" * 60)

dep_var = 'PCA综合得分'
indep_vars = ['Aging', 'Econ', 'Finance', 'Service']

panel_idx = panel.set_index(['地区', '年份'])
fe_res = PanelOLS(panel_idx[dep_var], panel_idx[indep_vars],
                  entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(f"\n  R²(within) = {fe_res.rsquared_within:.4f}")
print(f"  观测数 = {fe_res.nobs}")
print(f"\n  回归系数:")
for v in indep_vars:
    sig = '***' if fe_res.pvalues[v]<0.01 else ('**' if fe_res.pvalues[v]<0.05 else ('*' if fe_res.pvalues[v]<0.1 else ''))
    print(f"    {v:12s}: β={fe_res.params[v]:+.4f}, t={fe_res.tstats[v]:.2f}, p={fe_res.pvalues[v]:.4f} {sig}")

# ============================================================
# 6. ★ 过拟合检验：留一省交叉验证 (Leave-One-Province-Out)
# ============================================================
print("\n" + "=" * 60)
print("【步骤4】过拟合检验：留一省交叉验证")
print("=" * 60)
print("  说明：每次去掉一个省跑回归，用估计系数预测该省，")
print("        比较预测值与真实值。若非过拟合，CV-R²应接近原始R²。\n")

provinces = sorted(panel['地区'].unique())
cv_errors = []
cv_detail = []

for prov in provinces:
    # 训练集：去掉当前省
    train = panel[panel['地区'] != prov].copy()
    test = panel[panel['地区'] == prov].copy()
    
    # 训练集上跑个体FE
    train_idx = train.set_index(['地区', '年份'])
    try:
        cv_model = PanelOLS(train_idx[dep_var], train_idx[indep_vars],
                            entity_effects=True).fit()
        
        # 预测测试省（用系数 * X，不含固定效应截距）
        # 个体FE模型的预测：y_hat = X @ beta + 该省均值
        # 这里简化为用系数预测组内变化趋势
        y_pred = (test[indep_vars].values @ cv_model.params.values)
        # 加上测试省的均值调整
        y_true = test[dep_var].values
        y_pred_adj = y_pred - y_pred.mean() + y_true.mean()
        
        mse = np.mean((y_true - y_pred_adj) ** 2)
        cv_errors.append(mse)
        
        for i, (_, row) in enumerate(test.iterrows()):
            cv_detail.append({
                '省份': prov, '年份': row['年份'],
                '真实值': y_true[i], '预测值': y_pred_adj[i],
                '残差': y_true[i] - y_pred_adj[i]
            })
    except Exception as e:
        print(f"  {prov} 跳过: {e}")

# CV统计
total_var = panel[dep_var].var()
cv_mse = np.mean(cv_errors)
cv_r2 = 1 - cv_mse / total_var if total_var > 0 else float('nan')

# 组内CV-R²（更可比的指标）
df_cv = pd.DataFrame(cv_detail)
# 去掉省份均值后计算
df_cv['真实值_demean'] = df_cv.groupby('省份')['真实值'].transform(lambda x: x - x.mean())
df_cv['预测值_demean'] = df_cv.groupby('省份')['预测值'].transform(lambda x: x - x.mean())
ss_res = (df_cv['真实值_demean'] - df_cv['预测值_demean']).pow(2).sum()
ss_tot = df_cv['真实值_demean'].pow(2).sum()
cv_r2_within = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

print(f"  原始 R²(within)     = {fe_res.rsquared_within:.4f}")
print(f"  交叉验证 CV-R²(within) = {cv_r2_within:.4f}")
print(f"  差距                 = {abs(fe_res.rsquared_within - cv_r2_within):.4f}")
print()

if abs(fe_res.rsquared_within - cv_r2_within) < 0.10:
    cv_conclusion = "✓ 差距<0.10，不存在过拟合，R²可信"
    print(f"  {cv_conclusion}")
elif abs(fe_res.rsquared_within - cv_r2_within) < 0.20:
    cv_conclusion = "△ 差距在0.10-0.20之间，轻微过拟合风险，可接受"
    print(f"  {cv_conclusion}")
else:
    cv_conclusion = "✗ 差距>0.20，存在过拟合嫌疑，需考虑减少变量"
    print(f"  {cv_conclusion}")

# ============================================================
# 7. 额外稳健性：逐步加入维度看R²变化
# ============================================================
print("\n" + "=" * 60)
print("【步骤5】逐步加入维度（看是否某个维度导致R²虚高）")
print("=" * 60)

stepwise_results = []
for i in range(1, len(indep_vars) + 1):
    vars_i = indep_vars[:i]
    res_i = PanelOLS(panel_idx[dep_var], panel_idx[vars_i],
                     entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    print(f"  加入 {vars_i[-1]:12s} → R²(within)={res_i.rsquared_within:.4f} (共{i}个变量)")
    stepwise_results.append({
        '步骤': i, '新增变量': vars_i[-1],
        '当前变量': ', '.join(vars_i),
        'R²(within)': res_i.rsquared_within,
    })

# ============================================================
# 8. 输出Excel
# ============================================================
print("\n" + "=" * 60)
print("【步骤6】输出Excel")
print("=" * 60)

out_file = os.path.join(OUTPUT_DIR, "四维度熵权法回归_含过拟合检验.xlsx")

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    
    # Sheet1: 熵权法权重
    df_weights.to_excel(writer, sheet_name='熵权法权重', index=False)
    
    # Sheet2: 回归结果
    reg_df = pd.DataFrame({
        '变量': fe_res.params.index,
        '系数': fe_res.params.values,
        '标准误': fe_res.std_errors.values,
        't值': fe_res.tstats.values,
        'p值': fe_res.pvalues.values,
    })
    reg_df['显著性'] = reg_df['p值'].apply(
        lambda p: '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))
    reg_df.to_excel(writer, sheet_name='回归结果', index=False)
    
    # Sheet3: 模型摘要
    summary = [
        ['R²(within)', fe_res.rsquared_within],
        ['R²(overall)', fe_res.rsquared],
        ['观测数', fe_res.nobs],
        ['个体数', fe_res.entity_info['total']],
        ['固定效应', '个体固定效应'],
        ['标准误', '聚类稳健(cluster_entity)'],
        ['因变量', 'PCA综合得分'],
        ['', ''],
        ['--- 过拟合检验 ---', ''],
        ['原始R²(within)', fe_res.rsquared_within],
        ['CV-R²(within)', cv_r2_within],
        ['差距', abs(fe_res.rsquared_within - cv_r2_within)],
        ['结论', cv_conclusion],
    ]
    pd.DataFrame(summary, columns=['项目','值']).to_excel(writer, sheet_name='模型摘要与过拟合检验', index=False)
    
    # Sheet4: 逐步回归
    pd.DataFrame(stepwise_results).to_excel(writer, sheet_name='逐步加入维度', index=False)
    
    # Sheet5: 交叉验证详细结果
    df_cv.to_excel(writer, sheet_name='交叉验证预测详情', index=False)
    
    # Sheet6: 维度处理思路
    notes = []
    for dim, indicators in dim_indicators.items():
        w_dict = df_weights[df_weights['维度']==dim].set_index('指标')['熵权权重']
        w_str = '、'.join([f"{ind}({w_dict[ind]:.1%})" for ind in indicators])
        notes.append({
            '维度': dim,
            '指标与权重': w_str,
            '合成方法': '熵权法：MinMax归一化→信息熵→信息效用值→归一化权重，再对Z-score标准化后的指标加权求和',
        })
    pd.DataFrame(notes).to_excel(writer, sheet_name='维度处理思路', index=False)
    
    # Sheet7: 合成后面板数据
    out_cols = ['地区', '年份', 'PCA综合得分',
                '占比65', '抚养比', 'Aging',
                '人均GDP', 'ln_人均GDP', '城镇化率', 'Econ',
                '信贷深度', '金融业占比', 'Finance',
                '三产占比', '养老床位密度', 'Service']
    panel[out_cols].to_excel(writer, sheet_name='合成后面板数据', index=False)

print(f"  ✓ 输出: {out_file}")
print("\n" + "=" * 60)
print("【总结】")
print("=" * 60)
print(f"  赋权方法: 熵权法（数据驱动，客观赋权）")
print(f"  回归模型: 个体固定效应 + 聚类稳健标准误")
print(f"  R²(within) = {fe_res.rsquared_within:.4f}")
print(f"  过拟合检验: {cv_conclusion}")
print("  完成！")
