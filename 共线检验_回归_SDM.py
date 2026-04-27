# ============================================================
# 回归优化 + 共线性分析 + SDM
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\新增回归变量"
STUDY_YEARS = [2016,2017,2018,2019,2020,2021,2022,2023]

# 解释变量文件
EXPLANATORY_VARS = [
    # ---- 人口老龄化 ----
    (r"D:\用户\Desktop\topsis\回归数据处理\01六十五岁以上人口占比.xlsx", "老龄化率"),
    (r"D:\用户\Desktop\topsis\回归数据处理\02老年人口抚养比.xlsx", "老年抚养比"),
    # ---- 经济基础 ----
    (r"D:\用户\Desktop\topsis\回归数据处理\03人均地区生产总值.xlsx", "人均GDP"),
    (r"D:\用户\Desktop\topsis\回归数据处理\05城镇化率.xlsx", "城镇化率"),
    # ---- 金融生态 ----
    (r"D:\用户\Desktop\topsis\回归数据处理\06金融机构人民币贷款余额除以地区生产总值.xlsx", "信贷深度"),
    (r"D:\用户\Desktop\topsis\回归数据处理\07金融业增加值占GDP比重.xlsx", "金融业占比"),
    # ---- 服务产业 ----
    (r"D:\用户\Desktop\topsis\回归数据处理\08第三产业增加值占GDP比重.xlsx", "第三产业占比"),
    (r"D:\用户\Desktop\topsis\回归数据处理\09每千位老人床位数.xlsx", "养老床位密度"),
]

# 省份邻接矩阵
ADJACENCY = { 
    '北京': ['天津','河北'], '天津': ['北京','河北'], '河北': ['北京','天津','山西','河南','山东','内蒙古','辽宁'], 
    '山西': ['河北','内蒙古','陕西','河南'], '内蒙古': ['河北','山西','陕西','宁夏','甘肃','黑龙江','吉林','辽宁'],
    '辽宁': ['河北','内蒙古','吉林'], '吉林': ['辽宁','内蒙古','黑龙江'], '黑龙江': ['吉林','内蒙古'],
    '上海': ['江苏','浙江'], '江苏': ['上海','浙江','安徽','山东'], '浙江': ['上海','江苏','安徽','江西','福建'],
    '安徽': ['江苏','浙江','江西','湖北','河南','山东'], '福建': ['浙江','江西','广东'], '江西': ['浙江','安徽','福建','湖北','湖南','广东'],
    '山东': ['河北','河南','安徽','江苏'], '河南': ['河北','山西','陕西','湖北','安徽','山东'], '湖北': ['河南','陕西','重庆','湖南','江西','安徽'],
    '湖南': ['湖北','重庆','贵州','广西','广东','江西'], '广东': ['福建','江西','湖南','广西','海南'], '广西': ['广东','湖南','贵州','云南'],
    '海南': ['广东'], '重庆': ['湖北','陕西','四川','贵州','湖南'], '四川': ['重庆','陕西','甘肃','青海','西藏','云南','贵州'],
    '贵州': ['重庆','四川','云南','广西','湖南'], '云南': ['四川','西藏','广西','贵州'], '西藏': ['新疆','青海','四川','云南'],
    '陕西': ['山西','内蒙古','宁夏','甘肃','四川','重庆','湖北','河南'], '甘肃': ['内蒙古','宁夏','青海','新疆','陕西','四川'],
    '青海': ['甘肃','新疆','西藏','四川'], '宁夏': ['内蒙古','陕西','甘肃'], '新疆': ['西藏','青海','甘肃','内蒙古'],
}

# 省份名称映射
NAME_MAP = {
    '北京市':'北京','天津市':'天津','河北省':'河北','山西省':'山西',
    '内蒙古自治区':'内蒙古','辽宁省':'辽宁','吉林省':'吉林','黑龙江省':'黑龙江',
    '上海市':'上海','江苏省':'江苏','浙江省':'浙江','安徽省':'安徽','福建省':'福建','江西省':'江西',
    '山东省':'山东','河南省':'河南','湖北省':'湖北','湖南省':'湖南','广东省':'广东','广西壮族自治区':'广西',
    '海南省':'海南','重庆市':'重庆','四川省':'四川','贵州省':'贵州','云南省':'云南',
    '西藏自治区':'西藏','陕西省':'陕西','甘肃省':'甘肃','青海省':'青海','宁夏回族自治区':'宁夏','新疆维吾尔自治区':'新疆',
}

def clean_name(name):
    return NAME_MAP.get(str(name).strip(), str(name).strip())

# 读取宽表 -> 长表
def read_wide_table(filepath, var_name):
    df = pd.read_excel(filepath, sheet_name=0)
    df = df.rename(columns={df.columns[0]:'地区'})
    df['地区'] = df['地区'].apply(clean_name)
    # 年份列
    year_cols = {}
    for col in df.columns[1:]:
        col_str = str(col).strip().replace('年','').replace('.0','')
        try:
            year = int(float(col_str))
            if 2000 <= year <= 2030: year_cols[col] = year
        except: continue
    keep_cols = {orig: yr for orig,yr in year_cols.items() if yr in STUDY_YEARS}
    df_subset = df[['地区']+list(keep_cols.keys())].copy()
    df_subset = df_subset.rename(columns=keep_cols)
    long = df_subset.melt(id_vars='地区', var_name='年份', value_name=var_name)
    long['年份'] = long['年份'].astype(int)
    return long

# ============================================================
# 数据准备
# ============================================================
print("="*70)
print("数据准备")
print("="*70)

# 读取解释变量
all_vars, var_names = [], []
for filepath,var_name in EXPLANATORY_VARS:
    long = read_wide_table(filepath,var_name)
    all_vars.append(long)
    var_names.append(var_name)
    print(f"  ✓ {var_name}")

# 合并
exog = all_vars[0][['地区','年份',var_names[0]]].copy()
for i in range(1,len(all_vars)):
    exog = exog.merge(all_vars[i][['地区','年份',var_names[i]]], on=['地区','年份'], how='outer')

# 被解释变量
topsis = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')
panel = topsis[['地区','年份','PCA综合得分']].merge(exog,on=['地区','年份'],how='inner')

# 预处理
panel['ln人均GDP'] = np.log(panel['人均GDP'])
panel['老龄化率²'] = panel['老龄化率']**2
panel['老年抚养比²'] = panel['老年抚养比']**2

print(f"面板: {panel.shape[0]}行, {panel['地区'].nunique()}省 x {panel['年份'].nunique()}年")

# ============================================================
# 1. 共线性检查
# ============================================================
print("\n" + "="*70)
print("共线性检查")
print("="*70)

corr = panel[['老龄化率','老年抚养比']].corr()
print(corr)

if abs(corr.loc['老龄化率','老年抚养比']) > 0.8:
    print("⚠️ 高共线性 detected (>0.8)，建议使用综合指标或分模型分析")
    use_pca = True
else:
    use_pca = False

# ============================================================
# 2. 构建综合指标 
# ============================================================
if use_pca:
    scaler = StandardScaler()
    panel[['老龄化率_std','老年抚养比_std']] = scaler.fit_transform(panel[['老龄化率','老年抚养比']])
    panel['老龄化综合指数'] = panel[['老龄化率_std','老年抚养比_std']].mean(axis=1)
    panel['老龄化综合指数²'] = panel['老龄化综合指数']**2
    age_vars = ['老龄化综合指数','老龄化综合指数²']
else:
    age_vars = ['老龄化率','老龄化率²','老年抚养比','老年抚养比²']

# ============================================================
# 3. 固定效应模型 (倒U型检验)
# ============================================================
print("\n" + "="*70)
print("固定效应模型")
print("="*70)

var_list_base = age_vars + ['ln人均GDP','城镇化率','信贷深度','金融业占比','第三产业占比','养老床位密度']
panel_idx = panel.set_index(['地区','年份'])

fe_model = PanelOLS(panel_idx['PCA综合得分'], panel_idx[var_list_base],
                    entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(fe_model.summary.tables[1])
print(f"R²(within) = {fe_model.rsquared_within:.4f}")

# ============================================================
# 4. 构建空间权重矩阵 W
# ============================================================
print("\n" + "="*70)
print("构建空间权重矩阵")
print("="*70)

provinces = sorted(panel['地区'].unique())
n_prov = len(provinces)
prov_idx = {p:i for i,p in enumerate(provinces)}

W = np.zeros((n_prov,n_prov))
for prov, neighbors in ADJACENCY.items():
    if prov in prov_idx:
        i = prov_idx[prov]
        for nb in neighbors:
            if nb in prov_idx:
                j = prov_idx[nb]
                W[i,j] = 1

row_sums = W.sum(axis=1,keepdims=True)
row_sums[row_sums==0]=1
W_norm = W / row_sums
print(f"空间权重矩阵: {n_prov}x{n_prov}, 非零元素={int(W.sum())}")

# ============================================================
# 5. 计算空间滞后变量 WY 和 WX
# ============================================================
print("\n计算空间滞后变量")
var_list_sdm = ['ln人均GDP','城镇化率','信贷深度','金融业占比','第三产业占比','养老床位密度'] + age_vars

for year in STUDY_YEARS:
    year_mask = panel['年份']==year
    year_data = panel[year_mask].copy()
    year_data = year_data.set_index('地区').loc[provinces].reset_index()
    # Wy
    y_vec = year_data['PCA综合得分'].values
    Wy = W_norm @ y_vec
    panel.loc[year_mask,'W_综合得分'] = panel.loc[year_mask,'地区'].map(dict(zip(provinces,Wy)))
    # WX
    for var in var_list_sdm:
        x_vec = year_data[var].values
        Wx = W_norm @ x_vec
        panel.loc[year_mask,f'W_{var}'] = panel.loc[year_mask,'地区'].map(dict(zip(provinces,Wx)))

print("空间滞后变量计算完成")

# ============================================================
# 6. SDM模型
# ============================================================
print("\n" + "="*70)
print("SDM模型估计")
print("="*70)

sdm_vars = var_list_sdm.copy()
sdm_vars.append('W_综合得分')  # 空间自回归
for var in var_list_sdm:  # 空间滞后解释变量
    sdm_vars.append(f'W_{var}')

panel_sdm = panel.set_index(['地区','年份'])
panel_sdm = panel_sdm.dropna(subset=sdm_vars + ['PCA综合得分'])

sdm_result = PanelOLS(panel_sdm['PCA综合得分'], panel_sdm[sdm_vars],
                      entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(sdm_result.summary.tables[1])
print(f"\nR²(within) = {sdm_result.rsquared_within:.4f}")

# ============================================================
# 7. 输出结果
# ============================================================
output_path = os.path.join(OUTPUT_DIR,"回归优化_共线性_SDM.xlsx")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # FE模型
    fe_df = pd.DataFrame({
        '变量': fe_model.params.index,
        '系数': fe_model.params.values.round(6),
        '标准误': fe_model.std_errors.values.round(6),
        't值': fe_model.tstats.values.round(4),
        'p值': fe_model.pvalues.values.round(4)
    })
    fe_df.to_excel(writer, sheet_name='固定效应模型', index=False)

    # SDM
    sdm_df = pd.DataFrame({
        '变量': sdm_result.params.index,
        '系数': sdm_result.params.values.round(6),
        '标准误': sdm_result.std_errors.values.round(6),
        't值': sdm_result.tstats.values.round(4),
        'p值': sdm_result.pvalues.values.round(4)
    })
    sdm_df.to_excel(writer, sheet_name='SDM结果', index=False)

    # 空间权重矩阵
    w_df = pd.DataFrame(W_norm, index=provinces, columns=provinces).round(4)
    w_df.to_excel(writer, sheet_name='空间权重矩阵W')

    # 完整面板
    panel.to_excel(writer, sheet_name='完整数据', index=False)

print(f"\n✓ 输出完成: {output_path}")
