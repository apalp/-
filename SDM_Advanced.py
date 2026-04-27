import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

# 屏蔽不必要的警告
warnings.filterwarnings('ignore')

# ============================================================
# 1. 配置信息与路径
# ============================================================
INPUT_PATH = r"D:\用户\Desktop\pca\权重\PCA综合评价结果.xlsx"
OUTPUT_DIR = r"D:\用户\Desktop\pca\新增回归变量\优化"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# 解释变量路径设置
EXPLANATORY_VARS = [
    (r"D:\用户\Desktop\topsis\回归数据处理\01六十五岁以上人口占比.xlsx", "老龄化率"),
    (r"D:\用户\Desktop\topsis\回归数据处理\02老年人口抚养比.xlsx", "老年抚养比"),
    (r"D:\用户\Desktop\topsis\回归数据处理\03人均地区生产总值.xlsx", "人均GDP"),
    (r"D:\用户\Desktop\topsis\回归数据处理\05城镇化率.xlsx", "城镇化率"),
    (r"D:\用户\Desktop\topsis\回归数据处理\06金融机构人民币贷款余额除以地区生产总值.xlsx", "信贷深度"),
    (r"D:\用户\Desktop\topsis\回归数据处理\07金融业增加值占GDP比重.xlsx", "金融业占比"),
    (r"D:\用户\Desktop\topsis\回归数据处理\08第三产业增加值占GDP比重.xlsx", "第三产业占比"),
    (r"D:\用户\Desktop\topsis\回归数据处理\09每千位老人床位数.xlsx", "养老床位密度"),
]

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

NAME_MAP = {
    '北京市':'北京','天津市':'天津','河北省':'河北','山西省':'山西','内蒙古自治区':'内蒙古',
    '辽宁省':'辽宁','吉林省':'吉林','黑龙江省':'黑龙江','上海市':'上海','江苏省':'江苏',
    '浙江省':'浙江','安徽省':'安徽','福建省':'福建','江西省':'江西','山东省':'山东',
    '河南省':'河南','湖北省':'湖北','湖南省':'湖南','广东省':'广东','广西壮族自治区':'广西',
    '海南省':'海南','重庆市':'重庆','四川省':'四川','贵州省':'贵州','云南省':'云南',
    '西藏自治区':'西藏','陕西省':'陕西','甘肃省':'甘肃','青海省':'青海','宁夏回族自治区':'宁夏',
    '新疆维吾尔自治区':'新疆',
}

# ============================================================
# 2. 核心辅助函数
# ============================================================

def clean_name(name):
    return NAME_MAP.get(str(name).strip(), str(name).strip())

def read_wide_table(filepath, var_name):
    df = pd.read_excel(filepath, sheet_name=0)
    df = df.rename(columns={df.columns[0]:'地区'})
    df['地区'] = df['地区'].apply(clean_name)
    year_cols = {col: int(float(str(col).strip().replace('年',''))) for col in df.columns[1:] 
                 if str(col).strip().replace('年','').replace('.0','').isdigit()}
    keep_cols = {orig: yr for orig, yr in year_cols.items() if yr in STUDY_YEARS}
    df_subset = df[['地区'] + list(keep_cols.keys())].copy()
    df_subset = df_subset.rename(columns=keep_cols)
    return df_subset.melt(id_vars='地区', var_name='年份', value_name=var_name)

def calculate_rigorous_effects(model, W, var_names, rho_name='WY'):
    """
    严谨效应分解 (LeSage & Pace 2009)
    """
    N = W.shape[0]
    I = np.eye(N)
    rho = model.params[rho_name]
    inv_mat = np.linalg.inv(I - rho * W)
    
    results = []
    for v in var_names:
        beta = model.params[v]
        theta = model.params.get(f'W_{v}', 0)
        S_mat = inv_mat @ (beta * I + theta * W)
        
        direct = np.trace(S_mat) / N
        total = np.sum(S_mat) / N
        indirect = total - direct
        results.append({
            '变量': v, 
            '直接效应': direct, 
            '间接效应': indirect, 
            '总效应': total, 
            '反馈效应': direct - beta
        })
    return pd.DataFrame(results)

def compute_spatial_vars(df, W, target_y, x_vars):
    df_res = df.copy()
    prov_list = sorted(df_res['地区'].unique())
    for y in sorted(df_res['年份'].unique()):
        mask = df_res['年份'] == y
        data_y = df_res[mask].set_index('地区').reindex(prov_list)
        df_res.loc[mask, 'WY'] = W @ data_y[target_y].values
        for x in x_vars:
            df_res.loc[mask, f'W_{x}'] = W @ data_y[x].values
    return df_res

# ============================================================
# 3. 数据流水线
# ============================================================
print("1. 正在整合面板数据并处理内生性滞后...")
all_long_vars = [read_wide_table(fp, vn) for fp, vn in EXPLANATORY_VARS]
panel_data = all_long_vars[0]
for d in all_long_vars[1:]:
    panel_data = panel_data.merge(d, on=['地区', '年份'], how='inner')

topsis_df = pd.read_excel(INPUT_PATH, sheet_name='完整面板数据')
panel = topsis_df[['地区','年份','PCA综合得分']].merge(panel_data, on=['地区','年份'], how='inner')

panel['ln人均GDP'] = np.log(panel['人均GDP'])
scaler = StandardScaler()
panel[['A1','A2']] = scaler.fit_transform(panel[['老龄化率','老年抚养比']])
panel['老龄化综合指数'] = panel[['A1','A2']].mean(axis=1)

panel = panel.sort_values(['地区','年份'])
panel['老龄化综合指数_L1'] = panel.groupby('地区')['老龄化综合指数'].shift(1)
panel = panel.dropna(subset=['老龄化综合指数_L1']).reset_index(drop=True)

# ============================================================
# 4. 构建空间权重矩阵
# ============================================================
provinces = sorted(panel['地区'].unique())
N = len(provinces)
p_idx = {p: i for i, p in enumerate(provinces)}

W1_raw = np.zeros((N, N))
for p, nbs in ADJACENCY.items():
    if p in p_idx:
        for nb in nbs:
            if nb in p_idx: W1_raw[p_idx[p], p_idx[nb]] = 1
W1 = W1_raw / np.maximum(W1_raw.sum(axis=1, keepdims=True), 1)

W2_raw = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j: W2_raw[i, j] = 1 / (abs(i - j) + 1)
W2 = W2_raw / W2_raw.sum(axis=1, keepdims=True)

gdp_vec = panel.groupby('地区')['人均GDP'].mean().reindex(provinces).values
W_econ = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j: W_econ[i, j] = 1 / (1 + abs(gdp_vec[i] - gdp_vec[j]))
W3 = 0.5 * W1 + 0.5 * (W_econ / W_econ.sum(axis=1, keepdims=True))

# ============================================================
# 5. 模型估计与检验
# ============================================================
core_x = ['老龄化综合指数_L1', 'ln人均GDP', '城镇化率', '信贷深度', '金融业占比', '第三产业占比', '养老床位密度']
sdm_x = core_x + ['WY'] + [f'W_{x}' for x in core_x]

panel_w1 = compute_spatial_vars(panel, W1, 'PCA综合得分', core_x)
p_idx_df = panel_w1.set_index(['地区','年份'])

print("\n2. 执行模型显著性与退化检验...")
fe = PanelOLS(p_idx_df['PCA综合得分'], p_idx_df[core_x], entity_effects=True).fit()
re = RandomEffects(p_idx_df['PCA综合得分'], p_idx_df[core_x]).fit()
h_stat = 2 * abs(fe.loglik - re.loglik)
h_p = chi2.sf(h_stat, df=len(core_x))
print(f"Hausman检验 p值: {h_p:.4f}")

res_sdm = PanelOLS(p_idx_df['PCA综合得分'], p_idx_df[sdm_x], 
                   entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)

# 修复 Wald 检验
wx_constraints = [f"{v} = 0" for v in [f'W_{x}' for x in core_x]]
try:
    wald_res = res_sdm.wald_test(formula=wx_constraints)
    # 兼容不同版本的属性名
    wp = wald_res.pvalue if hasattr(wald_res, 'pvalue') else wald_res.pval
    print(f"Wald检验 (SDM vs SAR) p值: {wp:.4f}")
except Exception as e:
    print(f"Wald检验执行异常: {e}")
    wp = None

# ============================================================
# 6. 效应分解与鲁棒性
# ============================================================
print("\n3. 执行严谨效应分解 (偏微分矩阵法)...")
df_effects = calculate_rigorous_effects(res_sdm, W1, core_x)
print(df_effects[['变量', '直接效应', '间接效应', '总效应']].round(4))

print("\n4. 鲁棒性检验 (W2 & W3)...")
robust_summary = []
for name, W_mat in [('距离权重W2', W2), ('经济嵌套W3', W3)]:
    temp_p = compute_spatial_vars(panel, W_mat, 'PCA综合得分', core_x).set_index(['地区','年份']).dropna(subset=sdm_x)
    temp_res = PanelOLS(temp_p['PCA综合得分'], temp_p[sdm_x], entity_effects=True, time_effects=True).fit()
    robust_summary.append({
        '权重矩阵': name, 
        '核心系数': temp_res.params['老龄化综合指数_L1'], 
        'P值': temp_res.pvalues['老龄化综合指数_L1'],
        '空间Rho': temp_res.params['WY']
    })
print(pd.DataFrame(robust_summary).round(4))

# ============================================================
# 7. 导出结果
# ============================================================
final_out = os.path.join(OUTPUT_DIR, "空间计量全流程分析.xlsx")
with pd.ExcelWriter(final_out) as writer:
    # 系数表
    sdm_tab = pd.concat([res_sdm.params, res_sdm.std_errors, res_sdm.tstats, res_sdm.pvalues], axis=1)
    sdm_tab.columns = ['系数', '标准误', 't值', 'P值']
    sdm_tab.to_excel(writer, 'SDM全模型系数')
    
    # 效应分解表
    df_effects.to_excel(writer, '效应分解', index=False)
    
    # 鲁棒性表
    pd.DataFrame(robust_summary).to_excel(writer, '鲁棒性检验', index=False)
    
    # 检验结果汇总
    pd.DataFrame({
        '检验': ['Hausman p-val', 'Wald p-val', 'Within R2'],
        '数值': [h_p, wp, res_sdm.rsquared_within]
    }).to_excel(writer, '模型统计检验汇总', index=False)

print(f"\n🎉 空间计量全流程已完成！结果文件已存至:\n{final_out}")
