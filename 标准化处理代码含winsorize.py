

import pandas as pd
import numpy as np

# ============================================================

# ============================================================
FILE_PATH =r"D:\用户\Desktop\指标处理\013PPP养老_汇总结果0419.xlsx"
INDICATOR_NAME = "013PPP养老产业投资强度"
STUDY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]  # 研究期
FORCE_WINSORIZE = False  # True=强制缩尾, False=自动判断, None=强制不缩尾

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_excel(FILE_PATH)
print(f"【{INDICATOR_NAME}】")
print("=" * 60)

# 提取研究期年份(自动匹配存在的列)
available_years = [y for y in STUDY_YEARS if y in df.columns]
missing_years = [y for y in STUDY_YEARS if y not in df.columns]

print(f"文件中的年份: {list(df.columns[1:])}")
print(f"研究期需要:   {STUDY_YEARS}")
print(f"可用年份:     {available_years}")
if missing_years:
    print(f"⚠️ 缺失年份:  {missing_years}")
    print(f"   请先处理缺失年份再运行本脚本!")

# 截取研究期
data = df[['地区'] + available_years].copy()

# ============================================================
# 2. 转长面板
# ============================================================
long = data.melt(id_vars='地区', var_name='年份', value_name='原始值')
print(f"\n观测数: {len(long)}行 ({data.shape[0]}省 x {len(available_years)}年)")

# ============================================================
# 3. 自动检测是否需要 Winsorize
# ============================================================
vals = long['原始值'].dropna()
skewness = abs(vals.skew())
kurtosis_val = vals.kurtosis()

# IQR法检测异常值
q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
n_outliers = ((vals < lower_fence) | (vals > upper_fence)).sum()
outlier_pct = n_outliers / len(vals) * 100

print(f"\n{'=' * 60}")
print(f"【分布诊断】")
print(f"{'=' * 60}")
print(f"范围:     [{vals.min():.4f}, {vals.max():.4f}]")
print(f"均值:     {vals.mean():.4f}")
print(f"标准差:   {vals.std():.4f}")
print(f"偏度:     {skewness:.2f} {'(偏态明显)' if skewness > 1 else '(基本对称)'}")
print(f"峰度:     {kurtosis_val:.2f} {'(厚尾)' if kurtosis_val > 3 else '(正常)'}")
print(f"IQR异常值: {n_outliers}个 ({outlier_pct:.1f}%)")

# 判断规则: 满足任一条件则建议缩尾
need_winsorize = (skewness > 1.5) or (kurtosis_val > 5) or (outlier_pct > 5)

# 额外规则: 如果数据本身是比率(0-1之间)或百分比(0-100), 通常不需要
data_range = vals.max() - vals.min()
is_ratio = (vals.min() >= 0) and (vals.max() <= 1.01)
is_percentage = (vals.min() >= 0) and (vals.max() <= 100.1) and (data_range < 80)

# 手动开关优先
if FORCE_WINSORIZE is True:
    need_winsorize = True
    print(f"\n→ 判断: 手动强制开启Winsorize")
elif FORCE_WINSORIZE is None:
    need_winsorize = False
    print(f"\n→ 判断: 手动强制关闭Winsorize")
elif is_ratio or is_percentage:
    need_winsorize = False
    print(f"\n→ 判断: 该指标为比率/百分比型, 分布自然有界, 【不需要Winsorize】")
elif need_winsorize:
    print(f"\n→ 判断: 存在明显异常值/偏态, 【建议Winsorize】")
else:
    print(f"\n→ 判断: 分布基本正常, 【不需要Winsorize】")

# ============================================================
# 4. Winsorize (如果需要)
# ============================================================
long['缩尾后'] = long['原始值'].copy()

if need_winsorize:
    p5 = vals.quantile(0.05)
    p95 = vals.quantile(0.95)
    long['缩尾后'] = long['缩尾后'].clip(lower=p5, upper=p95)

    n_low = (long['原始值'] < p5).sum()
    n_high = (long['原始值'] > p95).sum()

    print(f"\n{'=' * 60}")
    print(f"【Winsorize 5%-95% 缩尾】")
    print(f"{'=' * 60}")
    print(f"缩尾前: [{long['原始值'].min():.4f}, {long['原始值'].max():.4f}]")
    print(f"阈值:   [{p5:.4f}, {p95:.4f}]")
    print(f"缩尾后: [{long['缩尾后'].min():.4f}, {long['缩尾后'].max():.4f}]")
    print(f"下截: {n_low}个, 上截: {n_high}个, 共{n_low + n_high}个观测被修正")

    # 显示被缩尾的具体观测
    clipped = long[
        (long['原始值'] < p5) | (long['原始值'] > p95)
    ][['地区', '年份', '原始值', '缩尾后']].copy()
    if len(clipped) > 0:
        print(f"\n被缩尾的观测:")
        for _, row in clipped.iterrows():
            print(f"  {row['地区']} {row['年份']}: {row['原始值']:.4f} -> {row['缩尾后']:.4f}")
else:
    print("  跳过Winsorize, 直接进入标准化")

# ============================================================
# 5. Min-Max 标准化 (全样本统一)
# ============================================================
s_min = long['缩尾后'].min()
s_max = long['缩尾后'].max()

if s_max == s_min:
    long['标准化'] = 0.5
    print("\n⚠️ 警告: 所有值相同, 标准化后统一赋0.5")
else:
    long['标准化'] = (long['缩尾后'] - s_min) / (s_max - s_min)

print(f"\n{'=' * 60}")
print(f"【Min-Max 标准化 (全样本端点)】")
print(f"{'=' * 60}")
print(f"全样本 min={s_min:.4f}, max={s_max:.4f}")
print(f"标准化后: [{long['标准化'].min():.4f}, {long['标准化'].max():.4f}], 均值={long['标准化'].mean():.4f}")

# ============================================================
# 6. 零值平移 (为熵值法准备)
# ============================================================
long['最终值'] = 0.99 * long['标准化'] + 0.01

print(f"\n零值平移: x' = 0.99x + 0.01")
print(f"最终范围: [{long['最终值'].min():.4f}, {long['最终值'].max():.4f}]")

# ============================================================
# 7. 输出
# ============================================================

# 宽表(和你原来格式一致)
wide = long.pivot(index='地区', columns='年份', values='最终值').reset_index()
wide.columns = ['地区'] + [int(c) for c in wide.columns[1:]]

# 保存
output_path = rf"D:\用户\Desktop\标准化处理/{INDICATOR_NAME}_标准化结果.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Sheet1: 最终结果(宽表, 直接合并到主表)
    wide.to_excel(writer, sheet_name='标准化结果', index=False)
    # Sheet2: 完整处理过程(长表, 可核查)
    long.to_excel(writer, sheet_name='处理过程', index=False)

print(f"\n{'=' * 60}")
print(f"【输出完成】")
print(f"{'=' * 60}")
print(f"文件: {output_path}")
print(f"Sheet1: 标准化结果 (宽表, 直接合并到主表)")
print(f"Sheet2: 处理过程   (长表, 含每步中间值)")

# ============================================================
# 8. 预览最终结果(前5省)
# ============================================================
print(f"\n{'=' * 60}")
print(f"【结果预览 (前5省)】")
print(f"{'=' * 60}")
print(wide.head().to_string(index=False))
