# improved_classification_and_export.py
# 保存为文件后在含有原始 CSV 的目录运行： python improved_classification_and_export.py
# 需要 pandas、numpy 安装： pip install pandas numpy openpyxl

import pandas as pd
import numpy as np
import os
from math import log2

# 文件名（请确保这些文件位于当前工作目录）
MEDALS_CSV = 'summerOly_medal_counts.csv'
ATHLETES_CSV = 'summerOly_athletes.csv'
HOSTS_CSV = 'summerOly_hosts.csv'
PROGRAMS_CSV = 'summerOly_programs.csv'

# 检查文件存在性
for f in [MEDALS_CSV, ATHLETES_CSV, HOSTS_CSV, PROGRAMS_CSV]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file not found: {f}  (put it in the script directory)")

# 载入数据
medals = pd.read_csv(MEDALS_CSV)
athletes = pd.read_csv(ATHLETES_CSV)
hosts = pd.read_csv(HOSTS_CSV)
programs = pd.read_csv(PROGRAMS_CSV, encoding='utf-8-sig')


# 保证类型
medals['Year'] = medals['Year'].astype(int)
if 'Year' in athletes.columns:
    athletes['Year'] = athletes['Year'].astype(int)

# ---------- 计算基础指标 ----------
# 历史奖牌总数、首次/末次年份和参赛年份数
total_hist = medals.groupby('NOC').agg(
    Total_all_time=('Total', 'sum'),
    first_year_all=('Year', 'min'),
    last_year_all=('Year', 'max'),
    participation_years=('Year', lambda x: x.nunique())
).reset_index()

# 项目多样性（ProjectDiversity）
if 'Sport' in athletes.columns:
    proj_div = athletes.groupby('NOC')['Sport'].nunique().reset_index().rename(columns={'Sport':'ProjectDiversity'})
else:
    proj_div = pd.DataFrame({'NOC': medals['NOC'].unique(), 'ProjectDiversity': 0})

# 按 NOC / Sport 的奖牌汇总（用于集中度与 entropy）
if 'Sport' in athletes.columns:
    # 合并 medals 与运动（尽可能匹配 NOC+Year）
    merged = athletes.merge(medals[['NOC','Year','Gold','Silver','Bronze','Total']], on=['NOC','Year'], how='left', suffixes=('','_m'))
    sport_medals = merged.groupby(['NOC','Sport']).agg(medals_by_sport_total=('Total','sum')).reset_index()
else:
    sport_medals = pd.DataFrame(columns=['NOC','Sport','medals_by_sport_total'])

# Shannon entropy 函数
def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

entropy_df = sport_medals.groupby('NOC')['medals_by_sport_total'].apply(lambda s: shannon_entropy(s)).reset_index().rename(columns={'medals_by_sport_total':'Shannon_Entropy'})

# 金/银/铜总数
medal_totals = medals.groupby('NOC').agg(
    Gold_total=('Gold','sum'),
    Silver_total=('Silver','sum'),
    Bronze_total=('Bronze','sum')
).reset_index()

# ---------- 增长/时间序列指标 ----------
def compute_growth_metrics(noc_df):
    noc_df = noc_df.sort_values('Year').reset_index(drop=True)
    years = noc_df['Year'].values
    totals = noc_df['Total'].values.astype(float)
    # span
    year_span = years[-1] - years[0] if len(years) > 0 else 0
    # first medal year (first Year with Total>0)
    nonzero = totals > 0
    first_medal_year = int(years[nonzero][0]) if nonzero.any() else np.nan
    # CAGR approx: use first non-zero to last non-zero; periods approximated in Olympic editions (every 4 years)
    cagr = np.nan
    if nonzero.sum() >= 2:
        idx0 = np.where(nonzero)[0][0]
        idx_last = np.where(nonzero)[0][-1]
        start_val = totals[idx0]
        end_val = totals[idx_last]
        # periods as number of 4-year intervals
        periods = (years[idx_last] - years[idx0]) / 4.0
        if start_val > 0 and periods > 0:
            cagr = (end_val / start_val) ** (1.0/periods) - 1.0
    # last3 growth: (mean(last3) - mean(prev3)) / mean(prev3) if prev3 exists and >0
    last3_growth = np.nan
    if len(totals) >= 6:  # require at least 6 editions to have last3 vs prev3
        last3_mean = totals[-3:].mean()
        prev3_mean = totals[-6:-3].mean()
        if prev3_mean > 0:
            last3_growth = (last3_mean - prev3_mean) / prev3_mean
    # average previous medals (mean of all except the most recent year)
    avg_prev = np.nan
    if len(totals) >= 2:
        avg_prev = totals[:-1].mean()
    most_recent_year = int(years[-1]) if len(years)>0 else np.nan
    most_recent_total = float(totals[-1]) if len(totals)>0 else 0.0
    return pd.Series({
        'year_span': year_span,
        'first_medal_year': first_medal_year,
        'cagr': cagr,
        'last3_growth': last3_growth,
        'avg_previous_medals': avg_prev,
        'most_recent_year': most_recent_year,
        'most_recent_total': most_recent_total
    })

growth_metrics = medals.groupby('NOC').apply(compute_growth_metrics).reset_index()

# concentration: minimal number of sports to reach 80% medals
def sports_to_reach_80pct(noc):
    subset = sport_medals[sport_medals['NOC']==noc].copy()
    if subset.empty or subset['medals_by_sport_total'].sum() == 0:
        return np.nan
    subset = subset.sort_values('medals_by_sport_total', ascending=False).reset_index(drop=True)
    subset['cum'] = subset['medals_by_sport_total'].cumsum()
    total = subset['medals_by_sport_total'].sum()
    needed_idx = subset[subset['cum'] / total >= 0.8].index
    if len(needed_idx) == 0:
        return len(subset)
    return needed_idx[0] + 1

concentration = pd.DataFrame({'NOC': sport_medals['NOC'].unique()})
concentration['sports_for_80pct'] = concentration['NOC'].apply(lambda n: sports_to_reach_80pct(n))

# ---------- 合并所有特征 ----------
df = total_hist.merge(proj_div, on='NOC', how='left') \
               .merge(entropy_df, on='NOC', how='left') \
               .merge(medal_totals, on='NOC', how='left') \
               .merge(growth_metrics, on='NOC', how='left') \
               .merge(concentration, on='NOC', how='left')

# 补 NaN
df['ProjectDiversity'] = df['ProjectDiversity'].fillna(0).astype(int)
df['Shannon_Entropy'] = df['Shannon_Entropy'].fillna(0.0)
df[['Gold_total','Silver_total','Bronze_total']] = df[['Gold_total','Silver_total','Bronze_total']].fillna(0).astype(int)
df['sports_for_80pct'] = df['sports_for_80pct'].fillna(np.inf)

# ---------- medal_score 与 score ----------
current_year = medals['Year'].max()
medals['time_weight'] = 0.9 ** (current_year - medals['Year'])
medals['medal_score_raw'] = medals['Gold']*3 + medals['Silver']*2 + medals['Bronze']*1
medal_score_by_noc = medals.groupby('NOC').apply(lambda d: (d['medal_score_raw'] * d['time_weight']).sum()).reset_index().rename(columns={0:'medal_score'})
df = df.merge(medal_score_by_noc, on='NOC', how='left')
df['medal_score'] = df['medal_score'].fillna(0.0)

# ParticipationRate = distinct years with data / (last - first + 1)
df['ParticipationRate'] = df.apply(lambda r: r['participation_years'] / (r['last_year_all'] - r['first_year_all'] + 1) if (r['last_year_all'] - r['first_year_all'] + 1) > 0 else 0.0, axis=1)

# score 0-100 归一化
mn, mx = df['medal_score'].min(), df['medal_score'].max()
df['score'] = ((df['medal_score'] - mn) / (mx - mn) * 100).fillna(0.0)

# ---------- 分类逻辑（按照你的文字描述实现） ----------
def classify(r):
    # Traditional
    if (r['Total_all_time'] > 1000) and (r['first_year_all'] < 1950) and (r['ProjectDiversity'] > 20) and (pd.notna(r['cagr']) and r['cagr'] < 0.10):
        return 'Traditional powerhouses'
    # Emerging
    if (pd.notna(r['last3_growth']) and r['last3_growth'] > 0.5) and (pd.notna(r['first_medal_year']) and r['first_medal_year'] > 2000) and (r['sports_for_80pct'] <= 5) and (200 <= r['Total_all_time'] <= 500):
        return 'Emerging Olympic countries'
    # Non-medal
    if r['Total_all_time'] == 0:
        return 'Countries that have not yet won a medal'
    # Medium strength (fallback)
    if (200 < r['Total_all_time'] <= 1000) and not (pd.notna(r['last3_growth']) and r['last3_growth'] > 0.5) and not (pd.notna(r['cagr']) and r['cagr'] > 0.20):
        return 'MediumStrengthCountries'
    # 默认放入 MediumStrengthCountries
    return 'MediumStrengthCountries'

df['classification'] = df.apply(classify, axis=1)

# ---------- 准备导出表（列顺序与字段） ----------
def prepare_out(g):
    out = pd.DataFrame()
    out['Year'] = g['most_recent_year'].astype(int)
    out['NOC'] = g['NOC']
    out['score'] = g['score'].round(3)
    out['ParticipationRate'] = g['ParticipationRate'].round(3)
    out['Average_Previous_Medals'] = g['avg_previous_medals'].round(3)
    out['Shannon_Entropy'] = g['Shannon_Entropy'].round(3)
    out['Total number of gold medals'] = g['Gold_total'].astype(int)
    out['Total number of silver medals'] = g['Silver_total'].astype(int)
    out['Total number of bronze medals'] = g['Bronze_total'].astype(int)
    out['medal_score'] = g['medal_score'].round(3)
    return out

trad_df = prepare_out(df[df['classification']=='Traditional powerhouses'])
med_df = prepare_out(df[df['classification']=='MediumStrengthCountries'])
emerg_df = prepare_out(df[df['classification']=='Emerging Olympic countries'])
non_df = prepare_out(df[df['classification']=='Countries that have not yet won a medal'])

# 导出到 Excel
trad_df.to_excel('traditional powerhouses.xlsx', index=False)
med_df.to_excel('medium-level.xlsx', index=False)
emerg_df.to_excel('emerging countries.xlsx', index=False)
non_df.to_excel('non-medal countries.xlsx', index=False)

print("Export finished. Files created in current directory:")
print(" - traditional powerhouses.xlsx (rows: {})".format(len(trad_df)))
print(" - medium-level.xlsx (rows: {})".format(len(med_df)))
print(" - emerging countries.xlsx (rows: {})".format(len(emerg_df)))
print(" - non-medal countries.xlsx (rows: {})".format(len(non_df)))
