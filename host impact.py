import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)


# =====================================================
# 1. Data Loading and Cleaning
# =====================================================
def load_and_clean_data(from_year=1980, exclude_boycott=True):
    """Load, harmonize, and prepare Olympic medal data."""
    medal = pd.read_csv('summerOly_medal_counts.csv')
    hosts = pd.read_csv('summerOly_hosts.csv')

    # Extract host country names
    hosts['Host_Country'] = hosts['Host'].str.extract(r',\s*([^,]+)$')
    hosts = hosts[hosts['Host_Country'].notna()]

    # Country harmonization
    country_mapping = {
        'Soviet Union': 'Russia', 'URS': 'RUS',
        'West Germany': 'Germany', 'FRG': 'GER',
        'East Germany': 'Germany', 'GDR': 'GER',
        'Czechoslovakia': 'Czech Republic', 'TCH': 'CZE',
        'Yugoslavia': 'Serbia', 'YUG': 'SRB',
        'Serbia and Montenegro': 'Serbia', 'SCG': 'SRB'
    }
    medal['NOC'] = medal['NOC'].replace(country_mapping)

    # Merge hosting flag
    medal = pd.merge(medal, hosts[['Year', 'Host_Country']], on='Year', how='left')
    medal['Is_Host'] = (medal['NOC'] == medal['Host_Country']).astype(int)

    # Total medals
    medal['Total'] = medal[['Gold', 'Silver', 'Bronze']].sum(axis=1)
    medal = medal.sort_values(['NOC', 'Year'])

    # Add one- and two-period lags
    for col in ['Gold', 'Silver', 'Bronze', 'Total']:
        medal[f'Past_{col}'] = medal.groupby('NOC')[col].shift(1)
        medal[f'Past2_{col}'] = medal.groupby('NOC')[col].shift(2)

    # Filter by year
    medal = medal[medal['Year'] >= from_year]
    if exclude_boycott:
        medal = medal[~medal['Year'].isin([1980, 1984])]

    # Winsorize medal counts (cap extreme values)
    for c in ['Gold', 'Silver', 'Bronze', 'Total']:
        cap = medal[c].quantile(0.99)
        medal[c] = np.minimum(medal[c], cap)

    return medal


# =====================================================
# 2. Linear Log FE Model
# =====================================================
def fit_linear_log_fe(df, medal_type):
    """Fit log-linear fixed-effects model using PanelOLS."""
    d = df.dropna(subset=[medal_type, 'Is_Host', f'Past_{medal_type}', f'Past2_{medal_type}']).copy()

    # Add 1 to avoid log(0)
    d[f'ln_{medal_type}'] = np.log1p(d[medal_type])
    panel_data = d.set_index(['NOC', 'Year'])

    exog_vars = ['Is_Host', f'Past_{medal_type}', f'Past2_{medal_type}', 'Past_Total', 'Past2_Total']
    exog = sm.add_constant(panel_data[exog_vars])

    model = PanelOLS(panel_data[f'ln_{medal_type}'], exog, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    return res


# =====================================================
# 3. Conditional Poisson FE Model
# =====================================================
def fit_conditional_poisson_fe(df, medal_type):
    """Approximate conditional Poisson fixed effects with clustering."""
    d = df.dropna(subset=[medal_type, 'Is_Host', f'Past_{medal_type}', f'Past2_{medal_type}']).copy()

    formula = f"{medal_type} ~ Is_Host + Past_{medal_type} + Past2_{medal_type} + Past_Total + Past2_Total + C(NOC) + C(Year)"
    model = smf.glm(formula=formula, data=d, family=sm.families.Poisson())
    res = model.fit(cov_type='cluster', cov_kwds={'groups': d['NOC']})
    return res


# =====================================================
# 4. Comparison Summary
# =====================================================
def compare_models(df, medal_types):
    summary = []
    for medal_type in medal_types:
        lin_res = fit_linear_log_fe(df, medal_type)
        pois_res = fit_conditional_poisson_fe(df, medal_type)

        lin_coef = lin_res.params['Is_Host']
        lin_ci = lin_res.conf_int().loc['Is_Host']
        pois_coef = pois_res.params['Is_Host']
        pois_ci = pois_res.conf_int().loc['Is_Host']

        lin_effect = np.expm1(lin_coef)
        pois_effect = np.expm1(pois_coef)
        lin_lower = np.expm1(lin_ci[0])
        lin_upper = np.expm1(lin_ci[1])
        pois_lower = np.expm1(pois_ci[0])
        pois_upper = np.expm1(pois_ci[1])

        summary.append({
            "Medal": medal_type,
            "Linear_log_coef": lin_coef,
            "Poisson_coef": pois_coef,
            "Linear_effect_%": lin_effect * 100,
            "Poisson_effect_%": pois_effect * 100,
            "Linear_CI": f"[{lin_lower*100:.1f}%, {lin_upper*100:.1f}%]",
            "Poisson_CI": f"[{pois_lower*100:.1f}%, {pois_upper*100:.1f}%]"
        })
    return pd.DataFrame(summary)


# =====================================================
# 5. Main Runner
# =====================================================
def main():
    print("Loading and cleaning data...")
    medal_data = load_and_clean_data()

    print("\nEstimating host effects for each medal type...\n")
    comparison = compare_models(medal_data, ['Gold', 'Silver', 'Bronze'])

    print("=== Host Advantage Comparison ===")
    for _, row in comparison.iterrows():
        print(f"\n{row['Medal']}:")
        print(f"  Linear-log FE -> Host effect: {row['Linear_effect_%']:.1f}%  {row['Linear_CI']}")
        print(f"  Poisson FE    -> Host effect: {row['Poisson_effect_%']:.1f}%  {row['Poisson_CI']}")
        print(f"  (raw coefs) Linear_log: {row['Linear_log_coef']:.4f}, Poisson: {row['Poisson_coef']:.4f}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
