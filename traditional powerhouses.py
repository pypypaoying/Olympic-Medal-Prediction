import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import os
import warnings

# Set styling and suppress warnings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# Load and prepare data
try:
    df = pd.read_excel('traditional powerhouses.xlsx', sheet_name='Sheet1')
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Clean data
df_clean = df.dropna(subset=['Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals', 'Total number of medals', 'Year', 'NOC'])
top_nocs = df_clean['NOC'].value_counts().index[:6].tolist()
df_filtered = df_clean[df_clean['NOC'].isin(top_nocs)]

results = {}

# Parameter grid for model optimization
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3],
    'subsample': [0.9, 1.0]
}

# Train models and calculate R² values
for noc in top_nocs:
    noc_data = df_filtered[df_filtered['NOC'] == noc].copy()
    X = noc_data[['Year']]

    if len(noc_data) < 5:
        continue

    targets = {
        'Gold': 'Total number of gold medals',
        'Silver': 'Total number of silver medals',
        'Bronze': 'Total number of bronze medals',
        'Total': 'Total number of medals'
    }

    noc_results = {}

    for medal_type, col_name in targets.items():
        y = noc_data[col_name]

        # Skip if target has no variation
        if y.nunique() < 2:
            continue

        try:
            # Train regression model
            model = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            model.fit(X, y)

            # Calculate R²
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            noc_results[medal_type] = r2

        except Exception as e:
            continue

    if noc_results:
        results[noc] = noc_results

# Generate R² comparison chart (English only)
if results:
    # Prepare data for visualization
    r2_data = []
    for noc, metrics in results.items():
        for medal_type, r2_value in metrics.items():
            r2_data.append({
                'NOC': noc,
                'Medal Type': medal_type,
                'R² Value': r2_value
            })

    r2_df = pd.DataFrame(r2_data)

    if not r2_df.empty:
        plt.figure(figsize=(14, 8))

        # Create bar chart
        ax = sns.barplot(
            data=r2_df,
            x='NOC',
            y='R² Value',
            hue='Medal Type',
            edgecolor='black',
            linewidth=0.5
        )

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9)

        # Set chart titles and labels in English
        plt.title('R² Values by NOC and Medal Type', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('National Olympic Committee (NOC)', fontsize=12)
        plt.ylabel('R² Value (Coefficient of Determination)', fontsize=12)
        plt.ylim(0, min(1.1, r2_df['R² Value'].max() * 1.2))
        plt.xticks(rotation=15)
        plt.legend(title='Medal Type', title_fontsize=11, fontsize=10)
        plt.tight_layout()

        # Save the chart
        plt.savefig('R²(traditional powerhouses).pdf',  bbox_inches='tight', format='pdf', facecolor='white')
        plt.close()

        print("✓ R² comparison chart successfully saved to: results/r2_comparison_en.png")
        print(f"Chart includes {len(r2_df)} data points across {len(top_nocs)} NOCs")
else:
    print("No valid results to visualize. Please check your data and model configuration.")