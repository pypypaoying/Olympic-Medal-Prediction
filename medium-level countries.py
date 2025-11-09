import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (14, 8)

# Create results directory
import os

os.makedirs('results_optimized', exist_ok=True)

# Load and prepare data
try:
    df = pd.read_excel('medium-level.xlsx', sheet_name='Sheet1')
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"✗ Error loading Excel file: {e}")
    exit()

# Clean and prepare data
df = df.dropna(subset=['Year', 'NOC', "Total number of gold medals", "Total number of silver medals", "Total number of bronze medals"])
df = df.sort_values(['NOC', 'Year'])
nocs = df['NOC'].unique()
print(f"✓ Found {len(nocs)} unique NOCs: {', '.join(nocs)}")


# Enhanced feature engineering function
def create_features(noc_df):
    """Create rich time series features for Olympic prediction"""
    df = noc_df.copy()

    # Basic time features
    df['Year_Squared'] = df['Year'] ** 2
    df['Olympic_Cycle'] = df['Year'] % 4

    # Lag features (previous performances)
    for medal in ['Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals']:
        for lag in [1, 2, 4]:  # 4-year and 8-year lags are meaningful for Olympics
            df[f'{medal}_lag_{lag}'] = df[medal].shift(lag)

    # Rolling statistics
    for medal in ['Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals']:
        df[f'{medal}_rolling_mean_2'] = df[medal].rolling(window=2, min_periods=1).mean()
        df[f'{medal}_rolling_std_2'] = df[medal].rolling(window=2, min_periods=1).std()

    # Medal ratios and interactions
    df['Total_Medals'] = df['Total number of gold medals'] + df['Total number of silver medals'] + df['Total number of bronze medals']
    df['Gold_Ratio'] = df['Total number of gold medals'] / df['Total_Medals'].replace(0, np.nan)
    df['Silver_Ratio'] = df['Total number of silver medals'] / df['Total_Medals'].replace(0, np.nan)

    # Time-based trends
    df['Years_Since_First'] = df['Year'] - df['Year'].min()

    # Fill NaN values with reasonable defaults
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df


# Comprehensive model training and evaluation
def train_and_evaluate_models(noc_df, noc_name):
    """Train multiple models and select the best one for each medal type"""
    results = []
    feature_importances = {}

    # Create features
    df_features = create_features(noc_df)
    years = df_features['Year'].values

    # Define target columns
    targets = ['Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals']
    target_names = ['Gold', 'Silver', 'Bronze']

    # Define features to use (exclude targets and year)
    feature_cols = [col for col in df_features.columns if col not in targets + ['Year', 'NOC']]

    # Prepare data
    X = df_features[feature_cols].values
    y_dict = {target: df_features[target].values for target in targets}

    # Split data chronologically (last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    years_train, years_test = years[:split_idx], years[split_idx:]

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Evaluate each target separately
    for target, target_name in zip(targets, target_names):
        y = y_dict[target]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Skip if target has no variation
        if np.std(y_train) < 0.1:
            continue

        # Scale target for neural networks
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Dictionary to store model performances
        model_performances = {}
        predictions = {}

        print(f"\n{'=' * 60}")
        print(f"NOC: {noc_name} | Target: {target_name} Medals")
        print(f"Training data points: {len(X_train)}, Test data points: {len(X_test)}")
        print(f"Features used: {len(feature_cols)}")

        # 1. Simple Linear Regression (baseline)
        try:
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            model_performances['Linear Regression'] = r2
            predictions['Linear Regression'] = y_pred
            print(f"Linear Regression R²: {r2:.4f}")
        except Exception as e:
            print(f"Linear Regression failed: {e}")

        # 2. Ridge Regression (handles multicollinearity)
        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            model_performances['Ridge Regression'] = r2
            predictions['Ridge Regression'] = y_pred
            print(f"Ridge Regression R²: {r2:.4f}")
        except Exception as e:
            print(f"Ridge Regression failed: {e}")

        # 3. Random Forest (handles non-linearity well)
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                       min_samples_leaf=2, max_features='sqrt')
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            model_performances['Random Forest'] = r2
            predictions['Random Forest'] = y_pred
            feature_importances[f'{target_name}_RF'] = dict(zip(feature_cols, rf.feature_importances_))
            print(f"Random Forest R²: {r2:.4f}")
        except Exception as e:
            print(f"Random Forest failed: {e}")

        # 4. Gradient Boosting (often best for tabular data)
        try:
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                           max_depth=3, random_state=42,
                                           subsample=0.8, min_samples_leaf=2)
            gb.fit(X_train_scaled, y_train)
            y_pred = gb.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            model_performances['Gradient Boosting'] = r2
            predictions['Gradient Boosting'] = y_pred
            print(f"Gradient Boosting R²: {r2:.4f}")
        except Exception as e:
            print(f"Gradient Boosting failed: {e}")

        # 5. XGBoost (state-of-the-art gradient boosting)
        try:
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1,
                               max_depth=4, random_state=42,
                               subsample=0.8, colsample_bytree=0.8,
                               tree_method='hist', eval_metric='rmse')
            xgb.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=10, verbose=False)
            y_pred = xgb.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            model_performances['XGBoost'] = r2
            predictions['XGBoost'] = y_pred
            print(f"XGBoost R²: {r2:.4f}")
        except Exception as e:
            print(f"XGBoost failed: {e}")

        # 6. LSTM (for time series patterns)
        try:
            # Reshape for LSTM [samples, timesteps, features]
            X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

            model = Sequential()
            model.add(Bidirectional(LSTM(32, return_sequences=True),
                                    input_shape=(1, X_train_scaled.shape[1])))
            model.add(Dropout(0.3))
            model.add(LSTM(16))
            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

            early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                       restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=5, min_lr=1e-6)

            model.fit(X_train_lstm, y_train_scaled,
                      validation_split=0.2, epochs=100, batch_size=4,
                      callbacks=[early_stop, reduce_lr], verbose=0)

            y_pred_scaled = model.predict(X_test_lstm).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            r2 = r2_score(y_test, y_pred)
            model_performances['LSTM'] = r2
            predictions['LSTM'] = y_pred
            print(f"LSTM R²: {r2:.4f}")
        except Exception as e:
            print(f"LSTM failed: {e}")

        # Select best model
        if model_performances:
            best_model_name = max(model_performances.items(), key=lambda x: x[1])[0]
            best_r2 = model_performances[best_model_name]
            best_pred = predictions[best_model_name]

            print(f"\n✓ Best model for {noc_name} {target_name}: {best_model_name} (R²: {best_r2:.4f})")

            # Store results
            results.append({
                'NOC': noc_name,
                'Medal Type': target_name,
                'Best Model': best_model_name,
                'R² Value': best_r2,
                'RMSE': np.sqrt(mean_squared_error(y_test, best_pred)),
                'Years': years_test,
                'Actual': y_test,
                'Predicted': best_pred
            })
        else:
            print(f"✗ No models trained successfully for {noc_name} {target_name}")

    return results, feature_importances


# Process each NOC and train models
all_results = []
all_feature_importances = {}

for noc in nocs:
    noc_df = df[df['NOC'] == noc].copy()

    if len(noc_df) < 8:  # Need sufficient data for feature engineering
        print(f"\n✗ Skipping {noc} - insufficient data points ({len(noc_df)})")
        continue

    print(f"\n{'=' * 80}")
    print(f"PROCESSING NOC: {noc.upper()}")
    print(f"{'=' * 80}")

    noc_results, noc_importances = train_and_evaluate_models(noc_df, noc)
    all_results.extend(noc_results)
    all_feature_importances.update(noc_importances)

# Generate R² comparison chart in English
# Generate R² comparison chart (Gold and Total only, output PDF)
# Generate R² comparison chart (Gold only, output as PDF)
if all_results:
    r2_df = pd.DataFrame(all_results)

    # 🔹 Keep only "Gold" medal type
    r2_gold = r2_df[r2_df['Medal Type'] == 'Gold'].copy()

    if r2_gold.empty:
        print("✗ No Gold medal R² data found. Check if models trained successfully.")
    else:
        plt.figure(figsize=(14, 8))

        # Bar plot for Gold R² values
        ax = sns.barplot(
            data=r2_gold,
            x='NOC',
            y='R² Value',
            color='#FFD700',  # gold color for clarity
            edgecolor='black',
            linewidth=1.0,
            alpha=0.9
        )

        # Add R² labels above bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge', padding=5,
                         fontsize=10, fontweight='bold', color='darkblue')

        # Title and labels
        plt.title('Model Performance: Gold Medal R² by NOC',
                  fontsize=18, fontweight='bold', pad=25, color='navy')
        plt.xlabel('National Olympic Committee (NOC)', fontsize=14, fontweight='bold')
        plt.ylabel('R² Value (Coefficient of Determination)', fontsize=14, fontweight='bold')

        # Y-axis limit
        max_r2 = min(1.0, r2_gold['R² Value'].max() + 0.1)
        plt.ylim(0, max_r2)

        plt.xticks(rotation=25, fontsize=12)
        plt.yticks(fontsize=12)

        # Grid and layout
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()

        # 🔹 Save to PDF (vector, editable)
        output_pdf = 'results_optimized/r2_comparison_gold_only.pdf'
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\n{'=' * 60}")
        print(f"✓ Gold Medal R² chart saved as PDF → {output_pdf}")

        # Save corresponding data
        r2_gold.to_csv('results_optimized/model_performance_gold_only.csv', index=False)
        print("✓ Gold medal model performance data saved to: results_optimized/model_performance_gold_only.csv")

else:
    print("✗ No valid results to visualize. Check data quality and model configuration.")



# Generate feature importance visualization if available
if all_feature_importances:
    try:
        plt.figure(figsize=(15, 10))

        # Get top 15 most important features across all models
        all_importances = {}
        for key, imp_dict in all_feature_importances.items():
            for feature, importance in imp_dict.items():
                if feature not in all_importances or importance > all_importances[feature]:
                    all_importances[feature] = importance

        # Sort and select top features
        top_features = dict(sorted(all_importances.items(),
                                   key=lambda x: x[1], reverse=True)[:15])

        # Create horizontal bar chart
        features = list(top_features.keys())
        importances = list(top_features.values())

        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, align='center', color='skyblue', edgecolor='navy')
        plt.yticks(y_pos, features, fontsize=12)
        plt.xlabel('Feature Importance Score', fontsize=14, fontweight='bold')
        plt.title('Top 15 Most Important Features for Olympic Medal Prediction',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        plt.savefig('results_optimized/feature_importance.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        print("✓ Feature importance chart saved to: results_optimized/feature_importance.png")

    except Exception as e:
        print(f"Feature importance visualization failed: {e}")
