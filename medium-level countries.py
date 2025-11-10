import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

warnings.filterwarnings("ignore")

# ================= CONFIG =================
EXCEL_PATH = "medium-level countries.xlsx"
OUTPUT_CSV = "medium_medal_predictions_gru_optimized.csv"
R2_PDF_PATH = "medium_r2_comparison_gru_optimized.pdf"
RANDOM_SEED = 42
SEQ_LEN = 5
EPOCHS = 300
BATCH_SIZE = 8
# =========================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------- Load data ----------------
def load_data(path):
    """Load Excel file and clean column names."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------- Feature Engineering ----------------
def create_features(df):
    """Generate rich features including lags, rolling stats, ratios, and time features."""
    df = df.sort_values(['NOC', 'Year']).copy()

    # Time features
    df['Year_Squared'] = df['Year'] ** 2
    df['Olympic_Cycle'] = df['Year'] % 4

    # Medal lag features
    medals = ['Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals']
    for medal in medals:
        for lag in [1, 2, 4]:
            df[f'{medal}_lag_{lag}'] = df.groupby('NOC')[medal].shift(lag)
        # Rolling mean and std
        df[f'{medal}_rolling_mean_2'] = df.groupby('NOC')[medal].rolling(2, min_periods=1).mean().reset_index(0,
                                                                                                              drop=True)
        df[f'{medal}_rolling_std_2'] = df.groupby('NOC')[medal].rolling(2, min_periods=1).std().reset_index(0,
                                                                                                            drop=True)

    # Total medals and ratios
    df['Total_Medals'] = df['Total number of gold medals'] + df['Total number of silver medals'] + df['Total number of bronze medals']
    df['Gold_Ratio'] = df['Total number of gold medals'] / df['Total_Medals'].replace(0, np.nan)
    df['Silver_Ratio'] = df['Total number of silver medals'] / df['Total_Medals'].replace(0, np.nan)

    # Years since first participation
    df['Years_Since_First'] = df.groupby('NOC')['Year'].transform(lambda x: x - x.min())

    # Fill NaNs
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df


# --------------- GRU Model ----------------
def build_gru_model(input_shape):
    """Build two-layer GRU model with dropout."""
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# --------------- 2028 Feature Prediction ----------------
def predict_features_2028(df_noc, features):
    """Predict 2028 features using linear trends."""
    years = df_noc["Year"].values.reshape(-1, 1)
    preds = {}
    for f in features:
        vals = df_noc[f].values
        if len(np.unique(years)) < 2 or np.isnan(vals).all():
            preds[f] = np.nanmean(vals)
            continue
        coeffs = np.polyfit(years.flatten(), vals, 1)
        preds[f] = np.polyval(coeffs, 2028)
    return preds


# --------------- GRU Fit & Predict ----------------
def fit_and_predict_gru(df_noc, features, target):
    """Train GRU for one NOC and medal type, return R² and 2028 prediction with CI."""
    df_clean = df_noc.dropna(subset=features + [target]).sort_values('Year')
    if len(df_clean) < SEQ_LEN + 2:
        return np.nan, np.nan, np.nan, np.nan

    X = df_clean[features].values
    y = df_clean[target].values.reshape(-1, 1)

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Prepare sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_seq.append(y_scaled[i + SEQ_LEN])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Train GRU
    model = build_gru_model((SEQ_LEN, X_seq.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
    model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop, reduce_lr])

    # Compute R² on training sequence
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq)
    r2 = r2_score(y_true, y_pred)

    # Predict 2028
    feat_2028 = predict_features_2028(df_clean, features)
    x_2028 = np.array([feat_2028[f] for f in features]).reshape(1, -1)
    x_2028_scaled = scaler_X.transform(x_2028)

    last_seq = X_scaled[-SEQ_LEN:]
    new_seq = np.concatenate([last_seq[1:], x_2028_scaled], axis=0).reshape(1, SEQ_LEN, X_seq.shape[2])

    # Ensemble prediction for uncertainty
    preds = [scaler_y.inverse_transform(model.predict(new_seq, verbose=0))[0][0] for _ in range(100)]
    mean_pred = np.mean(preds)
    low, high = np.percentile(preds, [2.5, 97.5])

    return r2, max(0, mean_pred), max(0, low), max(0, high)


# --------------- Main Function ----------------
def main():
    df = load_data(EXCEL_PATH)
    df = create_features(df)

    feature_cols = [c for c in df.columns if
                    c not in ['NOC', 'Year', 'Total number of gold medals', 'Total number of silver medals', 'Total number of bronze medals', 'medal_score']]
    targets = {
        "Gold": "Total number of gold medals",
        "Silver": "Total number of silver medals",
        "Bronze": "Total number of bronze medals",
        "Total": "medal_score"
    }

    nocs_2024 = df.loc[df["Year"] == 2024, "NOC"].unique()
    results, r2_records, all_results = [], [], []

    print(f"Processing {len(nocs_2024)} NOCs with optimized GRU...")

    for noc in nocs_2024:
        df_noc = df[df["NOC"] == noc]
        rec = {"NOC": noc}
        r2_entry = {"NOC": noc}

        for medal, col in targets.items():
            r2, mean, low, high = fit_and_predict_gru(df_noc, feature_cols, col)
            rec.update({
                f"{medal}_mean": mean,
                f"{medal}_low95": low,
                f"{medal}_high95": high
            })
            r2_entry[f"{medal}_r2"] = r2

        results.append(rec)
        r2_records.append(r2_entry)

        for medal_type in targets.keys():
            if f"{medal_type}_r2" in r2_entry:
                all_results.append({
                    "NOC": noc,
                    "Medal Type": medal_type,
                    "R² Value": r2_entry[f"{medal_type}_r2"]
                })

    # Save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved optimized GRU predictions to {OUTPUT_CSV}")

    # Plot R²
    if all_results:
        r2_df = pd.DataFrame(all_results)
        palette_colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32', 'Total': '#4682B4'}
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=r2_df,
            x='NOC',
            y='R² Value',
            hue='Medal Type',
            palette=palette_colors,
            edgecolor='black',
            linewidth=1.0
        )

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3,
                         fontsize=9, color='black', weight='bold')

        plt.title('Optimized GRU Model R² by NOC and Medal Type', fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('NOC', fontsize=14, fontweight='bold')
        plt.ylabel('R² Value', fontsize=14, fontweight='bold')
        plt.xticks(rotation=25, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Medal Type', loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(R2_PDF_PATH, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved optimized GRU R² chart to {R2_PDF_PATH}")


if __name__ == "__main__":
    main()
