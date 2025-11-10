import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

warnings.filterwarnings("ignore")

# ============= CONFIG =============
EXCEL_PATH = "medium_level_corrected.xlsx"  # Input Excel file
OUTPUT_CSV = "medium_medal_predictions_gru.csv"  # Output results CSV
R2_PDF_PATH = "medium_r2_comparison_gru.pdf"  # R² chart output
RANDOM_SEED = 42
EPOCHS = 150
BATCH_SIZE = 8
# =================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def load_data(path):
    """Load and clean dataset"""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def build_gru_model(input_shape):
    """Build a GRU model for medal prediction"""
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def predict_features_2028(df_noc, features):
    """Predict input features for the year 2028 using simple linear trends"""
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

def fit_and_predict_gru(df_noc, features, target):
    """Train GRU model for a single NOC and a single medal type"""
    df_clean = df_noc.dropna(subset=features + [target])
    if len(df_clean) < 5:
        return np.nan, np.nan, np.nan, np.nan

    df_clean = df_clean.sort_values("Year")
    X = df_clean[features].values
    y = df_clean[target].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Prepare time series sequences for GRU input
    X_seq, y_seq = [], []
    seq_len = 3
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    if len(X_seq) < 3:
        return np.nan, np.nan, np.nan, np.nan

    model = build_gru_model((seq_len, X_seq.shape[2]))
    model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Compute R² on training sequence
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq)
    r2 = r2_score(y_true, y_pred)

    # Predict for year 2028
    feat_2028 = predict_features_2028(df_clean, features)
    x_2028 = np.array([feat_2028[f] for f in features]).reshape(1, -1)
    x_2028_scaled = scaler_X.transform(x_2028)

    # Use the last seq_len years as input for future prediction
    last_seq = X_scaled[-seq_len:]
    new_seq = np.concatenate([last_seq[1:], x_2028_scaled], axis=0)
    new_seq = new_seq.reshape(1, seq_len, X_seq.shape[2])

    preds = [scaler_y.inverse_transform(model.predict(new_seq, verbose=0))[0][0]
             for _ in range(100)]
    mean_pred = np.mean(preds)
    low, high = np.percentile(preds, [2.5, 97.5])
    return r2, max(0, mean_pred), max(0, low), max(0, high)

def main():
    df = load_data(EXCEL_PATH)

    feature_cols = ['score', 'ParticipationRate', 'Average_Previous_Medals', 'Shannon_Entropy']
    targets = {
        "Gold": "Total number of gold medals",
        "Silver": "Total number of silver medals",
        "Bronze": "Total number of bronze medals",
        "Total": "medal_score_y"
    }

    nocs_2024 = df.loc[df["Year"] == 2024, "NOC"].unique()
    results, r2_records, all_results = [], [], []

    print(f"Processing {len(nocs_2024)} NOCs using GRU...")

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

        for medal_type in ["Gold", "Silver", "Bronze", "Total"]:
            if f"{medal_type}_r2" in r2_entry:
                all_results.append({
                    "NOC": noc,
                    "Medal Type": medal_type,
                    "R² Value": r2_entry[f"{medal_type}_r2"]
                })

    # === Save model predictions ===
    df_res = pd.DataFrame(results)
    df_r2 = pd.DataFrame(r2_records)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved GRU prediction summary to {OUTPUT_CSV}")

    # === Visualize R² values ===
    if all_results:
        r2_df = pd.DataFrame(all_results)
        medal_types = ['Gold', 'Silver', 'Bronze', 'Total']
        r2_filtered = r2_df[r2_df['Medal Type'].isin(medal_types)].copy()

        plt.figure(figsize=(14, 8))
        palette_colors = {
            'Gold': '#FFD700',
            'Silver': '#C0C0C0',
            'Bronze': '#CD7F32',
            'Total': '#4682B4'
        }

        ax = sns.barplot(
            data=r2_filtered,
            x='NOC',
            y='R² Value',
            hue='Medal Type',
            palette=palette_colors,
            edgecolor='black',
            linewidth=1.0
        )

        # Add labels above bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3,
                         fontsize=9, color='black', weight='bold')

        plt.title('GRU Model Performance: R² by NOC and Medal Type',
                  fontsize=18, fontweight='bold', pad=25, color='navy')
        plt.xlabel('National Olympic Committee (NOC)', fontsize=14, fontweight='bold')
        plt.ylabel('R² Value', fontsize=14, fontweight='bold')

        plt.xticks(rotation=25, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Medal Type', loc='center left', bbox_to_anchor=(1.02, 0.5),
                   fontsize=11, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        plt.savefig(R2_PDF_PATH, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Saved GRU R² comparison chart to {R2_PDF_PATH}")
    else:
        print("✗ No valid results to visualize.")

if __name__ == "__main__":
    main()
