import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============= CONFIG =============
EXCEL_PATH = "emerging countries.xlsx"
OUTPUT_CSV = "medal_predictions.csv"
R2_PDF_PATH = "typical_r2_plot.pdf"  # ← Save to PDF
RF_ESTIMATORS = 300
RANDOM_SEED = 42
# =================================

def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def predict_features_2028(df_noc, features):
    years = df_noc["Year"].values.reshape(-1, 1)
    preds = {}
    for f in features:
        vals = df_noc[f].values
        if len(np.unique(years)) < 2 or np.isnan(vals).all():
            preds[f] = np.nanmean(vals)
            continue
        lr = LinearRegression()
        lr.fit(years, vals)
        preds[f] = lr.predict([[2028]])[0]
    return preds

def fit_and_predict(df_noc, features, target):
    df_clean = df_noc.dropna(subset=features + [target])
    if len(df_clean) < 3:
        return np.nan, np.nan, np.nan, np.nan
    X = df_clean[features].values
    y = df_clean[target].values

    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED)
    rf.fit(X, y)
    r2 = r2_score(y, rf.predict(X))

    feat_2028 = predict_features_2028(df_clean, features)
    x_2028 = np.array([feat_2028[f] for f in features]).reshape(1, -1)
    tree_preds = np.array([tree.predict(x_2028)[0] for tree in rf.estimators_])
    mean_pred = np.mean(tree_preds)
    low, high = np.percentile(tree_preds, [2.5, 97.5])
    return r2, max(0, mean_pred), max(0, low), max(0, high)

def main():
    df = load_data(EXCEL_PATH)

    feature_cols = ['score', 'ParticipationRate', 'Average_Previous_Medals', 'Shannon_Entropy']
    targets = {
        "Gold": "Total number of gold medals",
        "Silver": "Total number of silver medals",
        "Bronze": "Total number of bronze medals",
        "Total": "medal_score"
    }

    nocs_2024 = df.loc[df["Year"] == 2024, "NOC"].unique()
    results, r2_records = [], []

    print(f"Processing {len(nocs_2024)} NOCs...")

    for noc in nocs_2024:
        df_noc = df[df["NOC"] == noc]
        rec = {"NOC": noc}
        r2_entry = {"NOC": noc}

        for medal, col in targets.items():
            r2, mean, low, high = fit_and_predict(df_noc, feature_cols, col)
            rec.update({
                f"{medal}_mean": mean,
                f"{medal}_low95": low,
                f"{medal}_high95": high
            })
            r2_entry[f"{medal}_r2"] = r2

        results.append(rec)
        r2_records.append(r2_entry)

    df_res = pd.DataFrame(results)
    df_r2 = pd.DataFrame(r2_records)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved prediction summary to {OUTPUT_CSV}")

    # --- Select top 3 countries by Total R² ---
    df_r2_sorted = df_r2.sort_values("Total_r2", ascending=False).head(3)
    print("\nTypical NOCs by Total R²:")
    print(df_r2_sorted[["NOC", "Gold_r2", "Silver_r2", "Bronze_r2", "Total_r2"]])

    # --- Plot R² for top 3 only ---
    x = np.arange(len(df_r2_sorted))
    width = 0.18

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, df_r2_sorted["Gold_r2"], width, label="Gold R²")
    plt.bar(x - 0.5*width, df_r2_sorted["Silver_r2"], width, label="Silver R²")
    plt.bar(x + 0.5*width, df_r2_sorted["Bronze_r2"], width, label="Bronze R²")
    plt.bar(x + 1.5*width, df_r2_sorted["Total_r2"], width, label="Total R²")

    plt.xticks(x, df_r2_sorted["NOC"], rotation=15, ha="center", fontsize=12)
    plt.xlabel("Typical NOCs (by Total R²)", fontsize=13)
    plt.ylabel("R² (Coefficient of Determination)", fontsize=13)
    plt.title("Countries with the Best Model Fit (R²)", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    # --- Save directly to PDF ---
    plt.savefig(R2_PDF_PATH, format="pdf")
    plt.close()

    print(f"✅ Saved R² plot to {R2_PDF_PATH}")

if __name__ == "__main__":
    main()