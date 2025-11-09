import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Read data
df = pd.read_excel('non-medal countries.xlsx', sheet_name='Sheet1')

# Clean column names
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)

# Verify required columns exist
required_columns = ['Year', 'NOC', 'score', 'ParticipationRate', 'Average_Previous_Medals', 'Shannon_Entropy']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Calculate R-squared values for each country's polynomial fit
r2_values = []
predictions_2028 = []

for noc, group in df.groupby('NOC'):
    if len(group) < 3:  # Need at least 3 points for quadratic fit
        continue

    X = group[['Year']].values
    y = group[['score']].values  # Focus on score for R-squared calculation

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Calculate R-squared
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    r2_values.append(r2)

    # Predict for 2028
    X_2028 = poly.transform(np.array([[2028]]))
    pred_2028 = model.predict(X_2028)[0][0]
    predictions_2028.append({'NOC': noc, 'score_2028': pred_2028})

# Create R-squared distribution plot
plt.figure(figsize=(10, 6))
plt.hist(r2_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(r2_values), color='red', linestyle='--',
            label=f'Mean R² = {np.mean(r2_values):.3f}')
plt.title('Distribution of R-squared Values for Polynomial Fits', fontsize=14)
plt.xlabel('R-squared Value', fontsize=12)
plt.ylabel('Number of Countries', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('r2_distribution.png', dpi=300)
plt.show()

# Create synthetic binary classification problem for ROC/AUC
# (Threshold based on score distribution)
threshold = df['score'].quantile(0.75)  # Use 75th percentile as threshold
df['synthetic_target'] = (df['score'] > threshold).astype(int)

# Prepare features and target
X = df[['score', 'ParticipationRate', 'Average_Previous_Medals', 'Shannon_Entropy']]
y = df['synthetic_target']

# Train logistic regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X, y)

# Calculate ROC curve and AUC
y_scores = logreg.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)

# Create ROC/AUC plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve for Synthetic Classification Task', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

output_pdf = 'AUC.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
plt.show()

print(f"Mean R-squared value across all countries: {np.mean(r2_values):.4f}")
print(f"ROC AUC score: {roc_auc:.4f}")