# Olympic Medal Prediction

This repository contains code for predicting Olympic medal counts using tier-specific machine learning models, as described in our research paper. The models categorize countries into four tiers (Traditional Powerhouses, Medium-Level Nations, Emerging Nations, and Non-Medal Nations) and apply tailored algorithms to forecast performance, incorporating factors like historical data and host effects. Predictions are demonstrated for the 2028 Los Angeles Olympic Games.

## Overview

The prediction pipeline uses historical Olympic data (1866-2024) to forecast future medal tallies. Key inputs include:
- **Historical Medal Count**: Total medals from past Games.
- **Host Status**: Binary indicator for hosting the event (impacts performance via "host effect").
- **Event Diversity**: Sport category participation and variety.
- **Representative Coaches**: Proxy for training quality and investment.

The architecture employs a hierarchical approach:
1. **Panel Regression**: Estimates host effects using Difference-in-Differences (DiD).
2. **XGBoost Classification**: Classifies countries into one of four tiers.
3. **Tier-Specific Models** (Machine Learning):
   - **Traditional Powerhouses**: Gradient Boosting Decision Trees (GBDT).
   - **Medium-Level Nations**: Gated Recurrent Unit (GRU).
   - **Emerging Nations**: Random Forest (RF).
   - **Non-Medal Nations**: Logistic Regression.
4. **Forecast Results**: Aggregated predictions with confidence intervals (e.g., 95% CI ±4.2-23.9).

Example forecasts for 2028:
- USA: 128 medals (+2)
- CHN: 85 medals (-6)
- JPN: 81 medals (+38)
- ... (56 nations will win their first medal)

This setup achieves high accuracy by customizing models to each tier's characteristics.

<img width="1120" height="473" alt="030879cd227aa67fcc1dff19bde67b79" src="https://github.com/user-attachments/assets/8d1a6c32-d8c3-417a-a6b5-ebbfcd59481d" />

## Installation

1. Clone the repository:git clone https://github.com/ppypaoying/Olympic-Medal-Prediction.git
cd Olympic-Medal-Prediction
2. Install dependencies:
pip install -r requirements.txt
(Requires Python 3.8+; key packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch` for GRU.)

3. Download data (if not included):
- Place historical data in `data/` (e.g., `historical_medals.csv` from Olympics.org or similar sources).

## Usage

Run the full pipeline:python main.py --data_path data/historical_medals.csv --output forecasts/2028_predictions.csv
- **Tier-Specific Scripts**:
  - `traditional_powerhouses.py`: Train/predict for powerhouses using GBDT.
  - `medium_level_countries.py`: Use GRU for medium-level nations.
  - `emerging_countries.py`: RF for emerging nations.
  - `non_medal_countries.py`: Logistic Regression for non-medal nations.
  - `classification.py`: XGBoost for initial tier classification.

For evaluation:
python evaluate.py --model_path models/gbdt_model.pkl --test_data data/test.csv
Outputs include CSV files with predicted medal counts and confidence intervals.

## Data

- Sources: Historical Olympic medal data (publicly available).
- Preprocessing: Handled in `data_preprocessing.py` (normalization, feature engineering for event diversity and coach proxies).

## Results

Our models predict shifts in medal distribution for 2028, with host effects boosting the USA. Full results in the paper; host effect: +14 medals (95% CI ±4.2-23.9). 56 nations are forecasted to win their first medal.

## Contributing

Pull requests welcome. For issues, open a GitHub issue.

## Citation

If using this code, cite:
@article{olympic_medal_prediction_2025,
title={Tier-Specific Models for Olympic Medal Prediction},
author={Zhixuan Huo},
year={2025},
journal={Your Journal}
}
## License

MIT License (see `LICENSE` file).
This README is self-contained, encourages reproducibility, and directly references the flowchart's elements. Customize sections like citation or usage commands based on your actual implementation. If you add the recommended files, update the README accordingly.
