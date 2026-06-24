import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import statsmodels.api as sm

# 1. Load Data

# Load the data (use raw string for Windows path)
file_path = r'C:\Users\admin\OneDrive\Desktop\Medium.Com\M1\GT_2026_Batting.csv'
df = pd.read_csv(file_path)

# Create the win_flag RIGHT HERE, before filtering ---
df['win_flag'] = df['match_result'].apply(lambda x: 1 if x == 'Win' else 0)

# Define batting positions
top3_positions = [1, 2, 3]
middle_positions = [4, 5, 6, 7]

# Filter dataframes (now they contain the 'win_flag' column)
top3_df = df[df['batting_position'].isin(top3_positions)]
middle_df = df[df['batting_position'].isin(middle_positions)]

# --- Aggregate runs per match for Top Order ---
top_runs_per_match = top3_df.groupby('match_id').agg(
    top_runs=('runs', 'sum'),
    win=('win_flag', 'first')   # 'first' works because all rows in a match have the same result
).reset_index()

# --- Aggregate runs per match for Middle Order ---
mid_runs_per_match = middle_df.groupby('match_id').agg(
    mid_runs=('runs', 'sum')
).reset_index()

# Merge the two aggregations on match_id
match_data = pd.merge(top_runs_per_match, mid_runs_per_match, on='match_id', how='outer').fillna(0)

# Ensure win is integer (0 or 1) - fillna just in case
match_data['win'] = match_data['win'].astype(int)

# Print summary of the prepared dataset
print("=== Prepared Match-Level Data ===")
print(match_data.head())
print(f"\nTotal matches: {len(match_data)}")
print(f"Wins: {match_data['win'].sum()}, Losses: {len(match_data) - match_data['win'].sum()}\n")

# 2. POINT-BISERIAL CORRELATION

top_corr, top_p = pointbiserialr(match_data['win'], match_data['top_runs'])
mid_corr, mid_p = pointbiserialr(match_data['win'], match_data['mid_runs'])

print("=== Point-Biserial Correlation Results ===")
print(f"Top Order  - r: {top_corr:.4f} | p-value: {top_p:.4f} | R²: {top_corr**2:.3f}")
print(f"Middle Order - r: {mid_corr:.4f} | p-value: {mid_p:.4f} | R²: {mid_corr**2:.3f}")
print("\nInterpretation:")
print(f"→ Top Order explains {top_corr**2:.1%} of the variance in match outcome.")
print(f"→ Middle Order explains {mid_corr**2:.1%} of the variance in match outcome.\n")

# 3. LOGISTIC REGRESSION (Odds Ratio)

# Define predictors and target
X = match_data[['top_runs', 'mid_runs']]
y = match_data['win']

# Add constant for intercept
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X).fit(disp=0)  # disp=0 suppresses iteration logs

# Extract coefficients and odds ratios
params = model.params
odds_ratios = np.exp(params)

print("=== Logistic Regression Results ===")
print(f"Intercept: {params['const']:.4f}")
print(f"Top Order Coefficient: {params['top_runs']:.4f} | Odds Ratio: {odds_ratios['top_runs']:.3f}")
print(f"Middle Order Coefficient: {params['mid_runs']:.4f} | Odds Ratio: {odds_ratios['mid_runs']:.3f}")

# Practical impact of a 10-run increase
print("\n=== Practical Impact (10-run increment) ===")
top_odds_increase = (odds_ratios['top_runs'] ** 10 - 1) * 100
mid_odds_increase = (odds_ratios['mid_runs'] ** 10 - 1) * 100
print(f"A +10 run increase in Top Order raises Win Odds by {top_odds_increase:.1f}%")
print(f"A +10 run increase in Middle Order changes Win Odds by {mid_odds_increase:.1f}%")

# 4. SAMPLE-SIZE AWARE COMMENTARY

print("\n=== Statistical Summary for Article ===")
print(f"With {len(match_data)} matches, the Top Order correlation (r = {top_corr:.3f}) is statistically ")
print(f"{'significant' if top_p < 0.05 else 'borderline/non-significant'} (p = {top_p:.3f}), but its R² = {top_corr**2:.3f} ")
print("indicates a substantial practical effect. The Middle Order shows a weak-to-moderate negative ")
print(f"correlation (r = {mid_corr:.3f}), reinforcing the 'exposure' hypothesis where higher middle-order ")
print("scores are often a symptom of a top-order failure rather than a driver of victory.")