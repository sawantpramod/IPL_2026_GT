import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import statsmodels.api as sm

# ==========================================
# FUNCTION TO PROCESS A SINGLE SEASON
# ==========================================

def analyze_season(file_path, season_name):
    """
    Loads batting CSV, computes per-match Top3 and Middle-order runs,
    and runs correlation + logistic regression.
    """
    df = pd.read_csv(file_path)
    
    # Create binary win flag
    df['win_flag'] = df['match_result'].apply(lambda x: 1 if x == 'Win' else 0)
    
    # Define positions
    top3_positions = [1, 2, 3]
    middle_positions = [4, 5, 6, 7]
    
    # Filter
    top3_df = df[df['batting_position'].isin(top3_positions)]
    middle_df = df[df['batting_position'].isin(middle_positions)]
    
    # Aggregate runs per match
    top_runs = top3_df.groupby('match_id').agg(
        top_runs=('runs', 'sum'),
        win=('win_flag', 'first')
    ).reset_index()
    
    mid_runs = middle_df.groupby('match_id').agg(
        mid_runs=('runs', 'sum')
    ).reset_index()
    
    # Merge
    match_data = pd.merge(top_runs, mid_runs, on='match_id', how='outer').fillna(0)
    match_data['win'] = match_data['win'].astype(int)
    
    # --- 1. Descriptive Stats (Wins vs Losses) ---
    wins_data = match_data[match_data['win'] == 1]
    losses_data = match_data[match_data['win'] == 0]
    
    desc_stats = {
        'Season': season_name,
        'Total Matches': len(match_data),
        'Wins': len(wins_data),
        'Losses': len(losses_data),
        'Top_Mean_Wins': wins_data['top_runs'].mean(),
        'Top_Mean_Losses': losses_data['top_runs'].mean(),
        'Mid_Mean_Wins': wins_data['mid_runs'].mean(),
        'Mid_Mean_Losses': losses_data['mid_runs'].mean(),
    }
    
    # --- 2. Correlation ---
    top_corr, top_p = pointbiserialr(match_data['win'], match_data['top_runs'])
    mid_corr, mid_p = pointbiserialr(match_data['win'], match_data['mid_runs'])
    
    # --- 3. Logistic Regression ---
    X = match_data[['top_runs', 'mid_runs']]
    y = match_data['win']
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    
    params = model.params
    odds_ratios = np.exp(params)
    
    # Store results
    results = {
        'desc': desc_stats,
        'top_r': top_corr,
        'top_p': top_p,
        'top_r2': top_corr**2,
        'mid_r': mid_corr,
        'mid_p': mid_p,
        'mid_r2': mid_corr**2,
        'top_odds': odds_ratios['top_runs'],
        'mid_odds': odds_ratios['mid_runs'],
        'top_odds_10': (odds_ratios['top_runs'] ** 10 - 1) * 100,
        'mid_odds_10': (odds_ratios['mid_runs'] ** 10 - 1) * 100,
    }
    
    return results

# ==========================================
# RUN FOR BOTH SEASONS
# ==========================================

# Adjust these file paths to match your folder structure
base_path = r'C:\Users\admin\OneDrive\Desktop\Medium.Com\M1'
results_2026 = analyze_season(f'{base_path}\\GT_2026_Batting.csv', 'IPL 2026')
results_2022 = analyze_season(f'{base_path}\\GT_2022_Batting.csv', 'IPL 2022')

# ==========================================
# PRINT COMPARATIVE OUTPUT TABLE
# ==========================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS: GUJARAT TITANS 2022 vs 2026")
print("="*80)

# Descriptive Stats
print("\n--- 1. DESCRIPTIVE STATISTICS (Mean Runs per Match) ---")
print(f"{'Metric':<30} {'IPL 2022':<20} {'IPL 2026':<20}")
print("-"*70)
print(f"{'Top 3 Runs (Wins)':<30} {results_2022['desc']['Top_Mean_Wins']:<20.1f} {results_2026['desc']['Top_Mean_Wins']:<20.1f}")
print(f"{'Top 3 Runs (Losses)':<30} {results_2022['desc']['Top_Mean_Losses']:<20.1f} {results_2026['desc']['Top_Mean_Losses']:<20.1f}")
print(f"{'Middle Order Runs (Wins)':<30} {results_2022['desc']['Mid_Mean_Wins']:<20.1f} {results_2026['desc']['Mid_Mean_Wins']:<20.1f}")
print(f"{'Middle Order Runs (Losses)':<30} {results_2022['desc']['Mid_Mean_Losses']:<20.1f} {results_2026['desc']['Mid_Mean_Losses']:<20.1f}")

# Correlation & R²
print("\n--- 2. POINT-BISERIAL CORRELATION ---")
print(f"{'Metric':<30} {'IPL 2022':<20} {'IPL 2026':<20}")
print("-"*70)
print(f"{'Top 3 - r (p-value)':<30} {results_2022['top_r']:<20.3f} ({results_2022['top_p']:.3f}) {results_2026['top_r']:<20.3f} ({results_2026['top_p']:.3f})")
print(f"{'Top 3 - R² (Variance)':<30} {results_2022['top_r2']:<20.3f} {results_2026['top_r2']:<20.3f}")
print(f"{'Middle - r (p-value)':<30} {results_2022['mid_r']:<20.3f} ({results_2022['mid_p']:.3f}) {results_2026['mid_r']:<20.3f} ({results_2026['mid_p']:.3f})")
print(f"{'Middle - R² (Variance)':<30} {results_2022['mid_r2']:<20.3f} {results_2026['mid_r2']:<20.3f}")

# Logistic Regression (Win Odds for +10 runs)
print("\n--- 3. LOGISTIC REGRESSION (Impact of +10 Runs) ---")
print(f"{'Metric':<30} {'IPL 2022':<20} {'IPL 2026':<20}")
print("-"*70)
print(f"{'Top 3 - Odds Increase (+10)':<30} {results_2022['top_odds_10']:<20.1f}% {results_2026['top_odds_10']:<20.1f}%")
print(f"{'Middle Order - Odds Increase (+10)':<30} {results_2022['mid_odds_10']:<20.1f}% {results_2026['mid_odds_10']:<20.1f}%")

print("\n" + "="*80)