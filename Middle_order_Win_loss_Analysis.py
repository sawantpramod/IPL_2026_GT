import pandas as pd
import numpy as np
from scipy import stats

# Load the data (use raw string for Windows path)
file_path = r'C:\Users\admin\OneDrive\Desktop\Medium.Com\M1\GT_2026_Batting.csv'
df = pd.read_csv(file_path)

# Define middle-order positions (4, 5, 6, 7)
middle_positions = [4, 5, 6, 7]

# Filter only middle-order batsmen
middle_df = df[df['batting_position'].isin(middle_positions)]

# Group by match_id and match_result, sum the runs
match_middle_runs = middle_df.groupby(['match_id', 'match_result'], as_index=False)['runs'].sum()

# Separate into wins and losses
wins = match_middle_runs[match_middle_runs['match_result'] == 'Win']['runs']
losses = match_middle_runs[match_middle_runs['match_result'] == 'Loss']['runs']

# Descriptive statistics
print("=== Middle‑order runs per match ===")
print(f"Wins:   n = {len(wins)}, mean = {wins.mean():.2f}, std = {wins.std():.2f}")
print(f"Losses: n = {len(losses)}, mean = {losses.mean():.2f}, std = {losses.std():.2f}\n")

# Welch's t-test (unequal variance)
t_stat, p_value = stats.ttest_ind(wins, losses, equal_var=False)
print(f"Welch's t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("\n→ Reject H₀: There is a statistically significant difference in middle‑order runs between wins and losses.")
else:
    print("\n→ Fail to reject H₀: No statistically significant difference in middle‑order runs between wins and losses.")

# Cohen's d (using pooled standard deviation with unequal sample sizes)
# For unequal variances, we can use a variant that weights by sample size.
# We'll compute the pooled standard deviation (weighted by n-1) and then d = (mean1 - mean2) / pooled_sd.
n1, n2 = len(wins), len(losses)
var1, var2 = wins.var(), losses.var()
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohen_d = (wins.mean() - losses.mean()) / pooled_sd
print(f"Cohen's d = {cohen_d:.4f}")

# Effect size interpretation
abs_d = abs(cohen_d)

if abs_d < 0.2:
    interpretation = "Negligible effect"
    cricket_meaning = "Virtually no dependency on the Middle Order runs between wins and losses."
elif abs_d < 0.5:
    interpretation = "Small effect"
    cricket_meaning = "Mild dependency on the Middle Order runs between wins and losses."
elif abs_d < 0.8:
    interpretation = "Medium effect"
    cricket_meaning = "Moderate dependency on the Middle Order runs between wins and losses."
elif abs_d < 1.2:
    interpretation = "Large effect"
    cricket_meaning = "Strong dependency on the Middle Order runs between wins and losses."
else:
    interpretation = "Very Large effect"
    cricket_meaning = "Extreme dependency on the Middle Order runs between wins and losses."

print("\nEffect Size Interpretation:")
print(f"Category: {interpretation}")
print(f"Cricket Interpretation: {cricket_meaning}")