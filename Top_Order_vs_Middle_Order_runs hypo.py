import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load data with raw string path
file_path = r'C:\Users\admin\OneDrive\Desktop\Medium.Com\M1\GT_2026_Batting.csv'
df = pd.read_csv(file_path)

# Define order positions
top_positions = [1, 2, 3]
middle_positions = [4, 5, 6, 7]

# Filter only positions 1–6
df_filtered = df[df['batting_position'].isin(top_positions + middle_positions)].copy()

# Add order_group
df_filtered['order_group'] = df_filtered['batting_position'].apply(
    lambda x: 'Top 3' if x in top_positions else 'Middle'
)

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='order_group', y='runs', data=df_filtered, palette='Set2')
plt.title('Runs Scored: Top 3 Order vs Middle Order (GT 2026)')
plt.xlabel('Batting Order Group')
plt.ylabel('Runs')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Summary stats
print(df_filtered.groupby('order_group')['runs'].describe())

# Optional t-test (Welch's)
top3 = df_filtered[df_filtered['order_group'] == 'Top 3']['runs']
middle = df_filtered[df_filtered['order_group'] == 'Middle']['runs']
t_stat, p_value = stats.ttest_ind(top3, middle, equal_var=False)
print(f"\nWelch's t-test: t = {t_stat:.4f}, p = {p_value:.6f}")

# decision rule
alpha = 0.05

if p_value < alpha:
    print("Reject H0")
    print("Significant difference exists")
else:
    print("Fail to Reject H0")
    print("No significant difference exists")


n1 = len(top3)
n2 = len(middle)

s1 = np.var(top3, ddof=1)
s2 = np.var(middle, ddof=1)

pooled_std =  np.sqrt(
            ((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)
)

cohens_d = (np.mean(top3) - np.mean(middle)) / pooled_std

print("cohen's d = ", round(cohens_d,3))

# Effect size interpretation
if abs(cohens_d) < 0.2:
    size = "negligible"
elif abs(cohens_d) < 0.5:
    size = "small"
elif abs(cohens_d) < 0.8:
    size = "medium"
else:
    size = "large"
print(f"Effect size (Cohen's d) is {size} (|d| = {abs(cohens_d):.4f})")
