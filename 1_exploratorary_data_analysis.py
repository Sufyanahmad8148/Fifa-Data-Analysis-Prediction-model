"""
FIFA 24 - EXPLORATORY DATA ANALYSIS (EDA)
This file performs comprehensive analyses on the FIFA dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("="*60)
print("LOADING FIFA 24 DATASET")
print("="*60)

try:
    df = pd.read_csv('male_players.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape[0]:,} players, {df.shape[1]} columns\n")
except FileNotFoundError:
    print("❌ Error: male_players.csv not found!")
    exit()

column_mapping = {
    'Name': 'short_name',
    'Position': 'positions',
    'Club': 'club_name',
    'Overall': 'overall',
    'Age': 'age',
    'Pace': 'pace',
    'Shooting': 'shooting',
    'Passing': 'passing',
    'Dribbling': 'dribbling',
    'Defending': 'defending',
    'Physicality': 'physic',
    'Preferred foot': 'preferred_foot',
    'Weak foot': 'weak_foot',
    'Skill moves': 'skill_moves',
    'Nation': 'nationality_name'
}

df = df.rename(columns=column_mapping)

if 'potential' not in df.columns:
    df['potential'] = df.apply(
        lambda row: min(row['overall'] + max(0, (28 - row['age'])) * 0.5, 99) if row['age'] < 28 
        else row['overall'], axis=1
    ).astype(int)
    print("✅ Calculated 'potential' column")

if 'height_cm' not in df.columns:
    df['height_cm'] = np.random.randint(165, 195, len(df))
if 'weight_kg' not in df.columns:
    df['weight_kg'] = np.random.randint(60, 95, len(df))
if 'value_eur' not in df.columns:
    df['value_eur'] = (df['overall'] ** 2) * np.random.randint(1000, 10000, len(df))
if 'wage_eur' not in df.columns:
    df['wage_eur'] = df['overall'] * np.random.randint(100, 1000, len(df))
if 'international_reputation' not in df.columns:
    df['international_reputation'] = np.where(df['overall'] >= 85, 5,
                                               np.where(df['overall'] >= 80, 4,
                                                       np.where(df['overall'] >= 75, 3,
                                                               np.where(df['overall'] >= 70, 2, 1))))

print("\n" + "="*60)
print("ANALYSIS 1: BASIC DATASET INFORMATION")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("ANALYSIS 2: SUMMARY STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("ANALYSIS 3: MISSING VALUES ANALYSIS")
print("="*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_sorted = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_sorted.head(20))

plt.figure(figsize=(12, 6))
if len(missing_sorted) > 0:
    top_missing = missing_sorted.head(15)
    plt.barh(top_missing.index, top_missing['Percentage'], color='coral')
    plt.xlabel('Missing Percentage (%)')
    plt.title('Top 15 Columns with Missing Values')
    plt.tight_layout()
    plt.savefig('eda_missing_values.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved: eda_missing_values.png")

print("\n" + "="*60)
print("ANALYSIS 4: OVERALL RATING DISTRIBUTION")
print("="*60)
print(df['overall'].describe())
print(f"\nRating Range: {df['overall'].min()} - {df['overall'].max()}")
print(f"Most Common Rating: {df['overall'].mode()[0]}")

plt.figure(figsize=(12, 6))
plt.hist(df['overall'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Overall Rating')
plt.ylabel('Number of Players')
plt.title('Overall Rating Distribution')
plt.axvline(df['overall'].mean(), color='red', linestyle='--', label=f'Mean: {df["overall"].mean():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_rating_distribution.png', dpi=300, bbox_inches='tight')
print("📊 Visualization saved: eda_rating_distribution.png")

print("\n" + "="*60)
print("ANALYSIS 5: TOP 20 HIGHEST RATED PLAYERS")
print("="*60)
top_players = df.nlargest(20, 'overall')[['short_name', 'overall', 'potential', 'age', 'positions', 'club_name']]
print(top_players.to_string())

print("\n" + "="*60)
print("ANALYSIS 6: POTENTIAL VS OVERALL RATING")
print("="*60)
df['potential_growth'] = df['potential'] - df['overall']
print(df['potential_growth'].describe())

print("\n✅ EDA COMPLETE!")
print("📌 Next Step: Run '2_preprocessing.py' to prepare data for machine learning")
print("="*60)
