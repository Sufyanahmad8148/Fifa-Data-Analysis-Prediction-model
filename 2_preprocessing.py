"""
FIFA 24 - DATA PREPROCESSING
This file prepares the data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FIFA 24 - DATA PREPROCESSING")
print("="*60)

# Load the dataset
print("\n📂 STEP 1: Loading Dataset...")
try:
    df = pd.read_csv('male_players.csv')
    print(f"✅ Dataset loaded: {df.shape[0]:,} players, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: male_players.csv not found!")
    exit()


column_mapping = {
    'Name': 'short_name',
    'Position': 'positions',
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
    'Skill moves': 'skill_moves'
}
df = df.rename(columns=column_mapping)

if 'potential' not in df.columns:
    df['potential'] = df.apply(
        lambda row: min(row['overall'] + max(0, (28 - row['age'])) * 0.5, 99) if row['age'] < 28 
        else row['overall'], axis=1
    ).astype(int)

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

print("\n🎯 STEP 2: Selecting Features...")
feature_columns = [
    'overall', 'age', 'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physic', 'height_cm', 'weight_kg', 'weak_foot',
    'skill_moves', 'international_reputation', 'value_eur', 'wage_eur'
]

feature_columns = [col for col in feature_columns if col in df.columns]
print(f"✅ Selected {len(feature_columns)} features")

target = 'potential'
print(f"🎯 Target variable: {target}")

X = df[feature_columns].copy()
y = df[target].copy()

print("\n🔧 STEP 3: Handling Missing Values...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f" Filled {col}: median ({median_value:.2f})")

mask = y.notnull()
X = X[mask]
y = y[mask]
print(f"✅ Final dataset shape: {X.shape}")

print("\n✂️ STEP 4: Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training set: {X_train.shape[0]:,} players ({(len(X_train)/len(X)*100):.1f}%)")
print(f"✅ Test set: {X_test.shape[0]:,} players ({(len(X_test)/len(X)*100):.1f}%)")

print("\n⚖️ STEP 5: Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

print("\n💾 STEP 6: Saving Preprocessed Data...")
with open('fifa_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: fifa_scaler.pkl")

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✅ Saved: feature_columns.pkl")

np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("✅ Saved: Training and test data")

print("\n✅ PREPROCESSING COMPLETE!")
print("📌 Next Step: Run '3_modeltraining.py' to train the machine learning model")
print("="*60)