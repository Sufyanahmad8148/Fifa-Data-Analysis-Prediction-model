"""
FIFA 24 - IMPROVED DATA PREPROCESSING
This file calculates realistic potential based on individual attributes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FIFA 24 - IMPROVED DATA PREPROCESSING")
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

# Add missing columns with realistic values
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

print("\n🎯 STEP 2: Calculating REALISTIC Potential Based on Individual Attributes...")

def calculate_realistic_potential(row):
    """
    Calculate potential based on:
    1. Age and growth curve
    2. Individual attribute strengths
    3. Weak foot and skill moves impact
    4. Physical attributes
    """
    current_overall = row['overall']
    age = row['age']
    
    # Base growth potential by age
    if age < 21:
        age_growth_factor = 8  # Young players can grow significantly
    elif age < 24:
        age_growth_factor = 5
    elif age < 27:
        age_growth_factor = 3
    elif age < 30:
        age_growth_factor = 1
    else:
        age_growth_factor = -1  # Decline phase
    
    # Calculate attribute-based potential modifier
    # Players with high technical skills relative to their overall should have higher potential
    technical_attributes = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    attribute_values = []
    
    for attr in technical_attributes:
        if pd.notna(row.get(attr)):
            attribute_values.append(row[attr])
    
    if len(attribute_values) > 0:
        avg_attributes = np.mean(attribute_values)
        max_attribute = np.max(attribute_values)
        
        # If attributes are higher than overall, player has hidden potential
        attribute_bonus = (avg_attributes - current_overall) * 0.3
        peak_attribute_bonus = (max_attribute - current_overall) * 0.2
    else:
        attribute_bonus = 0
        peak_attribute_bonus = 0
    
    # Weak foot and skill moves bonus (young players with high skills have more potential)
    if age < 25:
        weak_foot_bonus = (row.get('weak_foot', 3) - 3) * 0.5
        skill_moves_bonus = (row.get('skill_moves', 3) - 3) * 0.8
    else:
        weak_foot_bonus = 0
        skill_moves_bonus = 0
    
    # Calculate total potential growth
    total_growth = (
        age_growth_factor +
        attribute_bonus +
        peak_attribute_bonus +
        weak_foot_bonus +
        skill_moves_bonus
    )
    
    # Apply growth with diminishing returns as you approach 99
    proximity_to_max = (99 - current_overall) / 99
    adjusted_growth = total_growth * proximity_to_max
    
    # Calculate final potential
    potential = current_overall + adjusted_growth
    
    # Apply realistic constraints
    potential = max(current_overall, potential)  # Can't be lower than current
    potential = min(99, potential)  # Can't exceed 99
    
    # For older players, potential might decrease
    if age >= 30:
        potential = min(potential, current_overall + 1)
    if age >= 34:
        potential = current_overall - (age - 33) * 0.5
        potential = max(potential, current_overall - 3)
    
    return round(potential)

print("⚙️ Calculating potential for each player based on their attributes...")
df['potential'] = df.apply(calculate_realistic_potential, axis=1)
print("✅ Realistic potential calculated!")

# Show some examples
print("\n📊 Sample of Potential Calculations:")
sample = df[['short_name', 'age', 'overall', 'potential', 'pace', 'shooting', 'passing', 
             'dribbling', 'weak_foot', 'skill_moves']].head(10)
print(sample.to_string())

# Calculate growth
df['potential_growth'] = df['potential'] - df['overall']
print(f"\n📈 Average potential growth: {df['potential_growth'].mean():.2f} points")
print(f"📈 Max potential growth: {df['potential_growth'].max():.0f} points")
print(f"📉 Players with negative growth: {(df['potential_growth'] < 0).sum()}")

print("\n🎯 STEP 3: Selecting Enhanced Features...")
feature_columns = [
    'overall', 'age', 'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physic', 'height_cm', 'weight_kg', 'weak_foot',
    'skill_moves', 'international_reputation', 'value_eur', 'wage_eur'
]

# Add derived features for better predictions
print("🔧 Creating derived features...")
df['pace_shooting_avg'] = (df['pace'] + df['shooting']) / 2
df['passing_dribbling_avg'] = (df['passing'] + df['dribbling']) / 2
df['technical_score'] = (df['pace'] + df['shooting'] + df['passing'] + df['dribbling']) / 4
df['physical_score'] = (df['defending'] + df['physic']) / 2
df['age_squared'] = df['age'] ** 2  # Capture non-linear age effects
df['skill_weak_product'] = df['skill_moves'] * df['weak_foot']

# Add these to feature columns
enhanced_features = feature_columns + [
    'pace_shooting_avg', 'passing_dribbling_avg', 'technical_score',
    'physical_score', 'age_squared', 'skill_weak_product'
]

feature_columns = [col for col in enhanced_features if col in df.columns]
print(f"✅ Selected {len(feature_columns)} features (including derived features)")

target = 'potential'
print(f"🎯 Target variable: {target}")

X = df[feature_columns].copy()
y = df[target].copy()

print("\n🔧 STEP 4: Handling Missing Values...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f"  Filled {col}: median ({median_value:.2f})")

mask = y.notnull()
X = X[mask]
y = y[mask]
print(f"✅ Final dataset shape: {X.shape}")

print("\n✂️ STEP 5: Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training set: {X_train.shape[0]:,} players ({(len(X_train)/len(X)*100):.1f}%)")
print(f"✅ Test set: {X_test.shape[0]:,} players ({(len(X_test)/len(X)*100):.1f}%)")

print("\n⚖️ STEP 6: Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

print("\n💾 STEP 7: Saving Preprocessed Data...")
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

# Save the dataframe with calculated potentials for reference
df.to_csv('players_with_calculated_potential.csv', index=False)
print("✅ Saved: players_with_calculated_potential.csv")

print("\n✅ PREPROCESSING COMPLETE!")
print("📌 Next Step: Run '3_modeltraining.py' to train the machine learning model")
print("="*60)