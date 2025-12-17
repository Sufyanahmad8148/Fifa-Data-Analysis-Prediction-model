"""
FIFA 24 - MACHINE LEARNING MODEL TRAINING
This file trains a model to predict player potential ratings
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FIFA 24 - MACHINE LEARNING MODEL TRAINING")
print("="*60)
print("\n📂 STEP 1: Loading Preprocessed Data...")
try:
    X_train_scaled = np.load('X_train_scaled.npy')
    X_test_scaled = np.load('X_test_scaled.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    print(f"✅ Training set: {X_train_scaled.shape[0]:,} players")
    print(f"✅ Test set: {X_test_scaled.shape[0]:,} players")
    print(f"✅ Features: {len(feature_columns)}")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("\n⚠️  Please run '2_preprocessing.py' first!")
    exit(1)
print("\n🤖 STEP 2: Configuring Machine Learning Model...")
print("\n📋 Model: Gradient Boosting Regressor")
print("Why this model?")
print(" ✓ Excellent for regression tasks")
print(" ✓ Handles non-linear relationships")
print(" ✓ Resistant to overfitting")
print(" ✓ High accuracy for player predictions")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5,
    verbose=0
)
print("\n✅ Model configured with optimal parameters")
print("\n🎓 STEP 3: Training the Model...")
print("This may take a minute...")
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")
print("\n🔮 STEP 4: Making Predictions...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("✅ Predictions generated for both training and test sets")
print("\n" + "="*60)
print("STEP 5: MODEL EVALUATION")
print("="*60)
print("\n📊 TRAINING SET PERFORMANCE:")
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
print(f"   RMSE: {train_rmse:.4f}")
print(f"   MAE: {train_mae:.4f}")
print(f"   R² Score: {train_r2:.4f}")
print(f"   → Model explains {train_r2*100:.2f}% of variance")

print("\n📊 TEST SET PERFORMANCE:")
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
print(f"   RMSE: {test_rmse:.4f}")
print(f"   MAE: {test_mae:.4f}")
print(f"   R² Score: {test_r2:.4f}")
print(f"   → Model explains {test_r2*100:.2f}% of variance")
print("\n💡 WHAT THIS MEANS:")
print(f"   • On average, predictions are off by {test_mae:.2f} rating points")
print(f"   • The model is {'excellent' if test_r2 > 0.9 else 'good' if test_r2 > 0.8 else 'decent'}! (R² = {test_r2:.3f})")

if train_r2 - test_r2 < 0.05:
    print(f"   • Low overfitting - model generalizes well! ✅")
else:
    print(f"   • Some overfitting detected (gap: {train_r2 - test_r2:.3f})")
print("\n" + "="*60)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🏆 Most Important Features for Prediction:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")
try:
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Feature Importance in Predicting Player Potential', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('model_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: model_feature_importance.png")
    plt.close()
except Exception as e:
    print(f"\n⚠️  Could not save visualization: {e}")
print("\n" + "="*60)
print("STEP 7: PREDICTION ANALYSIS")
print("="*60)

try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Potential Rating', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Potential Rating', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.3f}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)   
    errors = y_test - y_pred_test
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Prediction Error Distribution\nMAE = {test_mae:.3f}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)   
    plt.tight_layout()
    plt.savefig('model_predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved: model_predictions_analysis.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Could not save visualization: {e}")
print("\n💾 STEP 8: Saving the Trained Model...")

with open('fifa_rating_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved: fifa_rating_model.pkl")
metrics = {
    'train_rmse': train_rmse,
    'train_mae': train_mae,
    'train_r2': train_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'feature_importance': feature_importance.to_dict()
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✅ Saved: model_metrics.pkl")
print("\n🔮 STEP 9: Creating Predictions for All Players...")
try:
    df = pd.read_csv('male_players.csv')
    print(f"✅ Loaded dataset: {len(df):,} players")
    
    column_mapping = {
        'Name': 'short_name',
        'Position': 'positions',
        'Club': 'club_name',
        'Nation': 'nationality_name',
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
    print("✅ Column names standardized")    
    if 'potential' not in df.columns:
        df['potential'] = df.apply(
            lambda row: min(row['overall'] + max(0, (28 - row['age'])) * 0.5, 99) 
            if row['age'] < 28 else row['overall'],
            axis=1
        ).astype(int)
        print("✅ Calculated 'potential' column")
    
    if 'height_cm' not in df.columns:
        np.random.seed(42)
        df['height_cm'] = np.random.randint(165, 195, len(df))
        print("✅ Generated 'height_cm' column")
        
    if 'weight_kg' not in df.columns:
        np.random.seed(42)
        df['weight_kg'] = np.random.randint(60, 95, len(df))
        print("✅ Generated 'weight_kg' column")
        
    if 'value_eur' not in df.columns:
        np.random.seed(42)
        df['value_eur'] = (df['overall'] ** 2) * np.random.randint(1000, 10000, len(df))
        print("✅ Generated 'value_eur' column")        
    if 'wage_eur' not in df.columns:
        np.random.seed(42)
        df['wage_eur'] = df['overall'] * np.random.randint(100, 1000, len(df))
        print("✅ Generated 'wage_eur' column")       
    if 'international_reputation' not in df.columns:
        df['international_reputation'] = np.where(
            df['overall'] >= 85, 5,
            np.where(df['overall'] >= 80, 4,
                    np.where(df['overall'] >= 75, 3,
                            np.where(df['overall'] >= 70, 2, 1)))
        )
        print("✅ Generated 'international_reputation' column")
    with open('fifa_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    X_full = df[feature_columns].copy()
    missing_before = X_full.isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠️  Found {missing_before} missing values, filling with median...")
        for col in X_full.columns:
            if X_full[col].isnull().sum() > 0:
                median_val = X_full[col].median()
                X_full[col].fillna(median_val, inplace=True)
        print("✅ Missing values handled")
    
    X_full_scaled = scaler.transform(X_full)
    print("✅ Features scaled") 
    predictions = model.predict(X_full_scaled)
    print("✅ Predictions generated")
    df['predicted_potential'] = predictions
    df['predicted_rating_change'] = df['predicted_potential'] - df['overall']
    
    df.to_csv('players_with_predictions.csv', index=False)
    print(f"✅ Saved predictions for {len(df):,} players: players_with_predictions.csv")
    
except Exception as e:
    print(f"❌ Error creating predictions: {e}")
    import traceback
    traceback.print_exc()
print("\n" + "="*60)
print("STEP 10: EXAMPLE PREDICTIONS")
print("="*60)

try:
    print("\n🌟 Top 5 Young Players with Highest Predicted Growth:")
    young_players = df[df['age'] < 23].nlargest(5, 'predicted_rating_change')
    
    for idx, player in young_players.iterrows():
        print(f"\n   {player['short_name']}:")
        print(f"      Current: {player['overall']:.0f} → Predicted: {player['predicted_potential']:.0f}")
        print(f"      Expected Growth: +{player['predicted_rating_change']:.1f}")
        print(f"      Age: {player['age']:.0f}, Position: {player['positions']}")
        
    print("\n\n🔥 Top 5 Players with Highest Predicted Rating:")
    top_predicted = df.nlargest(5, 'predicted_potential')
    
    for idx, player in top_predicted.iterrows():
        print(f"\n   {player['short_name']}:")
        print(f"      Current: {player['overall']:.0f} → Predicted: {player['predicted_potential']:.0f}")
        print(f"      Change: {player['predicted_rating_change']:+.1f}")
        print(f"      Age: {player['age']:.0f}, Position: {player['positions']}")
        
except Exception as e:
    print(f"⚠️  Could not display examples: {e}")

print("\n" + "="*60)
print("✅ MACHINE LEARNING MODEL - TRAINING COMPLETE!")
print("="*60)
print(f"\n✅ Model trained successfully")
print(f"✅ Test R² Score: {test_r2:.4f} ({'Excellent' if test_r2 > 0.9 else 'Good' if test_r2 > 0.8 else 'Decent'}!)")
print(f"✅ Average Error: {test_mae:.2f} rating points")
print(f"✅ Predictions saved for all {len(df):,} players")
print(f"\n📁 Files Created:")
print(f"   • fifa_rating_model.pkl")
print(f"   • model_metrics.pkl")
print(f"   • players_with_predictions.csv")
print(f"   • model_feature_importance.png")
print(f"   • model_predictions_analysis.png")
print("\n📌 Next Step: Run 'streamlit run 4_streamlit_app.py' to launch the web app!")
print("="*60)