"""
FIFA 24 - CHECK DATASET COLUMNS
Run this first to see what columns are in your CSV file
"""

import pandas as pd

print("="*60)
print("CHECKING FIFA 24 DATASET COLUMNS")
print("="*60)

try:
    # Load the dataset
    df = pd.read_csv('male_players.csv')
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    
    print("="*60)
    print("ALL COLUMN NAMES IN THE DATASET:")
    print("="*60)
    
    # Print all columns with their index
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    
    print("\n" + "="*60)
    print("FIRST 5 ROWS OF DATA:")
    print("="*60)
    print(df.head())
    
    print("\n" + "="*60)
    print("DATA TYPES:")
    print("="*60)
    print(df.dtypes)
    
    print("\n" + "="*60)
    print("LOOKING FOR RATING COLUMNS:")
    print("="*60)
    
    # Search for columns that might contain ratings
    rating_keywords = ['overall', 'rating', 'potential', 'ovr']
    found_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in rating_keywords:
            if keyword in col_lower:
                found_columns.append(col)
                break
    
    if found_columns:
        print("Found these rating-related columns:")
        for col in found_columns:
            print(f"  ✓ {col}")
            if not df[col].empty:
                print(f"    Sample values: {df[col].head(3).tolist()}")
    else:
        print("❌ No obvious rating columns found")
        print("\nPlease check the column names above and identify which columns contain:")
        print("  1. Player's current overall rating")
        print("  2. Player's potential rating")
        print("  3. Player's age")
        print("  4. Player's name")
    
    # Save column names to a file for reference
    with open('column_names.txt', 'w', encoding='utf-8') as f:
        f.write("FIFA 24 Dataset Column Names\n")
        f.write("="*60 + "\n\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:3d}. {col}\n")
    
    print("\n✅ Column names saved to: column_names.txt")
    
except FileNotFoundError:
    print("❌ ERROR: 'male_players.csv' not found!")
    print("\nPlease make sure:")
    print("  1. You downloaded the CSV file from Kaggle")
    print("  2. The file is named 'male_players.csv'")
    print("  3. The file is in the same folder as this script")
    print("\nKaggle link: https://www.kaggle.com/code/ugureker/fc-24-data-analysis/input")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("\nPlease check:")
    print("  1. The CSV file is not corrupted")
    print("  2. The file has proper column headers")
    print("  3. You have pandas installed: pip install pandas")

print("\n" + "="*60)