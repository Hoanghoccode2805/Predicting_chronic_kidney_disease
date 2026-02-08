import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_excel(r'D:\Full projet\Predicting_chronic_kidney_disease\Data\data_input.xlsx')
print(f"Original Data Shape: {df.shape}")        
print(df.columns)      # List of column names
print(df.info())       # Data types and number of non-null data entries
print(df.describe())   # Basic statistics of numeric columns

# 2. DATA CLEANING: Handle hidden characters
# Strip leading/trailing whitespace from all string columns
df = df.apply(lambda x: x.map(str).str.strip() if x.dtype == 'object' else x)

# 3. Handle Missing Values Indicators
# Replace '?' (and its variants) with NumPy's standard NaN
df.replace('?', np.nan, inplace=True)

# 4. Data Type Conversion
# Convert numerical columns from 'object' to 'numeric'
# 'errors=coerce' turns unparseable strings into NaN automatically
cols_num = ['age', 'specific gravity', 'albumin', 'sugar', 'blood pressure', 'blood glucose random', 'blood urea', 'serum creatinine', 
            'sodium', 'potassium', 'hemoglobin', 'packed cell volume', 
            'white blood cell count', 'red blood cell count']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Target Variable Mapping 
# Standardize the target class to binary: 1 for CKD (Positive), 0 for NotCKD (Negative)
if 'class' in df.columns:
    df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

# 6. Data Inspection
print("\n--- Data Info After Cleaning ---")
print(df.info())

print("\n--- Unique Values in Categorical Columns (Check for consistency) ---")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")

# 7. Save Cleaned Data 
df.to_excel('D:\Full projet\Predicting_chronic_kidney_disease\Data\data_output_after_processing_raw.xlsx', index=False)
print("\n[SUCCESS] Cleaned data saved to: data_output_after_processing.xlsx ")