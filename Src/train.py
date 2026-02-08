import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Data
df = pd.read_excel(r'D:\Full projet\Predicting_chronic_kidney_disease\Data\data_output_after_processing_raw.xlsx')

# 2. Separate Features (X) and Target (y)
X = df.drop('class', axis=1)
y = df['class']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify=y)

# 4. Define Preprocessing Pipeline
# Identify column types automatically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline for Numeric Data: Impute Median -> Scale 
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) ])

# Pipeline for Categorical Data: Impute 'missing' -> OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) ])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features) ])

# 5. Create the Full Model Pipeline
# This pipeline contains the Preprocessor AND the RandomForest Model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) ])

# 6. Train the Model
print("Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluate
print("\n--- Model Evaluation on Test Set ---")
y_pred = model_pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the Entire Pipeline
output_model_path = 'model_pipeline.pkl'
joblib.dump(model_pipeline, output_model_path)
print(f"\n[SUCCESS] Model pipeline saved to: {output_model_path}")
print("You can now use this file to predict on raw new data directly.")