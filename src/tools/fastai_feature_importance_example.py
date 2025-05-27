"""
Small example demonstrating the correct approach to calculating feature importance
with fastai's TabularLearner using Random Forest as a proxy.
"""
from fastai.tabular.all import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import os

# Create a simple dataset for demonstration
print("Creating example dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Define categorical and continuous variables
# For this example, we'll treat some continuous variables as categorical
cat_vars = ['HouseAge', 'AveRooms']
cont_vars = ['MedInc', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Convert categorical variables to strings to make fastai treat them as categorical
for cat in cat_vars:
    df[cat] = df[cat].astype(str)

print(f"Dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Using {len(cat_vars)} categorical variables and {len(cont_vars)} continuous variables")

# Create TabularPandas object
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize],
                  cat_names=cat_vars, cont_names=cont_vars,
                  y_names='target', splits=splits)

# Create DataLoaders - explicitly set device to CPU
dls = to.dataloaders(bs=64, device='cpu')

# Create and train TabularLearner
print("Building and training TabularLearner model...")
learn = tabular_learner(dls, metrics=rmse, layers=[200, 100])
learn.fit_one_cycle(5, 1e-3)

# Show model results
print("\nModel evaluation:")
learn.show_results()

# Calculate feature importance using RandomForest
print("\nCalculating feature importance using Random Forest...")

# Create a copy of the dataframe
df_rf = df.copy()

# Convert categorical variables to one-hot encoding for RandomForest
df_rf = pd.get_dummies(df_rf, columns=cat_vars, drop_first=False)

# Select features and target
X = df_rf.drop(columns=['target'])
y = df_rf['target']

# Train RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_names = X.columns
importances = {name: imp for name, imp in zip(feature_names, rf.feature_importances_)}

# Sort by importance
importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

# Print top features
print("\nTop 10 most important features from Random Forest:")
for i, (feature, imp) in enumerate(list(importances.items())[:10]):
    print(f"{i+1}. {feature}: {imp:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = list(importances.keys())[:10]
top_importances = [importances[feature] for feature in top_features]
plt.barh(range(len(top_features)), top_importances, align='center')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()

# Before saving figures, ensure the output directory exists
output_dir = os.path.join('output_graphs', 'fastai_feature_importance_example')
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
print("Feature importance plot saved to 'feature_importance.png'")

# Alternative approach: Using SHAP values (if shap is installed)
try:
    import shap
    print("\nCalculating feature importance using SHAP values...")
    
    # Sample rows for background data
    background_data = X.sample(min(100, len(X)))
    
    # Create explainer
    explainer = shap.Explainer(rf, background_data)
    
    # Calculate SHAP values on a sample of data
    sample_data = X.sample(min(200, len(X)))
    shap_values = explainer(sample_data)
    
    # Get global feature importance
    shap_importances = {name: np.abs(shap_values.values[:, i]).mean() 
                        for i, name in enumerate(feature_names)}
    
    # Sort by importance
    shap_importances = dict(sorted(shap_importances.items(), key=lambda x: x[1], reverse=True))
    
    # Print top features
    print("\nTop 10 most important features from SHAP values:")
    for i, (feature, imp) in enumerate(list(shap_importances.items())[:10]):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
    print("SHAP importance plot saved to 'shap_importance.png'")
    
except ImportError:
    print("\nSHAP library not installed. Skipping SHAP analysis.")

print("\nExample completed successfully!") 