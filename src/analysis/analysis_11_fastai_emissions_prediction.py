"""
FastAI-based Emissions Prediction Model

This script implements a machine learning model using FastAI's tabular data tools to predict
carbon emissions based on company characteristics, targets, and programs.

Input DataFrames:
1. company_emissions_df: Contains emission metrics and company characteristics
2. targets_df: Contains carbon reduction targets and progress
3. programs1_df: Contains GHG mitigation program details
4. programs2_df: Contains carbon program implementation details

Key Features:
- Handles both categorical and continuous features
- Uses FastAI's tabular data processing pipeline
- Implements a neural network for emission prediction
- Includes model training, validation, and evaluation
- Automatic GPU/CPU selection based on availability
- Feature importance analysis using Random Forest

Outputs:
1. Model performance metrics (RMSE, R²)
2. Actual vs. Predicted values plot (saved as 'actual_vs_predicted.png')
3. Feature importance plot (saved as 'feature_importance.png')

"""

# ===== IMPORTS =====
# Standard library imports
import os
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastai.tabular.all import *
from scipy.stats import mstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# Check for GPU availability and set device
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
except Exception as e:
    device = 'cpu'
    print(f"Error checking for GPU, defaulting to CPU. Error: {e}")

warnings.filterwarnings("ignore", message="is_sparse is deprecated")

# ===== DATA LOADING =====
def load_datasets():
    """
    Load and validate all required datasets from the data directory.
    
    Returns:
        tuple: A tuple containing four DataFrames in the following order:
            - company_emission_df: Company emissions data
            - program1_df: Program 1 implementation data
            - program2_df: Program 2 implementation data
            - target_df: Target data
            
    Raises:
        FileNotFoundError: If any of the required data files are missing
        ValueError: If any of the required columns are missing from the data
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data')
    
    company_emission_df = pd.read_csv(os.path.join(data_dir, 'company_emissions_merged.csv'))
    print(f"Loaded company_emission_df with shape: {company_emission_df.shape}")
    
    program1_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsprogramme 1 Results - 20241212 15_45_26.csv'))
    print(f"Loaded program1_df with shape: {program1_df.shape}")
    
    program2_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsprogramme 2 Results - 20241212 15_51_07.csv'))
    print(f"Loaded program2_df with shape: {program2_df.shape}")
    
    target_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsziele Results - 20241212 15_49_29.csv'))
    print(f"Loaded target_df with shape: {target_df.shape}")
    
    return company_emission_df, program1_df, program2_df, target_df

print("Loading datasets...")
company_emission_df, program1_df, program2_df, target_df = load_datasets()
print("\nData loading complete. Starting feature engineering...")

# ===== CONFIGURATION =====
# Define column categories based on data dictionaries for feature engineering
# These columns are grouped by their source and type (categorical/continuous) for processing

# ===== COLUMN DEFINITIONS =====
# 1. COMPANY EMISSIONS COLUMNS
emission_cat_cols = [
    'ISSUER_CNTRY_DOMICILE',
    'NACE_CLASS_CODE',
    'NACE_CLASS_DESCRIPTION',
    'CARBON_EMISSIONS_SCOPE_1_KEY',
    'CARBON_EMISSIONS_SCOPE_12_KEY',
    'CARBON_EMISSIONS_SCOPE_2_KEY'
]

emission_cont_cols = [
    'MarketCap_USD',
    'SALES_USD_RECENT',
    'CARBON_EMISSIONS_SCOPE_1_INTEN',
    'CARBON_EMISSIONS_EVIC_SCOPE_1_INTEN',
    'CARBON_EMISSIONS_SCOPE_2',
    'CARBON_EMISSIONS_SCOPE_2_INTEN',
    'CARBON_EMISSIONS_EVIC_SCOPE_2_INTEN',
    'CARBON_EMISSIONS_SCOPE_12',
    'CARBON_EMISSIONS_SCOPE_12_INTEN',
    'CARBON_EMISSIONS_EVIC_SCOPE_12_INTEN',
    'CARBON_EMISSIONS_SCOPE_3',
    'CARBON_SCOPE_12_INTEN_3Y_AVG',
    'EVIC_EUR',
    'EVIC_USD_RECENT'
]

# 2. TARGETS COLUMNS
target_cat_cols = [
    'CBN_TARGET_AGGR',
    'CBN_TARGET_CATEGORY',
    'CBN_TARGET_SCOPE',
    'CBN_TARGET_STATUS',
    'CBN_TARGET_STATUS_DETAIL',
    'TARGET_CARBON_TYPE',
    'TARGET_CARBON_UNITS'
]

target_cont_cols = [
    'CBN_TARGET_IMP_YEAR',
    'CBN_TARGET_REDUC_PCT',
    'CBN_TARGET_BASE_YEAR',
    'CBN_TARGET_BASE_YEAR_VAL',
    'TARGET_CARBON_PROGRESS_VALUE',
    'TARGET_CARBON_PROGRESS_PCT',
    'TARGET_CARBON_OFFSET_PCT',
    'TARGET_CARBON_OFFSET_VOLUME'
]

# 3. PROGRAMS1 COLUMNS
# These columns contain information about GHG mitigation programs
program1_cat_cols = [
    'CBN_GHG_MITIG_DISTRIBUTION',
    'CBN_GHG_MITIG_RAW_MAT',
    'CBN_GHG_MITIG_MFG',
    'CBN_GHG_MITIG_TRANSPORT',
    'CBN_GHG_MITIG_CAPTURE'
]

# 4. PROGRAMS2 COLUMNS
# These columns contain information about carbon program implementations
program2_cat_cols = [
    'CARBON_PROGRAMS_CATEGORY',
    'CARBON_PROGRAMS_TYPE',
    'CARBON_PROGRAMS_OVERSIGHT',
    'CARBON_PROGRAMS_SOURCE'
]

program2_cont_cols = [
    'CARBON_PROGRAMS_IMP_YEAR'
]

# ===== SPECIAL COLUMNS =====
# These columns contain free-form text that requires special processing
text_cols = [
    'CARBON_PROGRAMS_DESCRIPTION',
    'CARBON_PROGRAMS_COMMENT',
    'CARBON_PROGRAMS_LINK',
    'TARGET_CARBON_TYPE_OTHER',
    'TARGET_CARBON_OFFSET_DESC',
    'TARGET_CARBON_UNITS_OTHER'
]

# ID columns to exclude from modeling
id_cols = [
    'ISSUERID',
    'ISSUER_NAME',
    'ISSUER_TICKER',
    'ISSUER_ISIN'
]

# Function to aggregate categorical columns with one-hot encoding
def aggregate_categorical(df, cat_cols):
    """
    Create one-hot encoded features for categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing categorical columns
        cat_cols (list): List of column names to be one-hot encoded
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns and ISSUERID
    """
    result = df[['ISSUERID']].drop_duplicates()
    
    for col in cat_cols:
        if col in df.columns:
            # Skip columns with too many categories
            if df[col].nunique() > 100:
                continue
                
            # Create one-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col)
            dummy_df = df[['ISSUERID']].join(dummies)
            # Aggregate by max (1 if the category exists for that company)
            agg_df = dummy_df.groupby('ISSUERID').max().reset_index()
            result = result.merge(agg_df, on='ISSUERID', how='left')
    
    return result

def aggregate_continuous(df, cont_cols):
    """
    Create statistical aggregations (mean, std, min, max) for continuous columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing continuous columns
        cont_cols (list): List of column names to be aggregated
        
    Returns:
        pd.DataFrame: DataFrame with aggregated statistics and ISSUERID
    """
    result = df[['ISSUERID']].drop_duplicates()
    
    # Define aggregation functions for continuous variables
    agg_dict = {}
    for col in cont_cols:
        if col in df.columns:
            agg_dict[col] = ['mean', 'min', 'max', 'count']
    
    if agg_dict:
        agg_df = df.groupby('ISSUERID').agg(agg_dict)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()
        result = result.merge(agg_df, on='ISSUERID', how='left')
    
    return result

# Aggregate features from programs2_df
program2_cat_features = aggregate_categorical(program2_df, program2_cat_cols)
program2_cont_features = aggregate_continuous(program2_df, program2_cont_cols)
program2_features = program2_cat_features.merge(program2_cont_features, on='ISSUERID', how='left')

# Aggregate features from target_df
target_cat_features = aggregate_categorical(target_df, target_cat_cols)
target_cont_features = aggregate_continuous(target_df, target_cont_cols)
target_features = target_cat_features.merge(target_cont_features, on='ISSUERID', how='left')

# Merge all datasets
model_data = company_emission_df.copy()
model_data = model_data.merge(program1_df, on='ISSUERID', how='left')
model_data = model_data.merge(program2_features, on='ISSUERID', how='left')
model_data = model_data.merge(target_features, on='ISSUERID', how='left')

# Our target variable
target_col = 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'

# Create final categorical and continuous feature lists
# Start with known columns from the data dictionaries
all_cat_cols = emission_cat_cols + program1_cat_cols
all_cont_cols = emission_cont_cols + program2_cont_cols + target_cont_cols

# Add generated columns from one-hot encoding
for col in model_data.columns:
    if col.startswith(tuple(target_cat_cols)) or col.startswith(tuple(program2_cat_cols)):
        all_cat_cols.append(col)

# Filter to include only columns present in the merged dataset
cat_features = [col for col in all_cat_cols if col in model_data.columns and col not in id_cols]
cont_features = [col for col in all_cont_cols if col in model_data.columns and col != target_col]

# Fill missing values
model_data[cont_features] = model_data[cont_features].fillna(0)
model_data[cat_features] = model_data[cat_features].fillna('#na#')

# Drop rows where target is missing
model_data = model_data.dropna(subset=[target_col])

# Remove features that may cause leakage with target variable
leakage_cols = [
    col for col in model_data.columns 
    if 'CAGR' in col or 'TARGET' in col or 'FUTURE' in col or 'PROJECTION' in col
]
for col in leakage_cols:
    if col != target_col and col in cont_features:
        cont_features.remove(col)

print(f"Final dataset shape: {model_data.shape}")
print(f"Using {len(cat_features)} categorical features and {len(cont_features)} continuous features")

# Feature selection to reduce dimensionality
X = model_data[cont_features]
y = model_data[target_col]
selector = SelectKBest(f_regression, k=min(15, len(cont_features)))
selected_features = selector.fit(X, y)
feature_scores = pd.DataFrame({
    'Feature': cont_features,
    'Score': selector.scores_
})
print("Top continuous features by F-score:")
print(feature_scores.sort_values('Score', ascending=False).head(10))

# Keep only most important continuous features
top_cont_features = feature_scores.sort_values('Score', ascending=False).head(15)['Feature'].tolist()
cont_features = top_cont_features

# Winsorize instead of simple outlier removal
for col in cont_features:
    model_data[col] = mstats.winsorize(model_data[col], limits=[0.01, 0.01])

# Try more complex transformations like Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')
model_data[cont_features] = pt.fit_transform(model_data[cont_features])

# Add interaction features 
model_data['size_emissions_ratio'] = model_data['SALES_USD_RECENT'] / (model_data['CARBON_EMISSIONS_SCOPE_12'] + 1)

# Handle outliers in the target variable
target_outliers = np.abs(model_data[target_col]) > 100
model_data = model_data[~target_outliers]

# Log-transform skewed features
for col in cont_features:
    if model_data[col].skew() > 1:
        model_data[col] = np.log1p(np.abs(model_data[col]))


# Create industry-specific features
model_data['industry_emission_ratio'] = model_data.groupby('NACE_CLASS_CODE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')

# Create regional features
model_data['country_emission_ratio'] = model_data.groupby('ISSUER_CNTRY_DOMICILE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')

# Add industry-specific emission trends
model_data['industry_trend'] = model_data.groupby('NACE_CLASS_CODE')[target_col].transform('mean')

# Add industry-level emission variance (indicates how consistent companies are within an industry)
model_data['industry_emissions_std'] = model_data.groupby('NACE_CLASS_CODE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('std')

import warnings
warnings.filterwarnings("ignore", message="is_sparse is deprecated")

# Split data into training and validation sets (80/20)
# Since reporting_year column doesn't exist, use random split instead
from sklearn.model_selection import train_test_split
train_idx, valid_idx = train_test_split(range(len(model_data)), test_size=0.2, random_state=42)

# Create TabularPandas object
procs = [Categorify, FillMissing, Normalize]
splits = (list(train_idx), list(valid_idx))

to = TabularPandas(model_data, procs=procs, cat_names=cat_features, cont_names=cont_features, 
                   y_names=target_col, splits=splits)

# Device is already
# Create DataLo
batch_size = 128
dls = to.dataloaders(bs=batch_size, device=device)

# Define model
y_min = model_data[target_col].min() * 1.1  # Add 10% buffer
y_max = model_data[target_col].max() * 1.1

# Create config dictionary for model parameters
config = {
    'ps': [0.2, 0.2],  # Dropout probabilities
    'y_range': (y_min, y_max)
}

learn = tabular_learner(dls, layers=[200, 100], metrics=[rmse, mae],
                       wd=0.1,  # Increased from 0.01
                       config=config,
                       cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5)])

# Automatically find suitable learning rate
suggested_lr = learn.lr_find().valley
print(f"Using learning rate: {suggested_lr}")

# Train the model
epochs = 10
learn.fit_one_cycle(epochs, suggested_lr)

# Evaluate the model
valid_preds, valid_targets = learn.get_preds()
valid_preds = valid_preds.squeeze().cpu().numpy()
valid_targets = valid_targets.cpu().numpy()

# Calculate metrics
rmse_val = np.sqrt(mean_squared_error(valid_targets, valid_preds))
r2_val = r2_score(valid_targets, valid_preds)

print(f"Validation RMSE: {rmse_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(valid_targets, valid_preds, alpha=0.5)
plt.plot([-100, 100], [-100, 100], 'r--')
plt.xlabel('Actual Emission Trend (%)')
plt.ylabel('Predicted Emission Trend (%)')
plt.title('Actual vs Predicted 3-Year Emission Trends')
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.grid(True)

# Before saving any figures, ensure the output directory exists
output_dir = 'output/details/analysis_11'
os.makedirs(output_dir, exist_ok=True)

# Save actual vs predicted plot
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
plt.close()

# Feature importance using Random Forest as a surrogate model

# Get validation data
X_valid = valid_ds = to.valid.xs.copy()
y_valid = to.valid.y.values.flatten()

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_valid, y_valid)

# Get feature importance
rf = RandomForestRegressor()
rf.fit(X_valid, y_valid)
importance = rf.feature_importances_
feature_names = X_valid.columns

# Sort features by importance
indices = np.argsort(importance)[::-1]

# Plot top 15 features
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.barh(range(15), importance[indices[:15]], align='center')
plt.yticks(range(15), [feature_names[i] for i in indices[:15]])
plt.xlabel('Relative Importance')
plt.tight_layout()

# Save feature importance plot
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()