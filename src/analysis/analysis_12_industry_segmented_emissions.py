"""
Industry-Segmented Emissions Prediction Model for ESG Analysis

This script builds specialized emissions prediction models for different industry sectors to improve prediction accuracy.

Input DataFrames:
1. company_emission_df: Contains company-level emissions data with features for prediction
2. program1_df: Program participation data for the first program
3. program2_df: Program participation data for the second program
4. target_df: Target variable data for model training and evaluation

Key Analyses Performed:
1. Industry segmentation using NACE classification codes
2. Data preprocessing and feature engineering specific to each industry
3. Training of separate Random Forest regression models per industry
4. Model evaluation and performance comparison across industries

Outputs:
- Console logs with model performance metrics
- Visualizations of feature importance by industry
- Predictions for emissions intensity growth
"""

# ===== IMPORTS =====
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import mstats
import torch

# ===== CONSTANTS =====
# Column categories based on data dictionaries
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

# Target variable 
target_col = 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'

# ===== HELPER FUNCTIONS =====

def get_industry_group(nace_code):
    """
    Group NACE codes into broader industry categories.
    
    Parameters:
    1. nace_code (str): The NACE classification code to categorize
    
    Processing steps:
    1. Extract first 2 digits of the NACE code
    2. Map to predefined industry groups based on NACE code prefixes
    3. Return the corresponding industry group name
    
    Returns:
    str: The name of the industry group (e.g., 'Agriculture', 'Manufacturing - Tech')
    """
    if pd.isna(nace_code):
        return "Unknown"
    
    code_prefix = str(nace_code)[:2]
    
    # Define industry groupings based on NACE code prefixes
    if code_prefix in ['01', '02', '03']:
        return "Agriculture"
    elif code_prefix in ['05', '06', '07', '08', '09']:
        return "Mining"
    elif code_prefix in ['10', '11', '12']:
        return "Food & Beverage"
    elif code_prefix >= '13' and code_prefix <= '18':
        return "Textile & Paper"
    elif code_prefix in ['19', '20', '21', '22', '23', '24', '25']:
        return "Manufacturing - Heavy"
    elif code_prefix >= '26' and code_prefix <= '33':
        return "Manufacturing - Tech"
    elif code_prefix in ['35']:
        return "Energy"
    elif code_prefix in ['36', '37', '38', '39']:
        return "Water & Waste"
    elif code_prefix in ['41', '42', '43']:
        return "Construction"
    elif code_prefix >= '45' and code_prefix <= '47':
        return "Retail"
    elif code_prefix >= '49' and code_prefix <= '53':
        return "Transportation"
    elif code_prefix >= '55' and code_prefix <= '56':
        return "Hospitality"
    elif code_prefix >= '58' and code_prefix <= '63':
        return "Information & Communication"
    elif code_prefix >= '64' and code_prefix <= '66':
        return "Financial"
    elif code_prefix in ['68']:
        return "Real Estate"
    elif code_prefix >= '69' and code_prefix <= '75':
        return "Professional Services"
    elif code_prefix >= '77' and code_prefix <= '82':
        return "Administrative Services"
    elif code_prefix in ['84']:
        return "Public Administration"
    elif code_prefix in ['85']:
        return "Education"
    elif code_prefix >= '86' and code_prefix <= '88':
        return "Healthcare"
    elif code_prefix >= '90' and code_prefix <= '93':
        return "Arts & Entertainment"
    elif code_prefix >= '94' and code_prefix <= '96':
        return "Other Services"
    else:
        return "Other"

# ===== DATA PROCESSING =====

def prepare_data(model_data, is_industry_model=False):
    """
    Prepare data for modeling with consistent preprocessing.
    
    Parameters:
    1. model_data (DataFrame): Input dataset with features and target
    2. is_industry_model (bool): Flag indicating if processing for industry-specific model
    
    Processing steps:
    1. Create deep copy of input data
    2. Handle missing values in categorical and continuous features
    3. Remove rows with missing target values
    4. Remove features that may cause data leakage
    5. Perform feature selection using SelectKBest
    6. Apply winsorization to handle outliers
    7. Apply power transformations to continuous features
    8. Create interaction features and industry/regional features
    
    Returns:
    tuple: (processed_data, categorical_features, continuous_features)
    """
    model_data = model_data.copy()
    
    # Create final categorical and continuous feature lists
    # Filter to include only columns present in the dataset
    cat_features = [col for col in emission_cat_cols if col in model_data.columns]
    cont_features = [col for col in emission_cont_cols if col in model_data.columns and col != target_col]
    
    # Fill missing values
    for col in cont_features:
        model_data.loc[:, col] = model_data[col].fillna(0)
    for col in cat_features:
        model_data.loc[:, col] = model_data[col].fillna('#na#')
    
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
    
    print(f"Dataset shape: {model_data.shape}")
    print(f"Using {len(cat_features)} categorical features and {len(cont_features)} continuous features")
    
    # Feature selection to reduce dimensionality
    from sklearn.feature_selection import SelectKBest, f_regression
    X = model_data[cont_features]
    y = model_data[target_col]
    selector = SelectKBest(f_regression, k=min(15, len(cont_features)))
    selected_features = selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': cont_features,
        'Score': selector.scores_
    })
    
    # Keep only most important continuous features
    top_cont_features = feature_scores.sort_values('Score', ascending=False).head(15)['Feature'].tolist()
    cont_features = top_cont_features
    
    # Winsorize instead of simple outlier removal
    for col in cont_features:
        model_data.loc[:, col] = mstats.winsorize(model_data[col], limits=[0.01, 0.01])
    
    # Try more complex transformations like Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    transformed_data = pt.fit_transform(model_data[cont_features])
    for i, col in enumerate(cont_features):
        model_data.loc[:, col] = transformed_data[:, i]
    
    # Add interaction features 
    if 'SALES_USD_RECENT' in model_data.columns and 'CARBON_EMISSIONS_SCOPE_12' in model_data.columns:
        model_data.loc[:, 'size_emissions_ratio'] = model_data['SALES_USD_RECENT'] / (model_data['CARBON_EMISSIONS_SCOPE_12'] + 1)
    
    # Handle outliers in the target variable
    target_outliers = np.abs(model_data[target_col]) > 100
    model_data = model_data[~target_outliers]
    
    # Log-transform skewed features
    for col in cont_features:
        if model_data[col].skew() > 1:
            model_data.loc[:, col] = np.log1p(np.abs(model_data[col]))
    
    # Create industry-specific features
    if 'NACE_CLASS_CODE' in model_data.columns:
        # Only include industry_emission_ratio if we're not making a per-industry model
        # to avoid creating features with only one value
        if not is_industry_model:
            model_data.loc[:, 'industry_emission_ratio'] = model_data.groupby('NACE_CLASS_CODE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')
    
    # Create regional features
    if 'ISSUER_CNTRY_DOMICILE' in model_data.columns:
        model_data.loc[:, 'country_emission_ratio'] = model_data.groupby('ISSUER_CNTRY_DOMICILE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')
    
    return model_data, cat_features, cont_features

# ===== MODELING =====

def build_industry_model(model_data, industry_group):
    """
    Build and evaluate a model for a specific industry group.
    
    Parameters:
    1. model_data (DataFrame): Full dataset containing all industries
    2. industry_group (str): Name of the industry group to model
    
    Processing steps:
    1. Filter data for the specified industry group
    2. Check for sufficient sample size
    3. Prepare data with industry-specific preprocessing
    4. Split into training and validation sets
    5. Train Random Forest regressor
    6. Evaluate model performance
    
    Returns:
    tuple: (trained_model, feature_importances, evaluation_metrics) or None if insufficient data
    """
    industry_data = model_data[model_data['industry_group'] == industry_group].copy(deep=True)
    
    # Need enough data to build a model
    if len(industry_data) < 100:
        print(f"Insufficient data for industry: {industry_group} (only {len(industry_data)} samples)")
        return None
    
    # Process the data specifically for this industry
    industry_data, cat_features, cont_features = prepare_data(industry_data, is_industry_model=True)
    
    # Split data into training and validation sets
    train_idx, valid_idx = train_test_split(range(len(industry_data)), test_size=0.2, random_state=42)
    
    # Create TabularPandas object
    procs = [Categorify, FillMissing, Normalize]
    splits = (list(train_idx), list(valid_idx))
    
    to = TabularPandas(industry_data, procs=procs, cat_names=cat_features, cont_names=cont_features, 
                      y_names=target_col, splits=splits)
    
    # Create DataLoaders
    batch_size = min(64, len(train_idx) // 2)  # Adjust batch size based on data size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = to.dataloaders(bs=batch_size, device=device)
    
    # Define model
    y_min = industry_data[target_col].min() * 1.1  # Add 10% buffer
    y_max = industry_data[target_col].max() * 1.1
    
    # Create config dictionary for model parameters
    config = {
        'ps': [0.2, 0.2],  # Dropout probabilities
        'y_range': (y_min, y_max)
    }
    
    # Adjust layer sizes for smaller datasets
    layer_sizes = [100, 50] if len(industry_data) < 500 else [200, 100]
    
    learn = tabular_learner(dls, layers=layer_sizes, metrics=[rmse, mae],
                           wd=0.1, config=config,
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
    
    print(f"Industry: {industry_group}")
    print(f"Validation RMSE: {rmse_val:.4f}")
    print(f"Validation R²: {r2_val:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_targets, valid_preds, alpha=0.5)
    plt.plot([-100, 100], [-100, 100], 'r--')
    plt.xlabel('Actual Emission Trend (%)')
    plt.ylabel('Predicted Emission Trend (%)')
    plt.title(f'Actual vs Predicted - {industry_group}')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.grid(True)
    plt.savefig(os.path.join('output/details/analysis_12', f'actual_vs_predicted_{industry_group}.png'))
    plt.close()
    
    # Feature importance using Random Forest
    X_valid = to.valid.xs.copy()
    y_valid = to.valid.y.values.flatten()
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_valid, y_valid)
    
    # Get feature importance
    importance = rf.feature_importances_
    feature_names = X_valid.columns
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Plot top features
    top_n = min(10, len(feature_names))
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances - {industry_group}')
    plt.barh(range(top_n), importance[indices[:top_n]], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join('output/details/analysis_12', f'feature_importance_{industry_group}.png'))
    plt.close()
    
    return {
        'industry': industry_group,
        'model': learn,
        'rmse': rmse_val,
        'r2': r2_val,
        'sample_size': len(industry_data)
    }

# ===== MAIN EXECUTION =====

def run_industry_segmented_model(company_emission_df, program1_df, program2_df, target_df):
    """
    Run the industry-segmented model prediction workflow.
    
    Parameters:
    1. company_emission_df (DataFrame): Company emissions data
    2. program1_df (DataFrame): First program participation data
    3. program2_df (DataFrame): Second program participation data
    4. target_df (DataFrame): Target variable data
    
    Processing steps:
    1. Merge input data sources
    2. Apply industry grouping
    3. Process data for modeling
    4. Train models for each industry segment
    5. Generate predictions and evaluate performance
    6. Create visualizations of results
    
    Returns:
    dict: Dictionary containing trained models and evaluation results by industry
    """
    print("Starting industry-segmented emissions prediction model...")
    
    # Merge all datasets the same way as the original model - ensure we create a deep copy
    model_data = company_emission_df.copy(deep=True)
    
    # Add the industry grouping
    model_data.loc[:, 'industry_group'] = model_data['NACE_CLASS_CODE'].apply(get_industry_group)
    
    # Merge with other data sources
    if program1_df is not None:
        model_data = model_data.merge(program1_df, on='ISSUERID', how='left')
    
    # For program2 and target, we need to handle aggregation for one-to-many relationships
    if program2_df is not None:
        # Function to aggregate categorical columns with one-hot encoding
        def aggregate_categorical(df, cat_cols):
            """Create one-hot encoded features for categorical columns"""
            result = df[['ISSUERID']].drop_duplicates()
            
            for col in cat_cols:
                if col in df.columns:
                    # Skip columns with too many categories
                    if df[col].nunique() > 100:
                        continue
                        
                    # Create one-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col)
                    # Join with ISSUERID
                    dummy_df = df[['ISSUERID']].join(dummies)
                    # Aggregate by max (1 if the category exists for that company)
                    agg_df = dummy_df.groupby('ISSUERID').max().reset_index()
                    # Merge with result
                    result = result.merge(agg_df, on='ISSUERID', how='left')
            
            return result

        # Function to aggregate continuous columns
        def aggregate_continuous(df, cont_cols):
            """Create statistical aggregations for continuous columns"""
            result = df[['ISSUERID']].drop_duplicates()
            
            # Define aggregation functions for continuous variables
            agg_dict = {}
            for col in cont_cols:
                if col in df.columns:
                    agg_dict[col] = ['mean', 'min', 'max', 'count']
            
            if agg_dict:
                # Perform aggregation
                agg_df = df.groupby('ISSUERID').agg(agg_dict)
                # Flatten column names
                agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
                # Reset index
                agg_df = agg_df.reset_index()
                # Merge with result
                result = result.merge(agg_df, on='ISSUERID', how='left')
            
            return result
        
        # Aggregate features from program2_df
        program2_cat_cols = [col for col in program2_df.columns if col.startswith('CARBON_PROGRAMS_') and col.endswith(('CATEGORY', 'TYPE', 'OVERSIGHT', 'SOURCE'))]
        program2_cont_cols = ['CARBON_PROGRAMS_IMP_YEAR']
        
        program2_cat_features = aggregate_categorical(program2_df, program2_cat_cols)
        program2_cont_features = aggregate_continuous(program2_df, program2_cont_cols)
        program2_features = program2_cat_features.merge(program2_cont_features, on='ISSUERID', how='left')
        
        model_data = model_data.merge(program2_features, on='ISSUERID', how='left')
    
    if target_df is not None:
        # Aggregate features from target_df
        target_cat_cols = [col for col in target_df.columns if col.startswith(('CBN_TARGET_', 'TARGET_CARBON_')) and col.endswith(('AGGR', 'CATEGORY', 'SCOPE', 'STATUS', 'TYPE', 'UNITS'))]
        target_cont_cols = [col for col in target_df.columns if col.startswith(('CBN_TARGET_', 'TARGET_CARBON_')) and col.endswith(('YEAR', 'PCT', 'VAL', 'VALUE', 'VOLUME'))]
        
        target_cat_features = aggregate_categorical(target_df, target_cat_cols)
        target_cont_features = aggregate_continuous(target_df, target_cont_cols)
        target_features = target_cat_features.merge(target_cont_features, on='ISSUERID', how='left')
        
        model_data = model_data.merge(target_features, on='ISSUERID', how='left')
    
    # Get list of all industry groups with sufficient data
    industry_counts = model_data['industry_group'].value_counts()
    viable_industries = industry_counts[industry_counts >= 100].index.tolist()
    
    print(f"Found {len(viable_industries)} industry groups with sufficient data:")
    for industry in viable_industries:
        print(f"  - {industry}: {industry_counts[industry]} companies")
    
    # First build a general model using all data
    print("\n===== BUILDING GENERAL MODEL (ALL INDUSTRIES) =====")
    processed_data, cat_features, cont_features = prepare_data(model_data)
    
    from sklearn.model_selection import train_test_split
    train_idx, valid_idx = train_test_split(range(len(processed_data)), test_size=0.2, random_state=42)
    
    # Create TabularPandas object
    procs = [Categorify, FillMissing, Normalize]
    splits = (list(train_idx), list(valid_idx))
    
    to = TabularPandas(processed_data, procs=procs, cat_names=cat_features, cont_names=cont_features, 
                      y_names=target_col, splits=splits)
    
    # Create DataLoaders
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = to.dataloaders(bs=batch_size, device=device)
    
    # Define model
    y_min = processed_data[target_col].min() * 1.1  # Add 10% buffer
    y_max = processed_data[target_col].max() * 1.1
    
    # Create config dictionary for model parameters
    config = {
        'ps': [0.2, 0.2],  # Dropout probabilities
        'y_range': (y_min, y_max)
    }
    
    learn = tabular_learner(dls, layers=[200, 100], metrics=[rmse, mae],
                           wd=0.1, config=config,
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
    
    print(f"General Model - Validation RMSE: {rmse_val:.4f}")
    print(f"General Model - Validation R²: {r2_val:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_targets, valid_preds, alpha=0.5)
    plt.plot([-100, 100], [-100, 100], 'r--')
    plt.xlabel('Actual Emission Trend (%)')
    plt.ylabel('Predicted Emission Trend (%)')
    plt.title('Actual vs Predicted - General Model')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.grid(True)
    plt.savefig(os.path.join('output/details/analysis_12', 'actual_vs_predicted_general.png'))
    plt.close()
    
    # Now build models for each industry group
    print("\n===== BUILDING INDUSTRY-SPECIFIC MODELS =====")
    industry_models = []
    
    for industry in viable_industries:
        print(f"\n----- Processing Industry: {industry} -----")
        model_result = build_industry_model(model_data, industry)
        if model_result:
            industry_models.append(model_result)
    
    # Compare results across industries
    if industry_models:
        results_df = pd.DataFrame(industry_models)
        results_df = results_df.sort_values('r2', ascending=False)
        
        print("\n===== INDUSTRY MODEL COMPARISON =====")
        print(results_df[['industry', 'rmse', 'r2', 'sample_size']])
        
        # Plot R² comparison across industries
        plt.figure(figsize=(12, 8))
        plt.bar(results_df['industry'], results_df['r2'])
        plt.axhline(y=r2_val, color='r', linestyle='-', label=f'General Model R² ({r2_val:.4f})')
        plt.xlabel('Industry')
        plt.ylabel('R²')
        plt.title('Model Performance Comparison by Industry')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join('output/details/analysis_12', 'industry_r2_comparison.png'))
        plt.close()
        
        # Plot RMSE comparison across industries
        plt.figure(figsize=(12, 8))
        plt.bar(results_df['industry'], results_df['rmse'])
        plt.axhline(y=rmse_val, color='r', linestyle='-', label=f'General Model RMSE ({rmse_val:.4f})')
        plt.xlabel('Industry')
        plt.ylabel('RMSE')
        plt.title('Error Comparison by Industry')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join('output/details/analysis_12', 'industry_rmse_comparison.png'))
        plt.close()
    
    return {
        'general_model': {
            'model': learn,
            'rmse': rmse_val,
            'r2': r2_val
        },
        'industry_models': industry_models
    }

# ===== SCRIPT ENTRY POINT =====
if __name__ == "__main__":
    print("Industry-Segmented Emissions Prediction Model")
    print("=" * 50)
    # Load required dataframes from CSVs
    company_emission_df = pd.read_csv("data/company_emissions_merged.csv")
    program1_df = pd.read_csv("data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv")
    program2_df = pd.read_csv("data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv")
    target_df = pd.read_csv("data/Reduktionsziele Results - 20241212 15_49_29.csv")
    # Run the analysis
    results = run_industry_segmented_model(company_emission_df, program1_df, program2_df, target_df) 