"""
Industry-Segmented Emissions Prediction with Gradient Boosting Models

This script implements gradient boosting models (LightGBM) for predicting carbon emissions trends,
segmented by industry to improve predictive performance.

Input DataFrames:
1. company_emission_df: Contains company-level emissions and financial data
2. program1_df: First program participation data (optional)
3. program2_df: Second program participation data (optional)
4. target_df: Target variable data with emissions intensity growth rates

Key Analyses Performed:
1. Industry segmentation using NACE classification codes
2. Data preprocessing with industry-specific feature engineering
3. Training of LightGBM models for each industry segment
4. Model evaluation and performance comparison across industries
5. Feature importance analysis for each industry segment

Outputs:
- Console logs with model performance metrics for each industry
- Feature importance visualizations by industry
- Predictions for emissions intensity growth rates
"""

# ===== IMPORTS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import mstats, pearsonr
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

# ===== CONSTANTS =====

# Column categories based on data dictionaries
categorical_cols = [
    'ISSUER_CNTRY_DOMICILE',
    'NACE_CLASS_CODE',
    'NACE_CLASS_DESCRIPTION',
    'CARBON_EMISSIONS_SCOPE_1_KEY',
    'CARBON_EMISSIONS_SCOPE_12_KEY',
    'CARBON_EMISSIONS_SCOPE_2_KEY'
]

continuous_cols = [
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
    Map NACE classification codes to broader industry groups.
    
    Parameters:
    1. nace_code (str/int): NACE classification code to be categorized
    
    Processing steps:
    1. Convert input to string and extract first 2 digits
    2. Map numeric code to predefined industry groups
    3. Return corresponding industry group name
    
    Returns:
    str: Industry group name (e.g., 'Manufacturing - Tech', 'Energy')
    
    Side Effects:
    - None
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

def prepare_data_for_gbm(model_data, is_industry_model=False):
    """
    Prepare and preprocess data for GBM modeling.
    
    Parameters:
    1. model_data (DataFrame): Input dataset containing features and target
    2. is_industry_model (bool): Flag indicating if processing industry-specific data
    
    Processing steps:
    1. Create deep copy of input data
    2. Add industry grouping if not present
    3. Handle missing values in continuous and categorical features
    4. Remove rows with missing target values
    5. Remove features that may cause data leakage
    6. Handle outliers using winsorization
    7. Create interaction and derived features
    8. Ensure proper data types for modeling
    
    Returns:
    DataFrame: Processed dataset ready for modeling
    
    Side Effects:
    - Modifies input dataframe (creates a copy first)
    - Prints dataset shape and feature information
    """
    model_data = model_data.copy()
    
    # Add the industry grouping if not already present
    if 'industry_group' not in model_data.columns and 'NACE_CLASS_CODE' in model_data.columns:
        model_data.loc[:, 'industry_group'] = model_data['NACE_CLASS_CODE'].apply(get_industry_group)
    
    # Create final categorical and continuous feature lists
    # Filter to include only columns present in the dataset
    cat_features = [col for col in categorical_cols if col in model_data.columns]
    cont_features = [col for col in continuous_cols if col in model_data.columns and col != target_col]
    
    # Fill missing values
    for col in cont_features:
        model_data.loc[:, col] = pd.to_numeric(model_data[col], errors='coerce').fillna(0)
    
    for col in cat_features:
        model_data.loc[:, col] = model_data[col].fillna('unknown')
    
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
    
    # Handle outliers in the target variable
    target_outliers = np.abs(model_data[target_col]) > 100
    model_data = model_data[~target_outliers]
    
    # Winsorize continuous features (trim outliers)
    for col in cont_features:
        model_data.loc[:, col] = mstats.winsorize(model_data[col], limits=[0.01, 0.01])
    
    # Add interaction features that might be useful for GBM
    if 'SALES_USD_RECENT' in model_data.columns and 'CARBON_EMISSIONS_SCOPE_12' in model_data.columns:
        model_data.loc[:, 'size_emissions_ratio'] = model_data['SALES_USD_RECENT'] / (model_data['CARBON_EMISSIONS_SCOPE_12'] + 1)
        # Add to continuous features
        cont_features.append('size_emissions_ratio')
    
    # Create industry-specific features
    if 'NACE_CLASS_CODE' in model_data.columns:
        # Only include industry_emission_ratio if we're not making a per-industry model
        if not is_industry_model:
            model_data.loc[:, 'industry_emission_ratio'] = model_data.groupby('NACE_CLASS_CODE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')
            # Add to continuous features
            cont_features.append('industry_emission_ratio')
    
    # Create regional features
    if 'ISSUER_CNTRY_DOMICILE' in model_data.columns:
        model_data.loc[:, 'country_emission_ratio'] = model_data.groupby('ISSUER_CNTRY_DOMICILE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].transform('mean')
        # Add to continuous features
        cont_features.append('country_emission_ratio')
    
    # Create percentile ranks within industry
    if 'NACE_CLASS_CODE' in model_data.columns and 'CARBON_EMISSIONS_SCOPE_12_INTEN' in model_data.columns:
        model_data.loc[:, 'emissions_intensity_rank'] = model_data.groupby('NACE_CLASS_CODE')['CARBON_EMISSIONS_SCOPE_12_INTEN'].rank(pct=True)
        # Add to continuous features
        cont_features.append('emissions_intensity_rank')
    
    # Ensure the target column is numeric
    model_data[target_col] = pd.to_numeric(model_data[target_col], errors='coerce')
    
    # Ensure industry_group is included in categorical features if it exists
    if 'industry_group' in model_data.columns and 'industry_group' not in cat_features:
        cat_features.append('industry_group')
    
    return model_data

# ===== MODEL TRAINING =====

def train_industry_gbm(model_data, industry_group, cat_features, cont_features):
    """
    Train and evaluate a GBM model for a specific industry.
    
    Parameters:
    1. model_data (DataFrame): Full dataset containing all industries
    2. industry_group (str): Name of the industry group to model
    3. cat_features (list): List of categorical feature names
    4. cont_features (list): List of continuous feature names
    
    Processing steps:
    1. Filter data for the specified industry
    2. Check for sufficient sample size
    3. Prepare features and target variable
    4. Handle categorical features
    5. Split into training and testing sets
    6. Train LightGBM model with early stopping
    7. Evaluate model performance
    
    Returns:
    tuple: (trained_model, feature_importances, evaluation_metrics) or None if insufficient data
    
    Side Effects:
    - Prints model training progress and evaluation metrics
    - May create visualizations of feature importance
    """
    industry_data = model_data[model_data['industry_group'] == industry_group].copy(deep=True)
    
    # Need enough data to build a model
    if len(industry_data) < 50:  # Lower threshold for GBM compared to NN
        print(f"Insufficient data for industry: {industry_group} (only {len(industry_data)} samples)")
        return None
    
    # Split data into features and target
    X = industry_data[cat_features + cont_features].copy()
    y = industry_data[target_col].copy()
    
    # Handle categorical features - convert to category type
    cat_feature_indices = []
    for i, col in enumerate(X.columns):
        if col in cat_features:
            # Convert to categorical first, then to codes
            X[col] = X[col].astype('category').cat.codes.astype('int64')
            cat_feature_indices.append(i)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=cat_feature_indices
    )
    
    val_data = lgb.Dataset(
        X_val, 
        label=y_val,
        categorical_feature=cat_feature_indices,
        reference=train_data
    )
    
    # Train the model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print(f"Training GBM model for industry: {industry_group} with {len(X_train)} samples")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    pearson = pearsonr(y_val, y_pred)[0]
    
    print(f"Industry: {industry_group} - RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson:.4f}")
    
    # Return the model and metrics
    return {
        'model': model,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson
        },
        'importance': model.feature_importance(importance_type='gain'),
        'feature_names': X.columns.tolist()
    }

def train_general_gbm(model_data, cat_features, cont_features):
    """
    Train a general GBM model using all industries combined.
    
    Parameters:
    1. model_data (DataFrame): Dataset containing all industries
    2. cat_features (list): List of categorical feature names
    3. cont_features (list): List of continuous feature names
    
    Processing steps:
    1. Prepare features and target variable
    2. Handle categorical features
    3. Split into training and testing sets
    4. Train LightGBM model with cross-validation
    5. Evaluate model performance
    
    Returns:
    tuple: (trained_model, feature_importances, evaluation_metrics)
    
    Side Effects:
    - Prints model training progress and evaluation metrics
    - May create visualizations of feature importance
    """
    # Split data into features and target
    X = model_data[cat_features + cont_features].copy()
    y = model_data[target_col].copy()
    
    # Handle categorical features - convert to category type
    cat_feature_indices = []
    for i, col in enumerate(X.columns):
        if col in cat_features:
            # Convert to categorical first, then to codes
            X[col] = X[col].astype('category').cat.codes.astype('int64')
            cat_feature_indices.append(i)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=cat_feature_indices
    )
    
    val_data = lgb.Dataset(
        X_val, 
        label=y_val,
        categorical_feature=cat_feature_indices,
        reference=train_data
    )
    
    # Train the model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print(f"Training general GBM model with {len(X_train)} samples")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    pearson = pearsonr(y_val, y_pred)[0]
    
    print(f"General model - RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson:.4f}")
    
    # Return the model and metrics
    return {
        'model': model,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson
        },
        'importance': model.feature_importance(importance_type='gain'),
        'feature_names': X.columns.tolist()
    }

# ===== MAIN EXECUTION =====

def run_gbm_industry_segmentation(company_emission_df, program1_df=None, program2_df=None, target_df=None):
    """
    Run the GBM-based industry-segmented emissions prediction workflow.
    
    Parameters:
    1. company_emission_df (DataFrame): Company emissions data
    2. program1_df (DataFrame, optional): First program participation data
    3. program2_df (DataFrame, optional): Second program participation data
    4. target_df (DataFrame, optional): Target variable data
    
    Processing steps:
    1. Merge input data sources
    2. Apply industry grouping
    3. Process data for modeling
    4. Train general model on all industries
    5. Train industry-specific models
    6. Compare model performance
    7. Generate predictions and visualizations
    
    Returns:
    dict: Dictionary containing trained models and evaluation results
    
    Side Effects:
    - Prints progress information and model performance
    - Saves visualizations to disk
    """
    print("Starting GBM-based industry-segmented emissions prediction...")
    
    # Create a copy to avoid modifying the original
    model_data = company_emission_df.copy()
    
    # Verify the target column exists
    if target_col not in model_data.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return None
    
    # Add the industry grouping if not already present
    if 'industry_group' not in model_data.columns and 'NACE_CLASS_CODE' in model_data.columns:
        model_data.loc[:, 'industry_group'] = model_data['NACE_CLASS_CODE'].apply(get_industry_group)
    
    # Print summary of available industries
    if 'industry_group' in model_data.columns:
        industry_counts = model_data['industry_group'].value_counts()
        print(f"\nIndustry distribution in the data:")
        print(industry_counts)
    
    # Merge with program1_df if provided
    if program1_df is not None and 'ISSUERID' in program1_df.columns:
        print(f"\nMerging with program1 data ({program1_df.shape[0]} rows)...")
        # Select only common columns for merging
        common_columns = ['ISSUERID'] + [col for col in categorical_cols if col in program1_df.columns]
        program1_subset = program1_df[common_columns].drop_duplicates()
        model_data = model_data.merge(program1_subset, on='ISSUERID', how='left')
        print(f"After program1 merge: {model_data.shape}")
    
    # Merge with program2_df if provided
    if program2_df is not None and 'ISSUERID' in program2_df.columns:
        print(f"\nMerging with program2 data ({program2_df.shape[0]} rows)...")
        # For program2, we may have multiple entries per company, so we need to aggregate
        # Group by ISSUERID and take the most recent year's data
        if 'CBN_IMP_PROG_YEAR' in program2_df.columns:
            print("Aggregating program2 data by most recent year...")
            program2_subset = program2_df.sort_values('CBN_IMP_PROG_YEAR', ascending=False)
            program2_subset = program2_subset.drop_duplicates('ISSUERID')
        else:
            program2_subset = program2_df.drop_duplicates('ISSUERID')
        
        # Select only columns that might be useful
        useful_cols = ['ISSUERID'] + [
            col for col in program2_subset.columns 
            if any(keyword in col for keyword in ['CBN_PROG', 'CBN_EVIDENCE', 'EXEC_BODY', 'CBN_REG'])
        ]
        program2_subset = program2_subset[useful_cols].drop_duplicates()
        
        # Merge with main data
        model_data = model_data.merge(program2_subset, on='ISSUERID', how='left')
        print(f"After program2 merge: {model_data.shape}")
    
    # Merge with target_df if provided
    if target_df is not None and 'ISSUERID' in target_df.columns:
        print(f"\nMerging with target data ({target_df.shape[0]} rows)...")
        # For targets, we need to aggregate data by company
        # Group by ISSUERID and aggregate key metrics
        target_agg = target_df.groupby('ISSUERID').agg({
            'CBN_TARGET_REDUC_PCT': 'mean',
            'CBN_TARGET_IMP_YEAR': 'max',
            'CBN_TARGET_BASE_YEAR': 'min',
            'TARGET_CARBON_PROGRESS_PCT': 'max'
        }).reset_index()
        
        # Merge with main data
        model_data = model_data.merge(target_agg, on='ISSUERID', how='left')
        print(f"After target merge: {model_data.shape}")
    
    # Now prepare the data for GBM
    model_data = prepare_data_for_gbm(model_data, is_industry_model=True)
    print(f"Final dataset shape: {model_data.shape}")
    
    # Identify industries with enough data
    industry_counts = model_data['industry_group'].value_counts()
    sufficient_industries = industry_counts[industry_counts >= 50].index.tolist()
    
    print(f"Found {len(sufficient_industries)} industry groups with sufficient data:")
    for industry in sufficient_industries:
        print(f"  - {industry}: {industry_counts[industry]} companies")
    
    # Define features
    all_cat_features = [col for col in model_data.columns if col in categorical_cols and col in model_data.columns]
    all_cont_features = [col for col in model_data.columns if col in continuous_cols and col in model_data.columns]
    
    print(f"Using {len(all_cat_features)} categorical features and {len(all_cont_features)} continuous features")
    
    # Train general model first
    general_model_result = train_general_gbm(model_data, all_cat_features, all_cont_features)
    
    # Train industry-specific models
    industry_models = []
    for industry in sufficient_industries:
        result = train_industry_gbm(model_data, industry, all_cat_features, all_cont_features)
        if result is not None:
            result['industry'] = industry  # Add industry name to result
            industry_models.append(result)
    
    # Compare industry models to general model
    if industry_models:
        # Create a DataFrame for visualization
        results_df = pd.DataFrame([
            {
                'industry': model['industry'],
                'rmse': model['metrics']['rmse'],
                'r2': model['metrics']['r2'],
                'pearson': model['metrics']['pearson']
            } for model in industry_models
        ])
        
        # Sort by R²
        results_df = results_df.sort_values('r2', ascending=False)
        
        # Plot R² by industry
        plt.figure(figsize=(12, 8))
        plt.bar(results_df['industry'], results_df['r2'])
        plt.axhline(y=general_model_result['metrics']['r2'], color='r', linestyle='-', 
                    label=f'General Model R² ({general_model_result["metrics"]["r2"]:.4f})')
        plt.xlabel('Industry')
        plt.ylabel('R²')
        plt.title('Model Performance by Industry (R²)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        # Ensure output directory exists
        os.makedirs('output/summary/png', exist_ok=True)
        plt.savefig('output/summary/png/gbm_industry_r2_comparison.png')
        
        # Plot RMSE by industry
        plt.figure(figsize=(12, 8))
        plt.bar(results_df['industry'], results_df['rmse'])
        plt.axhline(y=general_model_result['metrics']['rmse'], color='r', linestyle='-', 
                    label=f'General Model RMSE ({general_model_result["metrics"]["rmse"]:.4f})')
        plt.xlabel('Industry')
        plt.ylabel('RMSE')
        plt.title('Model Performance by Industry (RMSE)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/summary/png/gbm_industry_rmse_comparison.png')
        
        # Identify top industries that outperform the general model
        outperformers = results_df[results_df['r2'] > general_model_result['metrics']['r2']]
        print(f"\n{len(outperformers)} industries outperform the general model:")
        for _, row in outperformers.iterrows():
            print(f"  - {row['industry']}: R² = {row['r2']:.4f} (vs. General: {general_model_result['metrics']['r2']:.4f})")
        
        # Display top features by industry
        print("\n===== TOP FEATURES BY INDUSTRY =====")
        for model in industry_models:
            if model['metrics']['r2'] > 0:  # Only show positive R² models
                # Get feature importance and names
                importance = model['importance']
                feature_names = model['feature_names']
                
                # Sort features by importance
                sorted_idx = np.argsort(importance)[::-1]
                
                print(f"\nTop 5 features for {model['industry']} (R² = {model['metrics']['r2']:.4f}):")
                for i in range(min(5, len(sorted_idx))):
                    idx = sorted_idx[i]
                    if idx < len(feature_names):
                        print(f"  - {feature_names[idx]}: {importance[idx]:.2f}")
    
    return {
        'general_model': general_model_result,
        'industry_models': industry_models
    }

# ===== SCRIPT ENTRY POINT =====

if __name__ == "__main__":
    # Load required dataframes from CSVs
    company_emission_df = pd.read_csv("data/company_emissions_merged.csv")
    program1_df = pd.read_csv("data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv")
    program2_df = pd.read_csv("data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv")
    target_df = pd.read_csv("data/Reduktionsziele Results - 20241212 15_49_29.csv")
    print("Running GBM industry-segmented emissions prediction...")
    results = run_gbm_industry_segmentation(company_emission_df, program1_df, program2_df, target_df)
    print("GBM industry-segmented emissions prediction completed.")
    # Optionally, print or save results
    if results is not None:
        print(results)