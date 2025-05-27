"""
Program and Target Focused Emissions Prediction Model

This module implements a LightGBM model to analyze the impact of sustainability
programs and targets on emissions trends across different industry sectors.

Input DataFrames:
- company_emission_df: Company-level emissions and financial data
- program1_df: First program participation data
- program2_df: Second program participation data
- target_df: Target setting and achievement data

Key Functionality:
1. Data preprocessing and feature engineering
2. Industry segmentation using NACE codes
3. Gradient Boosting Machine (GBM) model training
4. Model evaluation and interpretation
5. Visualization of results

Outputs:
- Console logs with model performance metrics
- Feature importance visualizations
- SHAP value analysis plots
- Program effectiveness comparison charts
"""

# ===== IMPORTS =====
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import mstats, pearsonr

warnings.filterwarnings('ignore', category=UserWarning)

# ===== CONSTANTS =====
# Target variable we're trying to predict
target_col = 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'

# Categorize features for reuse
# Basic company features that we'll control for
control_cols = [
    'ISSUER_CNTRY_DOMICILE',
    'NACE_CLASS_CODE',
    'MarketCap_USD',
    'SALES_USD_RECENT',
    'CARBON_EMISSIONS_SCOPE_1_INTEN',
    'CARBON_EMISSIONS_SCOPE_2_INTEN',
    'CARBON_EMISSIONS_SCOPE_12_INTEN',
    'CARBON_SCOPE_12_INTEN_3Y_AVG'
]

# Program features that are our main focus
program_cols = [
    # Will populate from available columns
]

# Target features that are our main focus
target_cols = [
    # Will populate from available columns
]

# Get a list of all program/target related columns from the actual data
def identify_program_columns(df):
    """
    Identify all program and target related columns in the dataframe.
    
    Parameters:
    1. df (DataFrame): Input dataframe
    
    Processing steps:
    1. Iterate through columns and check for keywords
    2. Return lists of program and target columns
    
    Returns:
    tuple: (program_cols, target_cols)
    
    Side Effects:
    - None
    """
    """Identify all program and target related columns in the dataframe"""
    prog_cols = [col for col in df.columns if any(keyword in col for keyword in 
                ['CBN_PROG', 'CBN_EVIDENCE', 'EXEC_BODY', 'CBN_REG'])]
    targ_cols = [col for col in df.columns if any(keyword in col for keyword in 
                ['CBN_TARGET', 'TARGET_CARBON'])]
    return prog_cols, targ_cols 

# ===== DATA LOADING =====

def load_data():
    """
    Load and prepare the datasets for emissions prediction.
    
    Processing steps:
    1. Load company emissions data from CSV
    2. Load program participation data (program1 and program2)
    3. Load target setting data
    4. Print dataset shapes for verification
    
    Returns:
    tuple: (company_emission_df, program1_df, program2_df, target_df)
    
    Side Effects:
    - Reads data from CSV files
    - Prints loading progress information
    """
    print("Loading datasets...")
    
    # Load company emissions data
    company_emission_df = pd.read_csv('data/company_emissions_merged.csv')
    print(f"Loaded company emissions data: {company_emission_df.shape}")
    
    # Load reduction programs 1 data
    program1_df = pd.read_csv('data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv', 
                              quotechar='"', escapechar='\\')
    print(f"Loaded program1 data: {program1_df.shape}")
    
    # Load reduction programs 2 data - using quoting=3 (QUOTE_NONE) to ignore quotes within fields
    program2_df = pd.read_csv('data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv', 
                              quotechar='"', escapechar='\\', on_bad_lines='skip')
    print(f"Loaded program2 data: {program2_df.shape}")
    
    # Load reduction targets data
    target_df = pd.read_csv('data/Reduktionsziele Results - 20241212 15_49_29.csv', 
                            quotechar='"', escapechar='\\', on_bad_lines='skip')
    print(f"Loaded target data: {target_df.shape}")
    
    return company_emission_df, program1_df, program2_df, target_df

# ===== DATA PREPARATION =====

def prepare_merged_data(company_emission_df, program1_df, program2_df, target_df):
    """
    Merge all data sources and prepare for modeling.
    
    Parameters:
    1. company_emission_df (DataFrame): Company emissions data
    2. program1_df (DataFrame): First program participation data
    3. program2_df (DataFrame): Second program participation data
    4. target_df (DataFrame): Target setting and achievement data
    
    Processing steps:
    1. Create a copy of the company emissions data
    2. Add industry grouping based on NACE codes
    3. Merge with program participation data
    4. Merge with target data
    5. Handle missing values and data types
    
    Returns:
    DataFrame: Merged and processed dataset ready for analysis
    
    Side Effects:
    - Prints merge progress and dataset shapes
    - May modify input dataframes (creates copies first)
    """
    print("Merging datasets...")
    
    # Create a copy to avoid modifying the original
    model_data = company_emission_df.copy()
    
    # Verify the target column exists
    if target_col not in model_data.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return None
    
    # Add the industry grouping based on NACE code
    model_data['industry_group'] = model_data['NACE_CLASS_CODE'].apply(get_industry_group)
    
    # Print summary of available industries
    if 'industry_group' in model_data.columns:
        industry_counts = model_data['industry_group'].value_counts()
        print(f"\nIndustry distribution in the data:")
        print(industry_counts)
    
    # Merge with program1_df if provided
    if program1_df is not None and 'ISSUERID' in program1_df.columns:
        print(f"\nMerging with program1 data ({program1_df.shape[0]} rows)...")
        program1_subset = program1_df.drop_duplicates('ISSUERID')
        model_data = model_data.merge(program1_subset, on='ISSUERID', how='left')
        print(f"After program1 merge: {model_data.shape}")
    
    # Merge with program2_df if provided - aggregate by most recent year
    if program2_df is not None and 'ISSUERID' in program2_df.columns:
        print(f"\nMerging with program2 data ({program2_df.shape[0]} rows)...")
        if 'CBN_IMP_PROG_YEAR' in program2_df.columns:
            program2_subset = program2_df.sort_values('CBN_IMP_PROG_YEAR', ascending=False)
            program2_subset = program2_subset.drop_duplicates('ISSUERID')
        else:
            program2_subset = program2_df.drop_duplicates('ISSUERID')
        model_data = model_data.merge(program2_subset, on='ISSUERID', how='left')
        print(f"After program2 merge: {model_data.shape}")
    
    # Merge with target_df if provided - aggregate key metrics
    if target_df is not None and 'ISSUERID' in target_df.columns:
        print(f"\nMerging with target data ({target_df.shape[0]} rows)...")
        target_agg = target_df.groupby('ISSUERID').agg({
            'CBN_TARGET_REDUC_PCT': 'mean',
            'CBN_TARGET_IMP_YEAR': 'max',
            'CBN_TARGET_BASE_YEAR': 'min',
            'TARGET_CARBON_PROGRESS_PCT': 'max'
        }).reset_index()
        model_data = model_data.merge(target_agg, on='ISSUERID', how='left')
        print(f"After target merge: {model_data.shape}")
    
    return model_data

# ===== INDUSTRY CLASSIFICATION =====

def get_industry_group(nace_code):
    """
    Map NACE classification codes to broader industry groups.
    
    Parameters:
    1. nace_code (str/int): NACE classification code to be categorized
    
    Processing steps:
    1. Handle missing values
    2. Convert input to string and extract first 2 digits
    3. Map numeric code to predefined industry groups
    
    Returns:
    str: Industry group name (e.g., 'Manufacturing', 'Energy')
    
    Side Effects:
    - None
    """
    if pd.isna(nace_code):
        return "Unknown"
    
    nace_str = str(nace_code)
    
    # Extract the first two digits for the division
    division = nace_str[:2] if len(nace_str) >= 2 else nace_str
    
    try:
        division_num = int(division)
        
        # Agriculture, Forestry and Fishing
        if 1 <= division_num <= 3:
            return "Agriculture"
        
        # Mining and Quarrying
        elif 5 <= division_num <= 9:
            return "Energy"
        
        # Manufacturing
        elif 10 <= division_num <= 18:
            return "Food & Beverage"
        elif 19 <= division_num <= 23:
            return "Manufacturing - Heavy"
        elif 24 <= division_num <= 25:
            return "Manufacturing - Heavy"
        elif 26 <= division_num <= 33:
            return "Manufacturing - Tech"
        
        # Utilities
        elif 35 <= division_num <= 39:
            return "Water & Waste"
        
        # Construction
        elif 41 <= division_num <= 43:
            return "Construction"
        
        # Wholesale and Retail Trade
        elif 45 <= division_num <= 47:
            return "Retail"
        
        # Transportation and Storage
        elif 49 <= division_num <= 53:
            return "Transportation"
        
        # Accommodation and Food Service
        elif 55 <= division_num <= 56:
            return "Hospitality"
        
        # Information and Communication
        elif 58 <= division_num <= 63:
            return "Information & Communication"
        
        # Financial and Insurance Activities
        elif 64 <= division_num <= 66:
            return "Financial"
        
        # Real Estate Activities
        elif division_num == 68:
            return "Real Estate"
        
        # Professional, Scientific and Technical
        elif 69 <= division_num <= 75:
            return "Professional Services"
        
        # Administrative and Support Service
        elif 77 <= division_num <= 82:
            return "Administrative Services"
        
        # Public Administration
        elif division_num == 84:
            return "Public Administration"
        
        # Education
        elif division_num == 85:
            return "Education"
        
        # Human Health and Social Work
        elif 86 <= division_num <= 88:
            return "Healthcare"
        
        # Arts, Entertainment and Recreation
        elif 90 <= division_num <= 93:
            return "Arts & Entertainment"
        
        # Other Service Activities
        elif 94 <= division_num <= 96:
            return "Other Services"
        
        # Other
        else:
            return "Other"
            
    except (ValueError, TypeError):
        return "Unknown" 

# ===== FEATURE ENGINEERING =====

def engineer_program_features(model_data):
    """
    Create derived features that highlight program and target impacts.
    
    Parameters:
    1. model_data (DataFrame): Input dataset with base features
    
    Processing steps:
    1. Calculate time since program implementation
    2. Create interaction terms between program participation and company characteristics
    3. Derive target achievement metrics
    4. Create industry-normalized metrics
    
    Returns:
    DataFrame: Enhanced dataset with new derived features
    
    Side Effects:
    - Modifies input dataframe (creates a copy first)
    - Prints feature engineering progress
    """
    print("Engineering program and target features...")
    
    # Create a working copy
    data = model_data.copy()
    
    # 1. Target ambition features
    if 'CBN_TARGET_REDUC_PCT' in data.columns:
        # Fill missing values with 0 (no target)
        data['CBN_TARGET_REDUC_PCT'] = data['CBN_TARGET_REDUC_PCT'].fillna(0)
        
        # Target timeline - years to target completion
        if 'CBN_TARGET_IMP_YEAR' in data.columns and 'CBN_TARGET_BASE_YEAR' in data.columns:
            data['target_timeline_years'] = data['CBN_TARGET_IMP_YEAR'] - data['CBN_TARGET_BASE_YEAR']
            data['target_timeline_years'] = data['target_timeline_years'].fillna(0)
            
            # Reduction per year (ambition pace)
            data['target_reduction_per_year'] = data['CBN_TARGET_REDUC_PCT'] / data['target_timeline_years']
            # Handle division by zero
            data['target_reduction_per_year'] = data['target_reduction_per_year'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Target vs. Industry benchmark
    if 'CBN_TARGET_REDUC_PCT' in data.columns and 'industry_group' in data.columns:
        # Calculate industry average target
        industry_avg_target = data.groupby('industry_group')['CBN_TARGET_REDUC_PCT'].transform('mean')
        # Target ambition relative to industry (>1 means more ambitious than industry average)
        data['target_vs_industry_avg'] = data['CBN_TARGET_REDUC_PCT'] / (industry_avg_target + 0.001)
        # Handle division issues
        data['target_vs_industry_avg'] = data['target_vs_industry_avg'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Target progress features
    if 'TARGET_CARBON_PROGRESS_PCT' in data.columns:
        data['TARGET_CARBON_PROGRESS_PCT'] = data['TARGET_CARBON_PROGRESS_PCT'].fillna(0)
        # Target achievement indicator (1 if achieved, 0 if not)
        data['target_achieved'] = (data['TARGET_CARBON_PROGRESS_PCT'] >= 100).astype(int)
        
        # Progress vs timeline - if the company is on track
        if 'CBN_TARGET_IMP_YEAR' in data.columns and 'CBN_TARGET_BASE_YEAR' in data.columns:
            current_year = 2024
            data['target_elapsed_pct'] = np.where(
                data['target_timeline_years'] > 0,
                (current_year - data['CBN_TARGET_BASE_YEAR']) / data['target_timeline_years'],
                0
            )
            data['target_elapsed_pct'] = data['target_elapsed_pct'].clip(0, 1).fillna(0)
            
            # Progress vs elapsed time ratio (>1 means ahead of schedule)
            data['progress_vs_timeline'] = np.where(
                data['target_elapsed_pct'] > 0,
                data['TARGET_CARBON_PROGRESS_PCT'] / 100 / data['target_elapsed_pct'],
                0
            )
            data['progress_vs_timeline'] = data['progress_vs_timeline'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 4. Program features
    # Count number of programs/initiatives (as a proxy for commitment)
    program_columns = [col for col in data.columns if any(x in col for x in ['CBN_PROG', 'CBN_EVIDENCE'])]
    for col in program_columns:
        # Replace NaNs with "No" or empty string
        if col in data.columns:
            data[col] = data[col].fillna('No evidence')
    
    # Create a program implementation score (higher = more comprehensive programs)
    program_indicators = []
    
    # Low carbon energy programs
    if 'CBN_PROG_LOW_CARB_RENEW' in data.columns:
        # Create a numeric score based on the text value
        data['low_carbon_program_score'] = data['CBN_PROG_LOW_CARB_RENEW'].map({
            'No evidence': 0,
            'Minimal evidence': 1,
            'Less than 50% of energy requirements from low-carbon emissions alternatives (estimated)': 2,
            'More than 50% of energy requirements from low-carbon emissions alternatives (estimated)': 3,
            'All energy from low-carbon emissions alternatives': 4
        }).fillna(0)
        program_indicators.append('low_carbon_program_score')
    
    # Energy efficiency programs
    if 'CBN_EVIDENCE_TARG_ENERGY_IMPROV' in data.columns:
        data['energy_efficiency_score'] = data['CBN_EVIDENCE_TARG_ENERGY_IMPROV'].map({
            'No evidence': 0,
            'Minimal evidence': 1,
            'Evidence of improvements in some facilities': 2,
            'Evidence of improvements across the business': 3,
            'Evidence of significant improvements across the business': 4
        }).fillna(0)
        program_indicators.append('energy_efficiency_score')
    
    # Carbon reduction programs
    if 'CBN_PROG_REDU_CARB_CORE_OP' in data.columns:
        data['carbon_reduction_score'] = data['CBN_PROG_REDU_CARB_CORE_OP'].map({
            'No evidence': 0,
            'Minimal evidence': 1,
            'Programs to reduce carbon emissions at selected operations': 2,
            'Programs to reduce carbon emissions across the business': 3,
            'Advanced programs to reduce carbon emissions across the business': 4
        }).fillna(0)
        program_indicators.append('carbon_reduction_score')
    
    # Governance/oversight features
    if 'EXEC_BODY_ENV_ISSUES' in data.columns:
        data['governance_score'] = data['EXEC_BODY_ENV_ISSUES'].map({
            'No evidence': 0,
            'Corporate Social Responsibility committee': 1,
            'Special committee': 2,
            'C-suite executive(s)': 3,
            'Board of directors': 4
        }).fillna(0)
        program_indicators.append('governance_score')
    
    # Energy audit programs
    if 'CBN_REG_ENERGY_AUDITS' in data.columns:
        data['energy_audit_score'] = data['CBN_REG_ENERGY_AUDITS'].map({
            'No evidence': 0,
            'General statement': 1,
            'Internal audits/assessments': 2,
            'Certifies to an external standard (e.g. ISO 50001)': 3,
            'External audits': 4
        }).fillna(0)
        program_indicators.append('energy_audit_score')
    
    # Create overall program implementation score
    if program_indicators:
        data['program_implementation_score'] = data[program_indicators].sum(axis=1)
        
        # Create program implementation score relative to industry
        industry_avg_score = data.groupby('industry_group')['program_implementation_score'].transform('mean')
        data['program_score_vs_industry'] = data['program_implementation_score'] / (industry_avg_score + 0.001)
        data['program_score_vs_industry'] = data['program_score_vs_industry'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5. Program age feature - how long programs have been in place
    if 'CBN_IMP_PROG_YEAR' in data.columns:
        data['program_age'] = 2024 - data['CBN_IMP_PROG_YEAR']
        data['program_age'] = data['program_age'].clip(0, 30).fillna(0)
    
    # 6. Interaction terms between program features and company characteristics
    # Program score * size interaction
    if 'program_implementation_score' in data.columns and 'SALES_USD_RECENT' in data.columns:
        data['program_score_by_size'] = data['program_implementation_score'] * np.log1p(data['SALES_USD_RECENT'])
        data['program_score_by_size'] = data['program_score_by_size'].fillna(0)
    
    # Target ambition * emissions intensity interaction
    if 'CBN_TARGET_REDUC_PCT' in data.columns and 'CARBON_EMISSIONS_SCOPE_12_INTEN' in data.columns:
        data['target_by_emissions_intensity'] = data['CBN_TARGET_REDUC_PCT'] * data['CARBON_EMISSIONS_SCOPE_12_INTEN']
        data['target_by_emissions_intensity'] = data['target_by_emissions_intensity'].fillna(0)
    
    # 7. Binary indicators for having specific types of programs
    # Has any carbon program
    data['has_carbon_program'] = ((data['program_implementation_score'] > 0) if 'program_implementation_score' in data.columns else 0).astype(int)
    
    # Has ambitious target (>30% reduction)
    data['has_ambitious_target'] = ((data['CBN_TARGET_REDUC_PCT'] > 30) if 'CBN_TARGET_REDUC_PCT' in data.columns else 0).astype(int)
    
    print(f"Created {len(data.columns) - len(model_data.columns)} new program-focused features")
    
    return data


# ===== MODEL PREPARATION =====

def prepare_model_inputs(data, focus_on_programs=True):
    """
    Prepare model inputs with a focus on program features.
    
    Parameters:
    1. data (DataFrame): Input dataset with all features
    2. focus_on_programs (bool): Whether to focus on program-related features
    
    Processing steps:
    1. Select relevant features based on focus
    2. Handle missing values
    3. Encode categorical variables
    4. Split into training and testing sets
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names, cat_indices)
    
    Side Effects:
    - Prints data preparation progress
    """
    print("Preparing model inputs...")
    
    # Identify available feature columns
    program_cols, target_cols = identify_program_columns(data)
    
    # Derive additional program features - use the original columns as reference
    original_cols = set(data.columns) - set([col for col in data.columns 
                                            if any(x in col for x in ['program_', 'target_', 'has_', '_vs_'])])
    program_feature_cols = [col for col in data.columns if col not in original_cols]
    
    print(f"Found {len(program_cols)} program columns, {len(target_cols)} target columns, "
          f"and {len(program_feature_cols)} derived program features")
    
    # Create control feature list
    control_features = [col for col in control_cols if col in data.columns]
    
    # Create program feature list
    program_features = program_cols + target_cols + program_feature_cols
    
    # Remove any with all missing values
    program_features = [col for col in program_features if not data[col].isna().all()]
    
    # Dataset selection based on focus
    if focus_on_programs:
        # Use control variables + program features
        selected_features = control_features + program_features
        print("Using combined control + program features")
    else:
        # Use only program features
        selected_features = program_features
        print("Using only program features")
    
    # Remove target from features if it's there
    if target_col in selected_features:
        selected_features.remove(target_col)
    
    # Drop rows with missing target values
    data_clean = data.dropna(subset=[target_col]).copy()
    print(f"Removed {len(data) - len(data_clean)} rows with missing target values.")
    
    # Create X, y
    X = data_clean[selected_features].copy()
    y = data_clean[target_col].copy()
    
    # Verify no NaN values in target
    print(f"Target has {y.isna().sum()} NaN values after cleaning.")
    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype.name in ['float64', 'int64']:
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna('unknown')
    
    # Convert categorical features to codes
    cat_features = [col for col in X.columns if X[col].dtype.name not in ['float64', 'int64']]
    cat_feature_indices = []
    
    for i, col in enumerate(X.columns):
        if col in cat_features:
            # Convert to categorical first, then to codes
            X[col] = X[col].astype('category').cat.codes.astype('int64')
            cat_feature_indices.append(i)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, cat_feature_indices, selected_features


# ===== MODEL TRAINING =====

def train_program_focused_gbm(X_train, y_train, X_test, y_test, cat_feature_indices):
    """
    Train a GBM model focused on program features.
    
    Parameters:
    1. X_train (DataFrame): Training features
    2. y_train (Series): Training target
    3. X_test (DataFrame): Testing features
    4. y_test (Series): Testing target
    5. cat_feature_indices (list): Indices of categorical features
    
    Processing steps:
    1. Set up LightGBM parameters
    2. Create LightGBM datasets
    3. Train model with early stopping
    4. Evaluate model performance
    
    Returns:
    tuple: (trained_model, evaluation_metrics)
    
    Side Effects:
    - Prints training progress and evaluation metrics
    - May create model checkpoints
    """
    print("Training program-focused GBM model...")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=cat_feature_indices
    )
    
    val_data = lgb.Dataset(
        X_test, 
        label=y_test,
        categorical_feature=cat_feature_indices,
        reference=train_data
    )
    
    # Define model parameters
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
    
    # Train the model
    print(f"Training program-focused GBM model with {len(X_train)} samples")
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
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)[0]
    
    print(f"Program-focused model - RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson:.4f}")
    
    return model, (rmse, r2, pearson), y_pred 

def analyze_feature_importance(model, feature_names, X_test, model_data):
    """
    Analyze and visualize feature importance and program impacts.
    
    Parameters:
    1. model: Trained LightGBM model
    2. feature_names (list): List of feature names
    3. X_test (DataFrame): Test features
    4. model_data (DataFrame): Full dataset for analysis
    
    Processing steps:
    1. Calculate SHAP values
    2. Generate feature importance plots
    3. Analyze program-related features
    4. Visualize SHAP summary plots
    
    Returns:
    None
    
    Side Effects:
    - Creates and saves visualization files
    - Prints analysis results to console
    """
    print("Analyzing feature importance...")
    importance = model.feature_importance(importance_type='gain')
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Print top features
    print("\nTop 15 Features by Importance:")
    for i, row in importance_df.head(15).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.2f}")
    
    # Visualize
    plt.figure(figsize=(12, 10))
    plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title('Top 15 Features by Importance')
    plt.gca().invert_yaxis()  # Put the most important feature at the top
    plt.tight_layout()
    # Create output directories if they don't exist
    output_base = 'output/details/program_focused_analysis'
    png_dir = os.path.join(output_base, 'png')
    pdf_dir = os.path.join(output_base, 'pdf')
    
    for directory in [output_base, png_dir, pdf_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save to both PNG and PDF formats
    plt.savefig(os.path.join(png_dir, 'fig14_program_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, 'fig14_program_feature_importance.pdf'), bbox_inches='tight')
    plt.close()
    
    return importance_df


def analyze_shap_values(model, X_test, feature_names):
    """
    Analyze SHAP values to understand individual feature impacts.
    
    Parameters:
    1. model: Trained LightGBM model
    2. X_test (DataFrame): Test features
    3. feature_names (list): List of feature names
    
    Processing steps:
    1. Calculate SHAP values for test set
    2. Generate summary plot of SHAP values
    3. Analyze top features by mean absolute SHAP value
    
    Returns:
    tuple: (shap_values, expected_value) for further analysis
    
    Side Effects:
    - Creates and saves SHAP summary plot
    - Prints top features by importance
    """
    print("Analyzing SHAP values...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Define output directories (same as in analyze_feature_importance)
    output_base = 'output/details/program_focused_analysis'
    png_dir = os.path.join(output_base, 'png')
    pdf_dir = os.path.join(output_base, 'pdf')
    for directory in [output_base, png_dir, pdf_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Visualize SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(os.path.join(png_dir, 'fig14_program_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, 'fig14_program_shap_summary.pdf'), bbox_inches='tight')
    plt.close()
    
    # Program-specific SHAP values
    program_features = [f for f in feature_names if any(keyword in f for keyword in 
                            ['CBN_PROG', 'CBN_EVIDENCE', 'EXEC_BODY', 'CBN_REG', 'CBN_TARGET',
                             'TARGET_CARBON', 'program_', 'target_'])
    ]
    
    if program_features:
        program_indices = [feature_names.index(f) for f in program_features]
        
        # Visualize program-specific SHAP values
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values[:, program_indices], 
            X_test.iloc[:, program_indices],
            feature_names=[feature_names[i] for i in program_indices],
            show=False
        )
        plt.tight_layout()
        # Save to both PNG and PDF formats
        plt.savefig(os.path.join(png_dir, 'fig14_program_specific_shap.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(pdf_dir, 'fig14_program_specific_shap.pdf'), bbox_inches='tight')
        plt.close()
    
    return shap_values


def compare_program_effectiveness(data, y_pred):
    """
    Compare effectiveness of different sustainability programs.
    
    Parameters:
    1. data (DataFrame): Dataset with program participation and predictions
    2. y_pred (array): Model predictions for each observation
    
    Processing steps:
    1. Calculate average emissions reduction by program
    2. Compare program effectiveness across industries
    3. Generate visualizations of program impacts
    
    Returns:
    DataFrame: Summary statistics of program effectiveness
    
    Side Effects:
    - Creates and saves program comparison visualizations
    - Prints program effectiveness metrics
    """
    print("Comparing program effectiveness...")
    results_df = data.copy()
    results_df['predicted_emission_trend'] = y_pred
    
    # Group analysis - program implementation score
    if 'program_implementation_score' in results_df.columns:
        # Check for enough unique values
        if results_df['program_implementation_score'].nunique() > 3:
            # Create score groups
            try:
                results_df['score_group'] = pd.qcut(
                    results_df['program_implementation_score'].clip(0, 15),
                    q=4, 
                    labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                    duplicates='drop'
                )
            except ValueError:
                # If qcut fails, use manual bins
                score_bins = [0, 2, 5, 10, 20]
                results_df['score_group'] = pd.cut(
                    results_df['program_implementation_score'].clip(0, 20),
                    bins=score_bins, 
                    labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                    include_lowest=True
                )
            
            # Analyze emission trends by score group
            score_analysis = results_df.groupby('score_group')[target_col].agg(['mean', 'count'])
            print("\nEmission Trends by Program Implementation Score:")
            print(score_analysis)
            
            # Visualize
            plt.figure(figsize=(10, 6))
            plt.bar(score_analysis.index, score_analysis['mean'])
            plt.xlabel('Program Implementation Score')
            plt.ylabel(f'Average {target_col}')
            plt.title('Emission Trends by Program Implementation Score')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            # Save to both PNG and PDF formats
            plt.savefig(os.path.join(png_dir, 'fig14_program_score_impact.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(pdf_dir, 'fig14_program_score_impact.pdf'), bbox_inches='tight')
            plt.close()
        else:
            print("\nNot enough unique program implementation scores for grouping analysis")
    
    # Target ambition analysis
    if 'CBN_TARGET_REDUC_PCT' in results_df.columns:
        # Check if we have enough non-zero values
        non_zero = (results_df['CBN_TARGET_REDUC_PCT'] > 0).sum()
        if non_zero > 50:  # Enough data for analysis
            # Create ambition groups
            try:
                results_df['target_group'] = pd.cut(
                    results_df['CBN_TARGET_REDUC_PCT'].clip(0, 100),
                    bins=[0, 1, 10, 30, 100], 
                    labels=['No Target', 'Low Ambition', 'Medium Ambition', 'High Ambition'],
                    include_lowest=True
                )
                
                # Analyze emission trends by target ambition
                target_analysis = results_df.groupby('target_group')[target_col].agg(['mean', 'count'])
                print("\nEmission Trends by Target Ambition:")
                print(target_analysis)
                
                # Visualize
                plt.figure(figsize=(10, 6))
                plt.bar(target_analysis.index, target_analysis['mean'])
                plt.xlabel('Target Ambition Level')
                plt.ylabel(f'Average {target_col}')
                plt.title('Emission Trends by Target Ambition')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                # Save to both PNG and PDF formats
                plt.savefig(os.path.join(png_dir, 'fig14_target_ambition_impact.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(pdf_dir, 'fig14_target_ambition_impact.pdf'), bbox_inches='tight')
                plt.close()
            except:
                print("\nCould not analyze target ambition due to data issues")
        else:
            print(f"\nNot enough companies with targets for ambition analysis (only {non_zero} with targets)")
    
    # Industry-specific program effectiveness
    if 'industry_group' in results_df.columns and 'program_implementation_score' in results_df.columns:
        # Get top 5 industries by company count
        top_industries = results_df['industry_group'].value_counts().head(5).index.tolist()
        industry_data = results_df[results_df['industry_group'].isin(top_industries)]
        
        if len(industry_data) > 100:  # Enough data for analysis
            try:
                # Create industry-score groups - use simple high/low split for better sample sizes
                median_score = industry_data['program_implementation_score'].median()
                industry_data['score_group'] = pd.cut(
                    industry_data['program_implementation_score'],
                    bins=[0, median_score, industry_data['program_implementation_score'].max() + 0.1], 
                    labels=['Low', 'High'],
                    include_lowest=True
                )
                
                # Analyze emission trends by industry and score
                industry_score_analysis = industry_data.groupby(['industry_group', 'score_group'])[target_col].mean().unstack()
                print("\nEmission Trends by Industry and Program Score:")
                print(industry_score_analysis)
                
                # Visualize
                plt.figure(figsize=(12, 8))
                industry_score_analysis.plot(kind='bar', ax=plt.gca())
                plt.xlabel('Industry')
                plt.ylabel(f'Average {target_col}')
                plt.title('Emission Trends by Industry and Program Implementation Score')
                plt.legend(title='Program Score')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                # Save to both PNG and PDF formats
                plt.savefig(os.path.join(png_dir, 'fig14_industry_program_impact.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(pdf_dir, 'fig14_industry_program_impact.pdf'), bbox_inches='tight')
                plt.close()
            except:
                print("\nCould not analyze industry-specific program effectiveness due to data issues")
        else:
            print("\nNot enough data for industry-specific program effectiveness analysis")
    
    return results_df


def run_program_focused_analysis(with_control=True):
    """
    Run the complete program-focused emission reduction analysis.
    
    Parameters:
    1. with_control (bool): Whether to include control variables
    
    Processing steps:
    1. Load and prepare data
    2. Engineer program-focused features
    3. Train GBM model
    4. Analyze feature importance
    5. Generate insights on program effectiveness
    
    Returns:
    dict: Dictionary containing analysis results and model artifacts
    
    Side Effects:
    - Creates output directories if they don't exist
    - Saves visualizations and model artifacts
    - Prints progress and results to console
    """
    print("Starting program-focused analysis...")
    output_dir = 'output/details/program_focused_analysis'
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.getcwd()
    
    try:
        # Load data - do this before changing directory
        print("Loading data...")
        company_emission_df, program1_df, program2_df, target_df = load_data()
        
        # Now change to output directory (only for saving results)
        os.chdir(output_dir)
        
        # Merge datasets
        model_data = prepare_merged_data(company_emission_df, program1_df, program2_df, target_df)
        
        # Engineer program-specific features
        enhanced_data = engineer_program_features(model_data)
        
        # Prepare model inputs
        X_train, X_test, y_train, y_test, cat_indices, feature_names = prepare_model_inputs(
            enhanced_data, focus_on_programs=with_control
        )
        
        # Train the model
        model, metrics, y_pred = train_program_focused_gbm(X_train, y_train, X_test, y_test, cat_indices)
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(model, feature_names, X_test, enhanced_data)
        
        # Generate SHAP analysis
        shap_values = analyze_shap_values(model, X_test, feature_names)
        
        # Compare program effectiveness
        results_df = compare_program_effectiveness(
            enhanced_data.iloc[X_test.index].reset_index(drop=True), 
            y_pred
        )
        
        # Save model and results
        model.save_model('program_focused_model.txt')
        importance_df.to_csv('feature_importance.csv', index=False)
        
        # Create summary report
        with open('analysis_summary.txt', 'w') as f:
            model_type = "Program-focused with control variables" if with_control else "Program-only without controls"
            f.write(f"Program Focused Emissions Prediction Model ({model_type})\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"RMSE: {metrics[0]:.4f}\n")
            f.write(f"R²: {metrics[1]:.4f}\n")
            f.write(f"Pearson Correlation: {metrics[2]:.4f}\n\n")
            
            f.write("Top 10 Features by Importance:\n")
            for i, row in importance_df.head(10).iterrows():
                f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.2f}\n")
            
            f.write("\nProgram Feature Analysis:\n")
            program_features = [
                f for f in feature_names if any(keyword in f for keyword in 
                ['CBN_PROG', 'CBN_EVIDENCE', 'EXEC_BODY', 'CBN_REG', 'CBN_TARGET',
                 'TARGET_CARBON', 'program_', 'target_'])
            ]
            
            # Get program features in importance ranking
            program_ranks = [
                (i+1, row['Feature'], row['Importance']) 
                for i, row in importance_df.iterrows() 
                if row['Feature'] in program_features
            ]
            
            for rank, feature, importance in program_ranks[:10]:  # Top 10 program features
                f.write(f"Rank {rank}: {feature} (Importance: {importance:.2f})\n")
        
        print(f"\nAnalysis complete. Results saved to {output_dir}")
        return model, enhanced_data, importance_df
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    print("=" * 80)
    print("Program and Target Focused Emissions Prediction Model")
    print("=" * 80)
    
    # Run with control variables first
    print("\nRunning analysis WITH control variables (company characteristics)")
    model_with_controls, data, importance = run_program_focused_analysis(with_control=True)
    
    # Then run without control variables (pure program impact)
    print("\nRunning analysis WITHOUT control variables (program features only)")
    model_program_only, _, _ = run_program_focused_analysis(with_control=False) 