"""
Predicts carbon emissions trends using fastai TabularLearner and feature engineering on ESG datasets.

Key analyses performed:
- Loads and merges company emissions, targets, and program datasets
- Engineers features for machine learning
- Builds and trains a fastai TabularLearner model for emissions prediction
- Calculates and visualizes feature importance using Random Forest

Outputs:
- Console output: Progress logs, model evaluation, and top feature importances
- Visualization: Feature importance plot (displayed, not saved)
"""
# ===== IMPORTS =====
import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from fastai.tabular.all import TabularPandas, TabularLearner, Categorify, FillMissing, Normalize, accuracy, range_of, load_learner, RandomSplitter, tabular_learner, rmse

# Step 1: Prepare and merge datasets
print("Preparing datasets for emissions prediction...")

def create_merged_dataset(company_emission_df, target_df, program1_df, program2_df):
    """
    Creates a merged dataset from company emissions, targets, and program data.
    
    Inputs:
    1. company_emission_df (DataFrame): Company emissions data
    2. target_df (DataFrame): Target information
    3. program1_df (DataFrame): Program 1 information
    4. program2_df (DataFrame): Program 2 information
    
    Processing:
    1. Merges all input dataframes on ISSUERID.
    2. Aggregates and encodes categorical columns.
    3. Filters for companies with emissions trend data.
    
    Returns:
    DataFrame: Merged dataset ready for feature engineering
    """
    # Start with company emissions as base
    merged_df = company_emission_df.copy()
    
    # Add target information - use mean values for companies with multiple targets
    targets_agg = target_df.groupby('ISSUERID').agg({
        'CBN_TARGET_REDUC_PCT': 'mean',
        'TARGET_CARBON_PROGRESS_PCT': 'mean',
        'CBN_TARGET_BASE_YEAR': 'mean',
        'CBN_TARGET_IMP_YEAR': 'mean'
    }).reset_index()
    
    merged_df = pd.merge(merged_df, targets_agg, on='ISSUERID', how='left')
    
    # Add program1 information (mitigation strategies)
    merged_df = pd.merge(merged_df, program1_df, on='ISSUERID', how='left')
    
    # For program2, we'll keep it simple and use only the categorical columns
    # First get average implementation year by company
    prog_year_agg = program2_df.groupby('ISSUERID')['CBN_IMP_PROG_YEAR'].mean().reset_index()
    merged_df = pd.merge(merged_df, prog_year_agg, on='ISSUERID', how='left')
    
    # Then create dummy variables for each program category and aggregate
    # We need to get the most common value for each categorical column
    categorical_cols = [
        'CBN_PROG_LOW_CARB_RENEW',
        'CBN_EVIDENCE_TARG_ENERGY_IMPROV',
        'EXEC_BODY_ENV_ISSUES',
        'CBN_REG_ENERGY_AUDITS',
        'CBN_PROG_REDU_CARB_CORE_OP'
    ]
    
    # Get the most common value for each categorical column for each company
    for col in categorical_cols:
        # Check if column exists
        if col not in program2_df.columns:
            print(f"Warning: Column {col} not found in program2_df")
            continue
            
        # Use mode (most frequent value) for categorical columns
        try:
            mode_df = program2_df.groupby('ISSUERID')[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None
            ).reset_index()
            mode_df = mode_df.rename(columns={col: f"{col}_mode"})
            merged_df = pd.merge(merged_df, mode_df, on='ISSUERID', how='left')
            
            # Create dummy variables from the mode
            if f"{col}_mode" in merged_df.columns:
                dummies = pd.get_dummies(merged_df[f"{col}_mode"], prefix=col, dummy_na=True)
                if not dummies.empty:
                    merged_df = pd.concat([merged_df, dummies], axis=1)
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
    
    # Filter to include only companies with emissions trend data (our target variable)
    merged_df = merged_df.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
    
    print(f"Created merged dataset with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
    return merged_df

# Step 2: Feature engineering
def engineer_features(df):
    """
    Engineers features for the emissions prediction model.
    
    Inputs:
    1. df (DataFrame): Merged dataset
    
    Processing:
    1. Creates region, sector, and log-transformed features.
    2. Computes governance and regulatory scores.
    3. Adds program age and region-industry interaction features.
    
    Returns:
    DataFrame: Dataset with engineered features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Create region feature based on country
    region_map = {
        'US': 'North America', 'CA': 'North America', 'MX': 'North America',
        'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'IT': 'Europe', 'ES': 'Europe', 
        'SE': 'Europe', 'CH': 'Europe', 'NL': 'Europe', 'BE': 'Europe', 'DK': 'Europe',
        'FI': 'Europe', 'NO': 'Europe', 'IE': 'Europe', 'AT': 'Europe', 'PT': 'Europe',
        'GR': 'Europe', 'PL': 'Europe', 'CZ': 'Europe', 'HU': 'Europe',
        'CN': 'Asia-Pacific', 'JP': 'Asia-Pacific', 'IN': 'Asia-Pacific',
        'KR': 'Asia-Pacific', 'TW': 'Asia-Pacific', 'HK': 'Asia-Pacific',
        'SG': 'Asia-Pacific', 'ID': 'Asia-Pacific', 'MY': 'Asia-Pacific',
        'TH': 'Asia-Pacific', 'PH': 'Asia-Pacific', 'VN': 'Asia-Pacific', 'AU': 'Asia-Pacific',
        'NZ': 'Asia-Pacific',
        'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
        'CO': 'South America', 'PE': 'South America',
        'ZA': 'Africa', 'EG': 'Africa', 'NG': 'Africa', 'KE': 'Africa',
        'AE': 'Middle East', 'SA': 'Middle East', 'IL': 'Middle East',
        'TR': 'Middle East', 'QA': 'Middle East'
    }
    df_processed['REGION'] = df_processed['ISSUER_CNTRY_DOMICILE'].map(region_map).fillna('Other')
    
    # Create industry groups based on NACE code
    # Extract first 2 digits of NACE code to group by sector
    if 'NACE_CLASS_CODE' in df_processed.columns:
        df_processed['NACE_SECTOR'] = df_processed['NACE_CLASS_CODE'].astype(str).str[:2]
    
    # Create log-transformed features for skewed numerical variables
    skewed_cols = ['CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_12_INTEN', 
                   'MarketCap_USD', 'SALES_USD_RECENT']
    
    for col in skewed_cols:
        if col in df_processed.columns:
            # Add small constant to handle zeros
            df_processed[f'{col}_LOG'] = np.log1p(df_processed[col].fillna(0))
    
    # Create target achievement indicator
    if 'TARGET_CARBON_PROGRESS_PCT' in df_processed.columns:
        df_processed['TARGET_ACHIEVEMENT'] = np.where(
            df_processed['TARGET_CARBON_PROGRESS_PCT'] >= 100, 1, 0)
    
    # Create governance quality score based on exec body
    board_cols = [col for col in df_processed.columns if 'EXEC_BODY_ENV_ISSUES_Board' in col]
    csuite_cols = [col for col in df_processed.columns if 'EXEC_BODY_ENV_ISSUES_C-suite' in col]
    special_cols = [col for col in df_processed.columns if 'EXEC_BODY_ENV_ISSUES_Special' in col]
    csr_cols = [col for col in df_processed.columns if 'EXEC_BODY_ENV_ISSUES_Corporate Social' in col]
    
    df_processed['GOVERNANCE_SCORE'] = 0
    
    for col in board_cols:
        df_processed['GOVERNANCE_SCORE'] += df_processed[col].fillna(0) * 4
    
    for col in csuite_cols:
        df_processed['GOVERNANCE_SCORE'] += df_processed[col].fillna(0) * 3
        
    for col in special_cols:
        df_processed['GOVERNANCE_SCORE'] += df_processed[col].fillna(0) * 2
        
    for col in csr_cols:
        df_processed['GOVERNANCE_SCORE'] += df_processed[col].fillna(0) * 1
    
    # Create regulatory compliance score
    external_cols = [col for col in df_processed.columns if 'CBN_REG_ENERGY_AUDITS_External' in col]
    internal_cols = [col for col in df_processed.columns if 'CBN_REG_ENERGY_AUDITS_Internal' in col]
    iso_cols = [col for col in df_processed.columns if 'CBN_REG_ENERGY_AUDITS_Certifies' in col]
    general_cols = [col for col in df_processed.columns if 'CBN_REG_ENERGY_AUDITS_General' in col]
    
    df_processed['REGULATORY_SCORE'] = 0
    
    for col in external_cols:
        df_processed['REGULATORY_SCORE'] += df_processed[col].fillna(0) * 4
        
    for col in internal_cols:
        df_processed['REGULATORY_SCORE'] += df_processed[col].fillna(0) * 3
        
    for col in iso_cols:
        df_processed['REGULATORY_SCORE'] += df_processed[col].fillna(0) * 2
        
    for col in general_cols:
        df_processed['REGULATORY_SCORE'] += df_processed[col].fillna(0) * 1
    
    # Create program implementation timeline feature
    if 'CBN_IMP_PROG_YEAR' in df_processed.columns:
        df_processed['PROGRAM_AGE'] = 2024 - df_processed['CBN_IMP_PROG_YEAR']
    
    # Create region-industry interaction term
    if 'NACE_SECTOR' in df_processed.columns:
        df_processed['REGION_INDUSTRY'] = df_processed['REGION'] + '_' + df_processed['NACE_SECTOR']
    
    print(f"Engineered features, now have {df_processed.shape[1]} columns")
    return df_processed

# Step 3: Create TabularLearner for emissions prediction
def build_emissions_prediction_model(df):
    """
    Builds and returns a fastai TabularLearner model for emissions prediction.
    
    Inputs:
    1. df (DataFrame): Dataset with engineered features
    
    Processing:
    1. Defines categorical and continuous variables.
    2. Creates TabularPandas and DataLoaders objects.
    3. Initializes TabularLearner with RMSE metric.
    
    Returns:
    tuple: (learn, dls)
        - learn (TabularLearner): Untrained emissions prediction model
        - dls (DataLoaders): DataLoaders object for training/validation
    """
    # Define dependent variable (what we're predicting)
    dep_var = 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'
    
    # Define categorical variables
    cat_vars = [
        'ISSUER_CNTRY_DOMICILE', 
        'NACE_SECTOR',
        'REGION',
        'REGION_INDUSTRY'
    ]
    
    # Define continuous variables
    cont_vars = [
        'CARBON_EMISSIONS_SCOPE_12_LOG',
        'CARBON_EMISSIONS_SCOPE_12_INTEN_LOG',
        'MarketCap_USD_LOG', 
        'GOVERNANCE_SCORE',
        'REGULATORY_SCORE',
        'PROGRAM_AGE',
        'CBN_TARGET_REDUC_PCT'
    ]
    
    # Filter to only include vars that exist in the dataframe
    cat_vars = [var for var in cat_vars if var in df.columns]
    cont_vars = [var for var in cont_vars if var in df.columns]
    
    print(f"Using {len(cat_vars)} categorical variables and {len(cont_vars)} continuous variables")
    
    # Create TabularPandas object
    splits = RandomSplitter(valid_pct=0.2)(range_of(df))
    to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize],
                      cat_names=cat_vars, cont_names=cont_vars,
                      y_names=dep_var, splits=splits)
    
    # Create DataLoaders - specify use of CPU to avoid CUDA issues
    dls = to.dataloaders(bs=64, device='cpu')
    
    # Create TabularLearner with rmse metric
    learn = tabular_learner(dls, metrics=rmse, 
                           layers=[200, 100], 
                           config=dict(use_bn=True, bn_final=True, bn_cont=True))
    
    print("Building TabularLearner model for emissions prediction...")
    return learn, dls

# Train the model
def train_model(learn, epochs=5, lr=1e-3):
    """
    Trains the TabularLearner model for emissions prediction.
    
    Inputs:
    1. learn (TabularLearner): Model to train
    2. epochs (int, optional): Number of training epochs (default: 5)
    3. lr (float, optional): Learning rate (default: 1e-3)
    
    Processing:
    1. Fits the model using one-cycle policy.
    
    Returns:
    TabularLearner: Trained model
    """
    print("Training the emissions prediction model...")
    learn.fit_one_cycle(epochs, lr)
    return learn

# Calculate feature importance using Random Forest
def calculate_rf_feature_importance(df, cat_vars, cont_vars, target_var='CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'):
    """
    Calculates feature importance using a Random Forest regressor.
    
    Inputs:
    1. df (DataFrame): Data for feature importance calculation
    2. cat_vars (list): List of categorical variable names
    3. cont_vars (list): List of continuous variable names
    4. target_var (str, optional): Name of target variable (default: 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR')
    
    Processing:
    1. Encodes categorical variables as dummies.
    2. Trains a Random Forest regressor.
    3. Extracts and sorts feature importances.
    
    Returns:
    dict: Feature importances (feature name -> importance score)
    """
    print("Calculating feature importance using Random Forest...")
    
    # Create a copy of the dataframe
    df_rf = df.copy()
    
    # Only use categorical variables that exist in the dataframe
    cat_vars = [col for col in cat_vars if col in df_rf.columns]
    
    # Create dummy variables for categorical variables
    df_rf = pd.get_dummies(df_rf, columns=cat_vars, drop_first=False)
    
    # Select features and target
    X = df_rf.drop(columns=[target_var])
    y = df_rf[target_var]

    # Drop any non-numeric columns (e.g., IDs)
    X = X.select_dtypes(include=[np.number])
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_names = X.columns
    importances = {name: imp for name, imp in zip(feature_names, rf.feature_importances_)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances

# Plot feature importance
def plot_feature_importance(importances, top_n=10):
    """
    Plots the top N most important features as a horizontal bar chart.
    
    Inputs:
    1. importances (dict): Feature importances (feature name -> importance score)
    2. top_n (int, optional): Number of top features to plot (default: 10)
    
    Processing:
    1. Selects top N features.
    2. Plots a horizontal bar chart using matplotlib.
    
    Returns:
    None (displays the plot)
    """
    plt.figure(figsize=(10, 6))
    
    # Get top N features
    top_features = list(importances.keys())[:top_n]
    top_importances = [importances[feature] for feature in top_features]
    
    # Create horizontal bar chart
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    # Save the plot to a file
    output_dir = 'output_graphs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fig17_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Main function to run the emissions prediction pipeline
def run_emissions_prediction():
    """
    Runs the full emissions prediction pipeline from data loading to feature importance visualization.
    
    Inputs:
    None (expects dataframes to be loaded or defined in the script)
    
    Processing:
    1. Merges and engineers features from input data.
    2. Builds and trains the emissions prediction model.
    3. Evaluates the model and prints results.
    4. Calculates and displays feature importances.
    
    Returns:
    tuple: (learn, dls, importances)
        - learn (TabularLearner): Trained model
        - dls (DataLoaders): DataLoaders object
        - importances (dict): Feature importances
    """
    # Step 1: Load data
    # Replace with your actual data loading code
    # company_emission_df = pd.read_csv('path/to/emissions.csv')
    # target_df = pd.read_csv('path/to/targets.csv')
    # program1_df = pd.read_csv('path/to/program1.csv')
    # program2_df = pd.read_csv('path/to/program2.csv')
    
    # Step 2: Create merged dataset
    merged_df = create_merged_dataset(company_emission_df, target_df, program1_df, program2_df)
    
    # Step 3: Engineer features
    emissions_prediction_df = engineer_features(merged_df)
    
    # Step 4: Build and train model
    learn, dls = build_emissions_prediction_model(emissions_prediction_df)
    learn = train_model(learn)
    
    # Step 5: Show model results
    print("Model evaluation:")
    learn.show_results()
    
    # Step 6: Calculate feature importance
    cat_vars = dls.cat_names
    cont_vars = dls.cont_names
    importances = calculate_rf_feature_importance(emissions_prediction_df, cat_vars, cont_vars)
    
    # Step 7: Show top 10 features
    print("\nTop 10 most important features:")
    for i, (feature, imp) in enumerate(list(importances.items())[:10]):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # Step 8: Plot feature importance
    plot_feature_importance(importances)
    
    return learn, dls, importances

if __name__ == "__main__":
    # Load required dataframes from CSVs
    company_emission_df = pd.read_csv("data/company_emissions_merged.csv")
    target_df = pd.read_csv("data/Reduktionsziele Results - 20241212 15_49_29.csv")
    program1_df = pd.read_csv("data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv")
    program2_df = pd.read_csv("data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv")
    run_emissions_prediction() 