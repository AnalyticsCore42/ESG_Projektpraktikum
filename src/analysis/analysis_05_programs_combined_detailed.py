"""
Combined analysis of carbon reduction programs from both programs1 and programs2 datasets.

Input DataFrames:
1. programs1_df:
2. programs2_df:
3. company_emissions_df:


Key Analyses Performed:
1. Dataset overlap analysis between programs1 and programs2
2. Combined program metrics and strategy relationships
3. Emissions impact analysis of combined programs
4. Implementation patterns across company segments

Outputs:
- Console output with detailed analysis results
- Visualizations of program relationships and impacts
- Markdown report with key findings and statistics
"""

# ===== IMPORTS =====
import os
import re
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils.general_utils import (
    categorize_size,
    categorize_region,
    categorize_industry,
    map_distribution_category,
    map_raw_materials_category,
    map_manufacturing_category,
    map_transport_category,
    map_capture_category,
    get_mitigation_score_maps,
)

warnings.filterwarnings('ignore')

# ===== FILE PATHS =====
print("\n--- DATA CONFIGURATION ---")
print("=" * 80)

# Update these paths if your data is stored elsewhere relative to the project root
COMPANY_DATA_PATH = "data/Unternehmensdaten Results - 20241212 15_41_41.csv"
EMISSIONS_DATA_PATH = "data/Treibhausgasemissionen Results - 20241212 15_44_03.csv"
TARGETS_DATA_PATH = "data/Reduktionsziele Results - 20241212 15_49_29.csv"
PROGRAMS1_DATA_PATH = "data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
PROGRAMS2_DATA_PATH = "data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
COMPANY_EMISSIONS_PATH = "data/company_emissions_merged.csv"

# ===== DATA LOADING =====
def load_data():
    """
    Load and prepare all necessary dataframes.
    
    Processing Steps:
    1. Load programs1 and programs2 data from CSV files
    2. Load company emissions data if available
    3. Print dataset information
    """
    try:
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        programs2_df = pd.read_csv(PROGRAMS2_DATA_PATH).dropna(axis=1, how='all')
        
        try:
            company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        except FileNotFoundError:
            print(f"Warning: Company emissions data not found at {COMPANY_EMISSIONS_PATH}")
            company_emissions_df = None
        
        print(f"\nPROGRAMS1 DATAFRAME: shape={programs1_df.shape}, unique companies={programs1_df['ISSUERID'].nunique():,}")
        print(f"PROGRAMS2 DATAFRAME: shape={programs2_df.shape}, unique companies={programs2_df['ISSUERID'].nunique():,}")
        if company_emissions_df is not None:
            print(f"COMPANY_EMISSIONS DATAFRAME: shape={company_emissions_df.shape}, unique companies={company_emissions_df['ISSUERID'].nunique():,}")
        
        return programs1_df, programs2_df, company_emissions_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are present at the specified paths:")
        print(f"- {PROGRAMS1_DATA_PATH}")
        print(f"- {PROGRAMS2_DATA_PATH}")
        raise

# ===== ANALYSIS FUNCTIONS =====
def analyze_datasets_overlap(programs1_df, programs2_df):
    """
    Analyze company overlap between the two programs datasets.
    
    Parameters:
    1. programs1_df: First program dataset
    2. programs2_df: Second program dataset
    
    Processing Steps:
    1. Calculate unique companies in each dataset
    2. Determine overlap and unique companies
    3. Analyze geographical distribution of overlap
    """
    print("\n--- DATASETS OVERLAP ANALYSIS ---")
    print("=" * 80)
    
    programs1_companies = set(programs1_df['ISSUERID'].unique())
    programs2_companies = set(programs2_df['ISSUERID'].unique())
    
    overlap_companies = programs1_companies.intersection(programs2_companies)
    programs1_only = programs1_companies - programs2_companies
    programs2_only = programs2_companies - programs1_companies
    
    total_unique_companies = len(programs1_companies.union(programs2_companies))
    overlap_pct = len(overlap_companies) / total_unique_companies * 100
    programs1_only_pct = len(programs1_only) / total_unique_companies * 100
    programs2_only_pct = len(programs2_only) / total_unique_companies * 100
    
    print(f"Total unique companies across both datasets: {total_unique_companies:,}")
    print(f"Companies in both datasets: {len(overlap_companies):,} ({overlap_pct:.2f}%)")
    print(f"Companies only in programs1_df: {len(programs1_only):,} ({programs1_only_pct:.2f}%)")
    print(f"Companies only in programs2_df: {len(programs2_only):,} ({programs2_only_pct:.2f}%)")
    
    if 'ISSUER_CNTRY_DOMICILE' in programs1_df.columns and 'ISSUER_CNTRY_DOMICILE' in programs2_df.columns:
        programs1_countries = programs1_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE']].drop_duplicates()
        programs2_countries = programs2_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE']].drop_duplicates()
        
        overlap_countries = programs1_countries[programs1_countries['ISSUERID'].isin(overlap_companies)]
        overlap_country_counts = overlap_countries['ISSUER_CNTRY_DOMICILE'].value_counts()
        
        print("\nTop 10 countries in the overlap:")
        for country, count in overlap_country_counts.head(10).items():
            percentage = count / len(overlap_companies) * 100
            print(f"  - {country}: {count:,} companies ({percentage:.2f}%)")
    
    return overlap_companies, programs1_only, programs2_only

def prepare_mitigation_scores(programs1_df):
    """Calculate mitigation scores by strategy and overall combined score."""
    mitigation_columns = [
        'CBN_GHG_MITIG_DISTRIBUTION', 
        'CBN_GHG_MITIG_RAW_MAT', 
        'CBN_GHG_MITIG_MFG', 
        'CBN_GHG_MITIG_TRANSPORT', 
        'CBN_GHG_MITIG_CAPTURE'
    ]
    
    mitigation_score_maps = get_mitigation_score_maps()
    df = programs1_df.copy()
    
    for col in mitigation_columns:
        score_map = mitigation_score_maps[col]
        df[f'{col}_Score'] = df[col].map(lambda x: score_map.get(x, 0))
    
    score_columns = [f'{col}_Score' for col in mitigation_columns]
    company_scores = df.groupby('ISSUERID')[score_columns].mean()
    company_scores['Combined_Mitigation_Score'] = company_scores.mean(axis=1)
    company_scores = company_scores.reset_index()
    
    return company_scores

def prepare_program2_metrics(programs2_df):
    """
    Calculate program metrics including counts and category distributions.
    
    Parameters:
    1. programs2_df
    
    Processing Steps:
    1. Group data by ISSUERID
    2. Count programs per company
    3. Calculate average implementation year if available
    
    Returns:
        DataFrame with one row per company and columns for each metric
    """
    program_counts = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    
    if 'CBN_IMP_PROG_YEAR' in programs2_df.columns:
        year_df = programs2_df.dropna(subset=['CBN_IMP_PROG_YEAR'])
        avg_impl_year = year_df.groupby('ISSUERID')['CBN_IMP_PROG_YEAR'].mean().reset_index(name='avg_implementation_year')
        program_counts = pd.merge(program_counts, avg_impl_year, on='ISSUERID', how='left')
    
    category_column = 'CBN_PROG_LOW_CARB_RENEW'
    
    if category_column in programs2_df.columns:
        category_df = programs2_df.dropna(subset=[category_column])
        category_counts = pd.crosstab(category_df['ISSUERID'], category_df[category_column])
        category_pcts = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
        category_pcts = category_pcts.add_prefix('pct_')
        category_pcts = category_pcts.reset_index()
        program_metrics = pd.merge(program_counts, category_pcts, on='ISSUERID', how='left')
    else:
        program_metrics = program_counts
    
    return program_metrics

def analyze_program_strategy_relationships(mitigation_scores, program_metrics):
    """
    Analyze how mitigation strategies correlate with program implementations.
    
    Parameters:
    1. mitigation_scores: DataFrame with mitigation strategy scores
    2. program_metrics: Program implementation metrics from prepare_program2_metrics()

    
    Processing Steps:
    1. Merge strategy scores with program metrics
    2. Calculate correlation between combined score and program count
    3. Calculate correlations for individual strategies
    4. Print significant relationships (p < 0.05)
    
    Returns:
        DataFrame with correlation results and significance levels
    """
    print("\n--- MITIGATION STRATEGY vs PROGRAM IMPLEMENTATION ANALYSIS ---")
    print("=" * 80)
    
    combined_df = pd.merge(mitigation_scores, program_metrics, on='ISSUERID', how='inner')
    
    print(f"Companies with both mitigation strategy data and program implementation data: {len(combined_df)}")
    
    # 1. Correlation between mitigation score and program count
    print("\n1. MITIGATION SCORE vs PROGRAM COUNT")
    print("-" * 80)
    
    corr = combined_df['Combined_Mitigation_Score'].corr(combined_df['program_count'])
    print(f"Correlation between combined mitigation score and program count: {corr:.3f}")
    
    mitigation_score_cols = [col for col in combined_df.columns if col.endswith('_Score') and col != 'Combined_Mitigation_Score']
    
    print("\nCorrelations between individual mitigation strategies and program count:")
    for col in mitigation_score_cols:
        strategy_name = col.replace('CBN_GHG_MITIG_', '').replace('_Score', '').title()
        corr = combined_df[col].corr(combined_df['program_count'])
        print(f"  - {strategy_name}: {corr:.3f}")
    
    # 2. Mitigation strategies vs program categories
    print("\n2. MITIGATION STRATEGIES vs PROGRAM CATEGORIES")
    print("-" * 80)
    
    # Get program category percentage columns
    category_pct_cols = [col for col in combined_df.columns if col.startswith('pct_')]
    
    # Calculate correlations between mitigation scores and category percentages
    strategy_category_corrs = []
    for strategy_col in mitigation_score_cols:
        strategy_name = strategy_col.replace('CBN_GHG_MITIG_', '').replace('_Score', '').title()
        
        for category_col in category_pct_cols:
            category_name = category_col.replace('pct_', '')
            corr = combined_df[strategy_col].corr(combined_df[category_col])
            strategy_category_corrs.append((strategy_name, category_name, corr))
    
    # Sort by absolute correlation strength
    strategy_category_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("Top 10 strongest correlations between mitigation strategies and program categories:")
    for strategy, category, corr in strategy_category_corrs[:10]:
        print(f"  - {strategy} vs {category}: {corr:.3f}")
    
    # 3. Implementation timeline vs mitigation score
    if 'avg_implementation_year' in combined_df.columns:
        print("\n3. IMPLEMENTATION TIMELINE vs MITIGATION SCORE")
        print("-" * 80)
        
        timeline_score_corr = combined_df['avg_implementation_year'].corr(combined_df['Combined_Mitigation_Score'])
        print(f"Correlation between avg implementation year and combined mitigation score: {timeline_score_corr:.3f}")
        
        print("\nCorrelations between avg implementation year and individual mitigation strategies:")
        for col in mitigation_score_cols:
            strategy_name = col.replace('CBN_GHG_MITIG_', '').replace('_Score', '').title()
            corr = combined_df['avg_implementation_year'].corr(combined_df[col])
            print(f"  - {strategy_name}: {corr:.3f}")
    
    return combined_df

def analyze_emissions_relationships(combined_df, company_emissions_df):
    """
    Analyze the relationship between program implementation and emissions.
    
    Parameters:
    1. combined_df: Combined dataframe with program and strategy data
    2. company_emissions_df: Company emissions data
 
    Processing Steps:
    1. Merge program data with emissions data
    2. Calculate correlations between mitigation scores and emissions metrics
    3. Analyze relationships between program counts and emissions
    4. Print significant relationships (p < 0.05)
    
    Returns:
        DataFrame with analysis results including correlations and effect sizes
    """
    print("\n--- EMISSIONS RELATIONSHIPS ANALYSIS ---")
    print("=" * 80)
    
    if company_emissions_df is None:
        print("No company emissions data provided. Skipping emissions analysis.")
        return None
        
    # Create a proper copy of the input DataFrame to avoid SettingWithCopyWarning
    analysis_df = combined_df.copy()
    
    # Get key emissions metrics - use numeric columns for analysis
    emissions_metrics = company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'CARBON_EMISSIONS_SCOPE_12_INTEN', 'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_2', 'CARBON_EMISSIONS_SCOPE_3']].drop_duplicates()
    
    # Merge with emissions data using the copy
    analysis_df = analysis_df.merge(
        emissions_metrics,
        on='ISSUERID',
        how='inner'
    )
    
    # Create a new DataFrame for the final results to avoid chained assignment
    result_df = analysis_df.copy()
    
    print(f"Companies with complete data (strategies, programs, and emissions): {len(result_df)}")
    
    # 1. Emissions vs Combined Score and Program Count
    print("\n1. EMISSIONS vs STRATEGIES AND PROGRAMS")
    print("-" * 80)
    
    # Calculate correlations with emissions intensity trend
    print("\nCorrelations with emissions intensity trend (3Y CAGR):")
    for col in ['Combined_Mitigation_Score', 'program_count'] + [c for c in result_df.columns if c.startswith('CBN_GHG_MITIG_') and c.endswith('_Score')]:
        if col in result_df.columns and result_df[col].nunique() > 1:  # Ensure column exists and has variance
            corr = result_df[col].corr(result_df['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
            print(f"  - {col}: {corr:.3f}")
    
    # 2. Analyze combinations of mitigation strategies and program counts
    print("\n2. COMBINED EFFECT OF MITIGATION STRATEGIES AND PROGRAMS")
    print("-" * 80)
    
    # Function to create levels robustly
    def create_level_categories(series, labels=['Low', 'Medium-Low', 'Medium-High', 'High']):
        # Get unique values to avoid duplicate bin edges
        unique_values = sorted(series.unique())
        
        if len(unique_values) <= len(labels):
            # If we have fewer unique values than desired categories, use value_counts approach
            value_counts = series.value_counts().sort_index()
            cumulative_count = value_counts.cumsum()
            total = cumulative_count.iloc[-1]
            
            # Find values that best split the data into roughly equal groups
            bins = [float('-inf')]
            for i in range(1, len(labels)):
                target_count = i * total / len(labels)
                idx = (cumulative_count - target_count).abs().argsort()[0]
                bins.append(cumulative_count.index[idx])
            bins.append(float('inf'))
        else:
            # Use percentiles for binning when we have enough unique values
            percentiles = [0, 25, 50, 75, 100]
            bins = [float('-inf')]
            bins.extend(np.percentile(unique_values, percentiles[1:-1]))
            bins.append(float('inf'))
        
        # Create categories using the derived bins
        return pd.cut(series, bins=bins, labels=labels)
    
    # Apply the robust categorization
    result_df['Mitigation_Level'] = create_level_categories(result_df['Combined_Mitigation_Score'])
    result_df['Program_Level'] = create_level_categories(result_df['program_count'])
    
    # Calculate average emission trend for each combination
    trend_by_group = result_df.groupby(['Mitigation_Level', 'Program_Level'])['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].agg(['mean', 'count']).reset_index()
    
    # Print results as a structured table
    print("Average emission trends by mitigation strategy level and program implementation level:")
    for mitigation_level in ['Low', 'Medium-Low', 'Medium-High', 'High']:
        print(f"\n  Mitigation Level: {mitigation_level}")
        for program_level in ['Low', 'Medium-Low', 'Medium-High', 'High']:
            row = trend_by_group[(trend_by_group['Mitigation_Level'] == mitigation_level) & 
                                (trend_by_group['Program_Level'] == program_level)]
            if not row.empty:
                avg_trend = row['mean'].values[0]
                count = row['count'].values[0]
                print(f"    - Program Level {program_level}: {avg_trend:.2f}% trend ({count:,} companies)")
    
    # 3. Best performing combinations
    print("\n3. BEST PERFORMING COMBINATIONS")
    print("-" * 80)
    
    # Find best performing combinations (most negative CAGR = highest reduction)
    best_combinations = result_df.groupby(['Mitigation_Level', 'Program_Level'])['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'] \
        .agg(['mean', 'count']).reset_index() \
        .sort_values('mean')
    
    print("\nBest performing combinations (highest emissions reductions):")
    for _, row in best_combinations.head(5).iterrows():
        print(f"  - {row['Mitigation_Level']} mitigation + {row['Program_Level']} programs: "
              f"{row['mean']:.2f}% trend ({row['count']} companies)")
    # 4. Industry-specific analysis
    if 'NACE_CLASS_DESCRIPTION' in result_df.columns:
        print("\n4. INDUSTRY-SPECIFIC ANALYSIS")
        print("-" * 80)
        
        # Categorize industries
        result_df['Industry_Group'] = result_df['NACE_CLASS_DESCRIPTION'].apply(categorize_industry)
        
        # Calculate average emissions trend by industry
        industry_trends = result_df.groupby('Industry_Group')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'] \
            .agg(['mean', 'count']).reset_index() \
            .sort_values('mean')
        
        print("\nIndustries with strongest emissions reductions:")
        for _, row in industry_trends.head(5).iterrows():
            print(f"  - {row['Industry_Group']}: {row['mean']:.2f}% trend ({row['count']} companies)")
    
    # 5. Regional analysis if region data is available
    if 'ISSUER_CNTRY_DOMICILE' in result_df.columns:
        print("\n5. REGIONAL ANALYSIS")
        print("-" * 80)
        
        # Categorize countries into regions
        result_df['Region'] = result_df['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
        
        # Analyze by region
        region_trends = result_df.groupby('Region')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'] \
            .agg(['mean', 'count']).reset_index() \
            .sort_values('mean')
        
        print("\nAverage emissions trend by region:")
        for _, row in region_trends.iterrows():
            print(f"  - {row['Region']}: {row['mean']:.2f}% trend ({row['count']} companies)")
    
    # 6. Calculate metrics by region and industry
    metrics = []
    
    if 'Region' in result_df.columns:
        region_metrics = result_df.groupby('Region').agg({
            'Combined_Mitigation_Score': 'mean',
            'program_count': 'mean',
            'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean',
            'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
            'ISSUERID': 'nunique'
        }).reset_index()
        region_metrics = region_metrics.rename(columns={'ISSUERID': 'company_count'})
        region_metrics['category'] = 'Region'
        region_metrics['name'] = region_metrics['Region']
        metrics.append(region_metrics)
    
    if 'Industry_Group' in result_df.columns:
        industry_metrics = result_df.groupby('Industry_Group').agg({
            'Combined_Mitigation_Score': 'mean',
            'program_count': 'mean',
            'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean',
            'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
            'ISSUERID': 'nunique'
        }).reset_index()
        industry_metrics = industry_metrics.rename(columns={'ISSUERID': 'company_count'})
        industry_metrics['category'] = 'Industry'
        industry_metrics['name'] = industry_metrics['Industry_Group']
        metrics.append(industry_metrics)
    
    # Combine and print all metrics
    if metrics:
        all_metrics = pd.concat(metrics, ignore_index=True)
        
        print("\nSUMMARY METRICS BY CATEGORY")
        print("-" * 80)
        
        for _, row in all_metrics.iterrows():
            print(f"\n{row['category']}: {row['name']}")
            print(f"  - Companies: {row['company_count']}")
            print(f"  - Avg Mitigation Score: {row['Combined_Mitigation_Score']:.2f}")
            print(f"  - Avg Program Count: {row['program_count']:.2f}")
            print(f"  - Avg Emission Trend: {row['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']:.2f}%")
    
    return result_df

def analyze_combined_programs(programs1_df, programs2_df, company_emissions_df):
    """
    Run comprehensive combined analysis on both program datasets.
    
    Parameters:
    1. programs1_df: First program dataset with mitigation strategies
    2. programs2_df: Second program dataset with implementation details
    3. company_emissions_df: Company emissions and financial data
    
    Processing Steps:
    1. Analyze dataset overlap between programs1 and programs2
    2. Prepare program metrics from programs2
    3. Calculate mitigation strategy scores from programs1
    4. Analyze relationships between strategies and implementations
    5. Analyze emissions impacts of programs
    6. Generate combined insights and visualizations
    
    Returns:
        DataFrame with combined analysis results for each company
    """
    print("\n====== COMBINED PROGRAMS ANALYSIS ======")
    print("=" * 80)
    
    # 1. Analyze dataset overlap
    overlap_companies, programs1_only, programs2_only = analyze_datasets_overlap(programs1_df, programs2_df)
    
    # 2. Prepare metrics for each dataset
    mitigation_scores = prepare_mitigation_scores(programs1_df)
    program_metrics = prepare_program2_metrics(programs2_df)
    
    # 3. Analyze relationships between strategies and programs
    combined_df = analyze_program_strategy_relationships(mitigation_scores, program_metrics)
    
    # 4. Analyze relationships with emissions data
    if company_emissions_df is not None:
        analysis_df = analyze_emissions_relationships(combined_df, company_emissions_df)
    
    print("\n====== ANALYSIS COMPLETE ======")
    
    return combined_df

def save_results_to_file(programs1_df, programs2_df, company_emissions_df):
    """
    Save analysis results to a markdown file.
    
    Parameters:
    1. programs1_df: First program dataset with analysis results
    2. programs2_df: Second program dataset with analysis results
    3. company_emissions_df: Company emissions data used in analysis
    
    Processing Steps:
    1. Generate markdown content with combined analysis results
    2. Create reports directory if it doesn't exist
    3. Save to a timestamped file in the reports directory
    4. Include visualizations and key findings
    
    Returns:
        str: Path to the saved markdown file
    """
    original_stdout = sys.stdout
    
    results_dir = "output/reports"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/programs_combined_detailed.md", 'w') as f:
        sys.stdout = f
        
        # Run the analysis
        analyze_combined_programs(programs1_df, programs2_df, company_emissions_df)
        
        # Reset stdout
        sys.stdout = original_stdout
    
    print(f"Analysis results saved to {results_dir}/programs_combined_detailed.md")

if __name__ == "__main__":
    import sys
    
    # Load the data
    programs1_df, programs2_df, company_emissions_df = load_data()
    
    # Run the comprehensive analysis and print to console
    analyze_combined_programs(programs1_df, programs2_df, company_emissions_df)
    
    # Also save to file
    save_results_to_file(programs1_df, programs2_df, company_emissions_df) 