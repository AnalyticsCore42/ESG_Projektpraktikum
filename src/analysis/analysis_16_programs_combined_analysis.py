"""
Performs descriptive and comparative analysis of emissions reduction programs and targets using ESG data.

Key analyses performed:
- Loads and merges company, emissions, target, and program datasets
- Analyzes distribution and types of reduction programs
- Examines sectoral and geographic patterns in program adoption
- Evaluates mitigation strategy implementation and quality
- Assesses relationships between programs, targets, and emissions trends

Outputs:
- Console output: Descriptive statistics, distributions, and analysis summaries
- File: Merged company and emissions data saved as data/company_emissions_merged.csv
"""

# ===== IMPORTS =====
import os
import sys
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from src.utils.analysis_utils import *
from src.utils.data_utils import *
from src.utils.visualization_utils import *
import src.config as config

warnings.filterwarnings('ignore')

# ===== Data Loading =====
# NOTE: Update these paths if your data is stored elsewhere relative to the project root.
COMPANY_DATA_PATH = "data/Unternehmensdaten Results - 20241212 15_41_41.csv"
EMISSIONS_DATA_PATH = "data/Treibhausgasemissionen Results - 20241212 15_44_03.csv"
TARGETS_DATA_PATH = "data/Reduktionsziele Results - 20241212 15_49_29.csv"
PROGRAMS1_DATA_PATH = "data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
PROGRAMS2_DATA_PATH = "data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
COMPANY_EMISSIONS_PATH = "data/company_emissions_merged.csv"

def load_data():
    """
    Loads and prepares all required ESG-related dataframes for analysis.
    
    Inputs:
    None
    
    Processing:
    1. Loads company, emissions, targets, and program CSV files as DataFrames.
    2. Merges company and emissions data, resolving column suffixes.
    3. Saves merged data to CSV if not already present.
    4. Prints dataset shapes and unique company counts.
    
    Returns:
    tuple: (company_df, emissions_df, targets_df, programs1_df, programs2_df, company_emissions_df)
        - company_df (DataFrame): Company-level metadata
        - emissions_df (DataFrame): Emissions data
        - targets_df (DataFrame): Reduction targets data
        - programs1_df (DataFrame): First reduction programs dataset
        - programs2_df (DataFrame): Second reduction programs dataset
        - company_emissions_df (DataFrame): Merged company and emissions data
    """
    try:
        # Load the base dataframes
        company_df = pd.read_csv(COMPANY_DATA_PATH).dropna(axis=1, how='all')
        emissions_df = pd.read_csv(EMISSIONS_DATA_PATH).dropna(axis=1, how='all')
        targets_df = pd.read_csv(TARGETS_DATA_PATH).dropna(axis=1, how='all')
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        programs2_df = pd.read_csv(PROGRAMS2_DATA_PATH).dropna(axis=1, how='all')

        # Check if the merged company_emissions CSV already exists
        if os.path.exists(COMPANY_EMISSIONS_PATH):
            print("Loading pre-merged company_emissions_df from CSV...")
            company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        else:
            print("Creating and saving merged company_emissions_df...")
            # Merge company and emissions data, resolving suffixes
            company_emissions_df = pd.merge(company_df, emissions_df, on="ISSUERID", how="left", suffixes=('_company', '_emissions'))

            x_columns = [col for col in company_emissions_df.columns if col.endswith('_company')]
            y_columns = [col for col in company_emissions_df.columns if col.endswith('_emissions')]
            x_to_original = {col: col[:-8] for col in x_columns}  # _company = 8 chars
            y_to_original = {col: col[:-10] for col in y_columns}  # _emissions = 10 chars

            for original_name in set(x_to_original.values()) & set(y_to_original.values()):
                x_col = f"{original_name}_company"
                y_col = f"{original_name}_emissions"

                if x_col not in company_emissions_df.columns or y_col not in company_emissions_df.columns:
                    continue

                x_null_count = company_emissions_df[x_col].isna().sum()
                y_null_count = company_emissions_df[y_col].isna().sum()

                # Keep the one with fewer nulls
                if y_null_count < x_null_count:
                    company_emissions_df[original_name] = company_emissions_df[y_col]
                else:
                    company_emissions_df[original_name] = company_emissions_df[x_col]

                # Drop the suffixed columns
                company_emissions_df = company_emissions_df.drop([x_col, y_col], axis=1)

            # Rename remaining suffixed columns
            company_emissions_df.columns = [col[:-8] if col.endswith('_company') else col for col in company_emissions_df.columns]
            company_emissions_df.columns = [col[:-10] if col.endswith('_emissions') else col for col in company_emissions_df.columns]
            
            # Save the merged dataframe to CSV
            company_emissions_df.to_csv(COMPANY_EMISSIONS_PATH, index=False)
            print(f"Merged dataframe saved to {COMPANY_EMISSIONS_PATH}")

        # Output dataset information
        print(f"Loaded company_df:           shape={company_df.shape}, unique_ISSUERIDs={company_df['ISSUERID'].nunique():,}")
        print(f"Loaded emissions_df:         shape={emissions_df.shape}, unique_ISSUERIDs={emissions_df['ISSUERID'].nunique():,}")
        print(f"Merged company_emissions_df: shape={company_emissions_df.shape}, unique_ISSUERIDs={company_emissions_df['ISSUERID'].nunique():,}")
        print(f"Loaded targets_df:           shape={targets_df.shape}, unique_ISSUERIDs= {targets_df['ISSUERID'].nunique():,}")
        print(f"Loaded programs1_df:         shape= {programs1_df.shape}, unique_ISSUERIDs= {programs1_df['ISSUERID'].nunique():,}")
        print(f"Loaded programs2_df:         shape={programs2_df.shape}, unique_ISSUERIDs={programs2_df['ISSUERID'].nunique():,}")

        return company_df, emissions_df, targets_df, programs1_df, programs2_df, company_emissions_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are present at the specified paths:")
        print(f"- {COMPANY_DATA_PATH}")
        print(f"- {EMISSIONS_DATA_PATH}")
        print(f"- {TARGETS_DATA_PATH}")
        print(f"- {PROGRAMS1_DATA_PATH}")
        print(f"- {PROGRAMS2_DATA_PATH}")
        raise

def analyze_basic_programs(programs1_df, programs2_df):
    """
    Performs basic descriptive analysis of reduction programs data.
    
    Inputs:
    1. programs1_df (DataFrame): First reduction programs dataset
    2. programs2_df (DataFrame): Second reduction programs dataset
    
    Processing:
    1. Counts unique companies with programs in each dataset.
    2. Analyzes distribution of program counts per company.
    3. Prints distributions of program types and categories if available.
    
    Returns:
    None (prints descriptive statistics and distributions to console)
    """
    print("\n--- Basic Programs Data Exploration ---")

    # Count of unique companies with programs
    print(f"Companies with reduction programs in programs1_df: {programs1_df['ISSUERID'].nunique()}")
    print(f"Companies with reduction programs in programs2_df: {programs2_df['ISSUERID'].nunique()}")

    # Look at the distribution of programs per company
    programs_per_company = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    print(f"\nDistribution of program counts per company (from programs2_df):")
    print(f"Min: {programs_per_company['program_count'].min()}")
    print(f"25th percentile: {programs_per_company['program_count'].quantile(0.25)}")
    print(f"Median: {programs_per_company['program_count'].median()}")
    print(f"75th percentile: {programs_per_company['program_count'].quantile(0.75)}")
    print(f"Max: {programs_per_company['program_count'].max()}")
    print(f"Mean: {programs_per_company['program_count'].mean():.2f}")

    # If TYPE column exists in programs1_df, show distribution
    if 'TYPE' in programs1_df.columns or any('TYPE' in col.upper() for col in programs1_df.columns):
        type_column = next((col for col in programs1_df.columns if 'TYPE' in col.upper()), None)
        if type_column:
            print(f"\nDistribution of program types (from programs1_df, column: {type_column}):")
            type_counts = programs1_df[type_column].value_counts()
            for prog_type, count in type_counts.items():
                print(f"- {prog_type}: {count} ({count/len(programs1_df)*100:.1f}%)")

    # If CATEGORY column exists in programs2_df, show distribution
    if 'PROGRAM_CATEGORY' in programs2_df.columns or any('CATEGORY' in col.upper() for col in programs2_df.columns):
        category_column = next((col for col in programs2_df.columns if 'CATEGORY' in col.upper()), None)
        if category_column:
            print(f"\nDistribution of program categories (from programs2_df, column: {category_column}):")
            category_counts = programs2_df[category_column].value_counts()
            for category, count in category_counts.head(10).items():
                print(f"- {category}: {count} ({count/len(programs2_df)*100:.1f}%)")
            if len(category_counts) > 10:
                print(f"... and {len(category_counts) - 10} more categories")

def analyze_detailed_programs(programs1_df, company_emissions_df):
    """
    Performs detailed analysis of program implementation, quality, and relationships with emissions.
    
    Inputs:
    1. programs1_df (DataFrame): First reduction programs dataset
    2. company_emissions_df (DataFrame): Merged company and emissions data
    
    Processing:
    1. Analyzes identifier, geographic, and categorical columns.
    2. Examines mitigation strategy implementation and quality.
    3. Assesses geographic distribution and relationship with emissions trends.
    4. Prints ANOVA results and summary statistics.
    
    Returns:
    None (prints detailed analysis and statistics to console)
    """
    print("\n--- Detailed Programs1 Analysis ---")

    # --- IDENTIFIER COLUMNS ANALYSIS ---
    print("\nIDENTIFIER COLUMNS ANALYSIS")
    print("-" * 80)

    id_columns = ['ISSUERID', 'ISSUER_NAME', 'ISSUER_TICKER', 'ISSUER_ISIN']
    for col in id_columns:
        missing = programs1_df[col].isna().sum()
        missing_pct = missing / len(programs1_df) * 100
        unique_count = programs1_df[col].nunique()
        print(f"{id_columns.index(col) + 1}. {col} - Missing Values: {missing} ({missing_pct:.2f}%), Unique Values: {unique_count}")

    # --- GEOGRAPHICAL DATA ANALYSIS ---
    print("\nGEOGRAPHICAL DATA ANALYSIS")
    print("-" * 80)
    print("5. ISSUER_CNTRY_DOMICILE\n")

    # Get top 10 countries
    top_countries = programs1_df['ISSUER_CNTRY_DOMICILE'].value_counts().head(10)
    total_records = len(programs1_df)

    print("Top 10 countries:")
    for country, count in top_countries.items():
        percentage = count / total_records * 100
        print(f"  - {country}: {count} ({percentage:.2f}%)")

    # Count unique countries
    unique_countries = programs1_df['ISSUER_CNTRY_DOMICILE'].nunique()
    print(f"Total unique countries: {unique_countries}")

    # --- CATEGORICAL COLUMNS ANALYSIS ---
    print("\nCATEGORICAL COLUMNS ANALYSIS")
    print("-" * 80)

    # Get mitigation strategy columns
    mitigation_columns = [col for col in programs1_df.columns if 'CBN_GHG_MITIG_' in col]

    # Analyze each mitigation column
    for i, column in enumerate(mitigation_columns, 6):
        # Get data type and missing values information
        dtype = programs1_df[column].dtype
        missing = programs1_df[column].isna().sum()
        missing_pct = missing / len(programs1_df) * 100
        unique_count = programs1_df[column].nunique()
        
        print(f"{i}. {column}")
        print(f"Data Type: {dtype}, Missing Values: {missing} ({missing_pct:.2f}%), Unique Values: {unique_count}")
        
        # Get category frequencies
        value_counts = programs1_df[column].value_counts(dropna=True)
        total = len(programs1_df)
        
        print("Categories by frequency:")
        for category, count in value_counts.items():
            percentage = count / total * 100
            print(f"  - {category}: {count} ({percentage:.2f}%)")
        print()

    # --- ADDITIONAL ANALYSES ---

    # 1. Analyze implementation rates for each strategy
    print("\nIMPLEMENTATION RATES ANALYSIS")
    print("-" * 80)
    implementation_rates = []

    for column in mitigation_columns:
        # Calculate implementation rate (non-null percentage)
        non_null_count = programs1_df[column].count()
        total = len(programs1_df)
        implementation_rate = non_null_count / total * 100
        clean_name = column.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
        
        implementation_rates.append((clean_name, implementation_rate))

    # Sort by implementation rate
    implementation_rates.sort(key=lambda x: x[1], reverse=True)

    print("Implementation rates for mitigation strategies:")
    for strategy, rate in implementation_rates:
        print(f"  - {strategy}: {rate:.2f}%")

    # 2. Geographic distribution of mitigation efforts
    print("\nGEOGRAPHIC DISTRIBUTION OF MITIGATION EFFORTS")
    print("-" * 80)

    # Get top 5 countries
    top5_countries = programs1_df['ISSUER_CNTRY_DOMICILE'].value_counts().head(5).index.tolist()

    print("Mitigation efforts by top 5 countries:")
    for country in top5_countries:
        print(f"\n{country}:")
        country_companies = programs1_df[programs1_df['ISSUER_CNTRY_DOMICILE'] == country]
        total_companies = len(country_companies)
        
        for column in mitigation_columns:
            clean_name = column.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
            
            # Calculate percentage with any mitigation effort
            has_mitigation = country_companies[
                (country_companies[column].notna()) & 
                (country_companies[column] != 'No') &
                (country_companies[column] != 'No evidence')
            ]
            
            mitigation_pct = len(has_mitigation) / total_companies * 100
            print(f"  - {clean_name}: {mitigation_pct:.2f}% of companies have mitigation efforts")

    # 3. Relationship with emissions trends
    print("\nRELATIONSHIP WITH EMISSIONS TRENDS")
    print("-" * 80)

    # Merge with emissions data
    merged_df = pd.merge(programs1_df, company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']], 
                         on='ISSUERID', how='inner')

    # Filter out companies with missing emissions trend data
    merged_df = merged_df.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])

    print(f"Companies with both emissions trend data and programs1 data: {len(merged_df)}")

    for column in mitigation_columns:
        clean_name = column.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
        print(f"\n{clean_name} Strategy:")
        
        # Get all non-null categories for this strategy
        categories = merged_df[column].dropna().unique().tolist()
        
        # Sort categories in a logical order if possible
        if 'No' in categories:
            categories.remove('No')
            categories = ['No'] + categories
        if 'No evidence' in categories:
            categories.remove('No evidence')
            categories = ['No evidence'] + categories
            
        # Try to move the most comprehensive strategies to the end
        for comprehensive in ['All or most stores and distribution centers', 
                           'All or core products', 
                           'All or core production facilities',
                           'Improvements in fleet, routes, AND load/packaging optimization',
                           'Aggressive efforts']:
            if comprehensive in categories:
                categories.remove(comprehensive)
                categories.append(comprehensive)
        
        # Calculate average emissions trend for each category
        for category in categories:
            cat_data = merged_df[merged_df[column] == category]['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']
            if not cat_data.empty:
                mean_val = cat_data.mean()
                count = len(cat_data)
                print(f"  - {category}: Average emissions trend = {mean_val:.2f}% (n={count})")
        
        # Perform ANOVA to check for significant differences
        groups = [merged_df[merged_df[column] == cat]['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].values 
                  for cat in categories if len(merged_df[merged_df[column] == cat]) > 0]
        
        if len(groups) > 1 and all(len(group) > 0 for group in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"  ANOVA test: F={f_stat:.2f}, p={p_value:.3f}")
            
            if p_value < 0.05:
                print("  Significant differences exist between categories")
            else:
                print("  No significant differences between categories")

    # 4. Quality of mitigation strategies
    print("\nQUALITY OF MITIGATION STRATEGIES")
    print("-" * 80)

    print("Distribution of response quality across mitigation strategies:")
    for column in mitigation_columns:
        clean_name = column.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
        print(f"\n{clean_name}:")
        
        # Group responses by quality
        high_quality = ['All or most stores and distribution centers', 
                       'All or core products', 
                       'All or core production facilities',
                       'Improvements in fleet, routes, AND load/packaging optimization',
                       'Aggressive efforts']
        
        medium_quality = ['Some stores/distribution centers (anecdotal cases)', 
                         'Some products (anecdotal cases)', 
                         'Some facilities (anecdotal cases)',
                         'Improvements in fleet, routes, OR load/packaging optimization',
                         'Some efforts']
        
        low_quality = ['General statement', 'Limited efforts / information']
        
        negative = ['No', 'No evidence']
        
        # Count responses in each category
        high_count = programs1_df[programs1_df[column].isin(high_quality)].shape[0]
        medium_count = programs1_df[programs1_df[column].isin(medium_quality)].shape[0]
        low_count = programs1_df[programs1_df[column].isin(low_quality)].shape[0]
        negative_count = programs1_df[programs1_df[column].isin(negative)].shape[0]
        missing_count = programs1_df[column].isna().sum()
        
        total = len(programs1_df)
        
        print(f"  High Quality:   {high_count:4d} ({high_count/total*100:5.1f}%)")
        print(f"  Medium Quality: {medium_count:4d} ({medium_count/total*100:5.1f}%)")
        print(f"  Low Quality:    {low_count:4d} ({low_count/total*100:5.1f}%)")
        print(f"  Negative:       {negative_count:4d} ({negative_count/total*100:5.1f}%)")
        print(f"  Missing:        {missing_count:4d} ({missing_count/total*100:5.1f}%)")

def main():
    """
    Runs the full analysis workflow for combined programs and targets.
    
    Inputs:
    None
    
    Processing:
    1. Loads all required data.
    2. Runs basic and detailed program analyses.
    3. Handles and prints any errors encountered.
    
    Returns:
    None (prints progress and results to console)
    """
    print("Starting Programs Combined Analysis...")
    
    try:
        # Load all data
        company_df, emissions_df, targets_df, programs1_df, programs2_df, company_emissions_df = load_data()
        
        # Run basic programs analysis
        analyze_basic_programs(programs1_df, programs2_df)
        
        # Run detailed programs1 analysis
        analyze_detailed_programs(programs1_df, company_emissions_df)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 