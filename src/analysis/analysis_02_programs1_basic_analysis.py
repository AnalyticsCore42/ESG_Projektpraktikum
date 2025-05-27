"""
Analyzes corporate carbon reduction programs and their relationship to emissions performance.

This script performs three main types of analysis:
1. Basic exploration of carbon reduction programs data
2. Analysis of mitigation strategy implementation
3. Correlation between program implementation and emissions performance (when emissions data is available)

Input DataFrames (both required):
1. company_emissions_df: Contains company emissions metrics and characteristics
2. programs1_df: Contains company carbon reduction program data

Outputs:
- Console output with program statistics and analysis results
- Returns a DataFrame with combined analysis results (or None if errors occur)
- Identifies key patterns in program effectiveness
"""

# ===== IMPORTS =====
import os
import sys
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from src.config import paths
from src.utils.general_utils import (
    categorize_size,
    categorize_region,
    categorize_industry,
    get_friendly_name_mapping,
    get_mitigation_score_maps,
)


warnings.filterwarnings('ignore')

PROGRAMS1_DATA_PATH = paths.DATA_DIR / "Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
COMPANY_EMISSIONS_PATH = paths.DATA_DIR / "company_emissions_merged.csv"

# ===== DATA LOADING =====

def load_data():
    """
    Load and prepare all necessary dataframes for programs1 and emissions analysis.
    
    Returns:
        tuple: (programs1_df, company_emissions_df) - The loaded dataframes.
               company_emissions_df will be None if the file is not found.
               Exits with error code 1 if programs1 data cannot be loaded.
    """
    print("\n--- LOADING DATA ---")
    print("=" * 80)
    
    # Data containers
    programs1_df = None
    company_emissions_df = None
    
    try:
        # --- Load Programs1 Data (Required) ---
        print(f"Attempting to load Programs 1 data from: {PROGRAMS1_DATA_PATH}")
        if not PROGRAMS1_DATA_PATH.is_file():
             print(f"ERROR: File not found at {PROGRAMS1_DATA_PATH}")
             import sys; sys.exit(1)

        # Load and clean the main programs1 dataset
        # Drop columns that are completely empty to reduce memory usage
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        print(f"Loaded Programs 1: shape={programs1_df.shape}")

        # --- Load Company Emissions Data (Optional) ---
        print(f"\nChecking for Company Emissions data at: {COMPANY_EMISSIONS_PATH}")
        if COMPANY_EMISSIONS_PATH.is_file():
            # Load company emissions data if available
            # This enables additional correlation analysis
            company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
            print(f"Loaded Company Emissions: shape={company_emissions_df.shape}")
            print("Note: Emissions correlation analysis will be performed.")
        else:
            print(f"Warning: Company emissions data not found at {COMPANY_EMISSIONS_PATH}")
            print("Proceeding with basic analysis only (no emissions correlation).")

        return programs1_df, company_emissions_df

    except Exception as e:
        # Handle any unexpected errors during data loading
        print(f"\nERROR: Critical failure during data loading: {e}")
        print("Please check the input files and try again.")
        import traceback
        traceback.print_exc()
        import sys; sys.exit(1)


# ===== BASIC ANALYSIS =====

def analyze_basic_programs1(programs1_df):
    """
    Perform comprehensive analysis of the programs1 dataset.
    
    This function provides insights into the structure, completeness, and quality
    of the carbon reduction programs data. It's designed to run independently
    of the emissions data.
    
    Key Analyses:
    1. Dataset Overview: Basic dimensions and content
    2. Data Quality: Missing values and completeness
    3. Identifier Analysis: Uniqueness and coverage
    4. Geographic Distribution: Country and region breakdown
    5. Mitigation Strategies: Implementation rates and quality assessment
    
    Parameters:
        programs1_df (DataFrame): Input data containing program 1 information.
                              Must contain at least the basic program data.
    
    Returns:
        None: Results are printed to console.
    """
    if programs1_df is None or programs1_df.empty:
        print("\nSkipping basic programs1 analysis: DataFrame is None or empty.")
        return

    print("\n--- BASIC PROGRAMS1 DATA EXPLORATION ---")
    print("=" * 80)

    # ===== 1. DATASET OVERVIEW =====
    print("\n1. BASIC DATAFRAME INFORMATION")
    print("-" * 80)
    print(f"Number of rows: {len(programs1_df)}")
    print(f"Number of columns: {len(programs1_df.columns)}")
    
    # Count unique companies using ISSUERID if available
    unique_companies = programs1_df['ISSUERID'].nunique() if 'ISSUERID' in programs1_df.columns else 'N/A (ISSUERID missing)'
    print(f"Unique companies: {unique_companies}")
    
    # Provide a quick summary of the data structure

    print("\nColumns in programs1_df:")
    for i, col in enumerate(programs1_df.columns, 1):
        print(f"{i}. {col}")

    # ===== 2. DATA QUALITY =====
    print("\n2. MISSING VALUES SUMMARY")
    print("-" * 80)
    
    # Calculate missing value statistics
    missing_values = programs1_df.isna().sum()
    missing_pct = (missing_values / len(programs1_df)) * 100
    
    # Create a summary dataframe sorted by missing values (descending)
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Values', ascending=False)

    # Display top 10 columns with most missing values
    print("Top 10 columns with missing values (most to least):")
    for i, (column, row) in enumerate(missing_summary.head(10).iterrows(), 1):
        print(f"{i}. {column}: {row['Missing Values']} values missing ({row['Missing Percentage']:.2f}%)")
        
        # Add a note for columns with complete data
        if i == 1 and row['Missing Values'] == 0:
            print("  Note: No missing values found in top columns.")
            break

    # ===== 3. IDENTIFIER ANALYSIS =====
    print("\n3. IDENTIFIER COLUMNS ANALYSIS")
    print("-" * 80)
    
    # Define the key identifier columns we expect to find
    id_columns = ['ISSUERID', 'ISSUER_NAME', 'ISSUER_TICKER', 'ISSUER_ISIN']
    
    # Check each identifier column for data quality
    for idx, col in enumerate(id_columns):
        if col in programs1_df.columns:
            # Calculate missing values and uniqueness metrics
            missing = programs1_df[col].isna().sum()
            missing_pct = missing / len(programs1_df) * 100 if len(programs1_df) > 0 else 0
            unique_count = programs1_df[col].nunique()
            print(f"{idx + 1}. {col} - Missing Values: {missing} ({missing_pct:.2f}%), Unique Values: {unique_count}")
        else:
            print(f"{idx + 1}. {col} - Column not found in DataFrame.")

    # ===== 4. GEOGRAPHIC DISTRIBUTION =====
    print("\n4. GEOGRAPHICAL DATA ANALYSIS")
    print("-" * 80)

    # Analyze country distribution
    country_col = 'ISSUER_CNTRY_DOMICILE'
    if country_col in programs1_df.columns:
        # Get top 10 countries by count of programs
        top_countries = programs1_df[country_col].value_counts().head(10)
        total_records = len(programs1_df)

        print("Top 10 countries:")
        if not top_countries.empty:
            for country, count in top_countries.items():
                percentage = count / total_records * 100 if total_records > 0 else 0
                print(f"  - {country}: {count} ({percentage:.2f}%)")
        else:
            print("  No country data available.")

        unique_countries = programs1_df[country_col].nunique()
        print(f"Total unique countries: {unique_countries}")
    else:
        print(f"'{country_col}' column not found.")

    region_col = 'ISSUER_REGION'
    if region_col in programs1_df.columns:
        print("\nRegional Distribution:")
        region_counts = programs1_df[region_col].value_counts()
        total_records = len(programs1_df)
        if not region_counts.empty:
            for region, count in region_counts.items():
                percentage = count / total_records * 100 if total_records > 0 else 0
                print(f"  - {region}: {count} ({percentage:.2f}%)")
        else:
            print("  No regional data available.")
    else:
         print(f"\n'{region_col}' column not found, skipping regional distribution.")

    # --- 5. Mitigation Strategies Analysis ---
    print("\n5. MITIGATION STRATEGIES OVERVIEW")
    print("-" * 80)

    # Find all columns related to mitigation strategies using naming pattern
    mitigation_columns = [col for col in programs1_df.columns if 'CBN_GHG_MITIG_' in col]
    print(f"Number of mitigation strategy columns found: {len(mitigation_columns)}")

    if mitigation_columns:
        # List all mitigation strategies with their data availability
        print("\nMitigation strategy columns (with non-null counts):")
        for i, col in enumerate(mitigation_columns, 1):
            # Clean up column names for display
            clean_name = col.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
            non_null_count = programs1_df[col].count()
            non_null_pct = (non_null_count / len(programs1_df)) * 100 if len(programs1_df) > 0 else 0
            print(f"{i}. {clean_name}: {non_null_count} non-null values ({non_null_pct:.2f}%)")

        # ===== 6. IMPLEMENTATION RATE =====
        print("\n6. IMPLEMENTATION RATES ANALYSIS")
        print("-" * 80)
        print("Calculating implementation rates for each strategy...")
        implementation_rates = []
        total_records = len(programs1_df)

        if total_records > 0:
            for column in mitigation_columns:
                non_null_count = programs1_df[column].count()
                implementation_rate = non_null_count / total_records * 100
                clean_name = column.replace('CBN_GHG_MITIG_', '').replace('_', ' ').title()
                implementation_rates.append((clean_name, implementation_rate))

            implementation_rates.sort(key=lambda x: x[1], reverse=True)

            print("Implementation rates for mitigation strategies:")
            for strategy, rate in implementation_rates:
                print(f"  - {strategy}: {rate:.2f}%")
        else:
            print("  Cannot calculate implementation rates (empty DataFrame).")

        # ===== 7. QUALITY ASSESSMENT =====
        print("\n7. QUALITY DISTRIBUTION ANALYSIS")
        print("-" * 80)
        print("Assessing quality of reported mitigation strategies...")
        
        # --- Quality categories ---
        high_quality = [
            'All or most stores and distribution centers',
            'All or core products',
            'All or core production facilities',
            'Improvements in fleet, routes, AND load/packaging optimization',
            'Aggressive efforts'
        ]

        medium_quality = [
            'Some stores/distribution centers (anecdotal cases)',
            'Some products (anecdotal cases)',
            'Some facilities (anecdotal cases)',
            'Improvements in fleet, routes, OR load/packaging optimization',
            'Some efforts'
        ]

        low_quality = [
            'General statement', 
            'Limited efforts / information'
        ]
        
        negative = [
            'No', 
            'No evidence'
        ]

        # --- Aggregate metrics ---
        high_total = 0
        medium_total = 0
        low_total = 0
        negative_total = 0
        missing_total = 0
        total_datapoints = len(programs1_df) * len(mitigation_columns)

        if total_datapoints > 0:
            for column in mitigation_columns:
                if column in programs1_df.columns:
                    high_total += programs1_df[programs1_df[column].isin(high_quality)].shape[0]
                    medium_total += programs1_df[programs1_df[column].isin(medium_quality)].shape[0]
                    low_total += programs1_df[programs1_df[column].isin(low_quality)].shape[0]
                    negative_total += programs1_df[programs1_df[column].isin(negative)].shape[0]
                    missing_total += programs1_df[column].isna().sum()

            print(f"High Quality:   {high_total:6d} ({high_total/total_datapoints*100:5.1f}%)")
            print(f"Medium Quality: {medium_total:6d} ({medium_total/total_datapoints*100:5.1f}%)")
            print(f"Low Quality:    {low_total:6d} ({low_total/total_datapoints*100:5.1f}%)")
            print(f"Negative:       {negative_total:6d} ({negative_total/total_datapoints*100:5.1f}%)")
            print(f"Missing:        {missing_total:6d} ({missing_total/total_datapoints*100:5.1f}%)")
        else:
            print("  No mitigation columns found or DataFrame empty, cannot analyze quality.")
    else:
        print("Skipping Mitigation Strategies Overview, Implementation Rates, and Quality Analysis as no mitigation columns were found.")


# ===== MITIGATION STRATEGY ANALYSIS =====

def analyze_mitigation_strategies_improved(programs_df, emissions_df):
    """
    Analyze the relationship between carbon reduction programs and emissions performance.
    
    This function performs a detailed analysis of how different mitigation strategies
    correlate with emissions trends. It combines program implementation data with
    emissions metrics to assess effectiveness.
    
    Key Features:
    - Individual strategy effectiveness analysis
    - Combined mitigation scoring across all strategies
    - Segmentation by company size, region, and industry
    - Correlation analysis with emissions trends
    
    Parameters:
        programs_df (DataFrame): Contains program implementation data with columns
                              for different mitigation strategies.
        emissions_df (DataFrame): Contains emissions metrics and company information.
                              Must include ISSUERID for merging.
    
    Returns:
        DataFrame: Combined analysis results with mitigation scores and segmentation.
                 Returns None if required data is missing or if errors occur.
                 
    Note:
        This function is designed to be robust to missing data and will provide
        detailed error messages if required columns are missing.
    """
    # Validate input data
    if programs_df is None or programs_df.empty:
        print("\nSkipping mitigation strategies analysis: programs1 data is None or empty.")
        return None
    if emissions_df is None or emissions_df.empty:
        print("\nSkipping detailed mitigation strategies analysis: emissions data is None or empty.")
        print("Note: Basic program analysis is still available without emissions data.")
        return None

    print("\n--- DETAILED MITIGATION STRATEGIES ANALYSIS ---")
    print("=" * 80)

    # --- Data Validation ---
    # Check for required columns in emissions data
    required_emissions_cols = [
        'ISSUERID',  # For merging with programs data
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',  # Emissions trend metric
        'CARBON_EMISSIONS_SCOPE_12_INTEN',  # Emissions intensity
        'MarketCap_USD',  # For size-based segmentation
        'NACE_CLASS_DESCRIPTION'  # For industry classification
    ]
    
    # Identify any missing required columns
    missing_req_cols = [col for col in required_emissions_cols if col not in emissions_df.columns]
    if missing_req_cols:
        print(f"ERROR: Missing required columns in emissions_df: {missing_req_cols}")
        print("These columns are required for the emissions correlation analysis.")
        return None

    # Check for ISSUERID in programs_df
    if 'ISSUERID' not in programs_df.columns:
        print("ERROR: 'ISSUERID' column missing from programs_df. Cannot merge.")
        return None

    # Prepare columns for merge (include country if available in either df)
    merge_cols_emissions = required_emissions_cols[:] # Copy list
    if 'ISSUER_CNTRY_DOMICILE' in emissions_df.columns:
        merge_cols_emissions.append('ISSUER_CNTRY_DOMICILE')

    # --- Data Integration ---
    # Merge programs data with emissions data
    try:
        # Perform an inner join to only keep companies present in both datasets
        # This ensures we only analyze companies with complete information
        analysis_df = pd.merge(
            programs_df,  # Left dataframe (programs data)
            emissions_df[merge_cols_emissions].drop_duplicates(subset=['ISSUERID']),  # Right dataframe (emissions data)
            on='ISSUERID',  # Join key
            how='inner'  # Only include matching records from both dataframes
        )
    except Exception as merge_error:
        print(f"ERROR during merge: {merge_error}")
        return None


    if analysis_df.empty:
        print("Warning: Merge resulted in an empty DataFrame. No common ISSUERIDs between programs1 and emissions data?")
        return None

    # Ensure country data is present (handle potential merge issues, prefer emissions if available)
    country_col = 'ISSUER_CNTRY_DOMICILE'
    if country_col not in analysis_df.columns and country_col in programs_df.columns:
         # If country was in programs but not emissions, merge it back carefully
         print(f"  Attempting to add missing '{country_col}' from programs_df...")
         country_data = programs_df[['ISSUERID', country_col]].drop_duplicates('ISSUERID')
         analysis_df = pd.merge(analysis_df, country_data, on='ISSUERID', how='left', suffixes=('', '_prog'))
         # Use the column from programs if it was successfully merged
         if f'{country_col}_prog' in analysis_df.columns:
              analysis_df[country_col] = analysis_df[f'{country_col}_prog']
              analysis_df.drop(columns=[f'{country_col}_prog'], inplace=True)
         else:
              print(f"  Warning: Could not merge '{country_col}' back from programs_df.")


    # Apply segmentation to the dataframe using the external helper functions
    print("Applying segmentation (Size, Region, Industry)...")
    try:
        # Check for required columns before applying functions
        if 'MarketCap_USD' in analysis_df.columns:
            analysis_df['Size_Category'] = analysis_df['MarketCap_USD'].apply(categorize_size)
        else:
            print("  Warning: 'MarketCap_USD' missing, cannot categorize size.")
            analysis_df['Size_Category'] = 'Unknown'

        if country_col in analysis_df.columns:
            analysis_df['Region'] = analysis_df[country_col].apply(categorize_region)
        else:
             print(f"  Warning: '{country_col}' missing, cannot categorize region.")
             analysis_df['Region'] = 'Unknown' # Handle missing country data

        if 'NACE_CLASS_DESCRIPTION' in analysis_df.columns:
            analysis_df['Industry_Group'] = analysis_df['NACE_CLASS_DESCRIPTION'].apply(categorize_industry) # <<< Needs import check
        else:
            print("  Warning: 'NACE_CLASS_DESCRIPTION' missing, cannot categorize industry.")
            analysis_df['Industry_Group'] = 'Unknown'

    except NameError as ne:
        print(f"ERROR during segmentation: A required function might be missing from imports: {ne}")
        return None # Stop analysis if segmentation fails
    except Exception as seg_error:
        print(f"ERROR during segmentation: {seg_error}")
        return None


    # Define mitigation strategy columns (ensure these exist in programs_df)
    mitigation_columns_base = [
        'CBN_GHG_MITIG_DISTRIBUTION',
        'CBN_GHG_MITIG_RAW_MAT',
        'CBN_GHG_MITIG_MFG',
        'CBN_GHG_MITIG_TRANSPORT',
        'CBN_GHG_MITIG_CAPTURE'
    ]
    # Filter to only columns actually present in the merged df
    mitigation_columns = [col for col in mitigation_columns_base if col in analysis_df.columns]
    if not mitigation_columns:
         print("Warning: No mitigation columns found in the merged data for detailed analysis.")
         # Decide if we should return None or continue without this part
         # Let's try to continue to the combined analysis if possible
         # return None
    else:
        print(f"Found mitigation columns for analysis: {mitigation_columns}")


    # Get friendly name mapping for display
    print("Getting friendly name mapping...")
    try:
        category_name_map = get_friendly_name_mapping() # <<< Needs import check
    except NameError as ne:
        print(f"ERROR getting name mapping: Function 'get_friendly_name_mapping' missing from imports: {ne}")
        return None
    except Exception as map_error:
        print(f"ERROR getting name mapping: {map_error}")
        return None


    # --- Individual Strategy Analysis ---
    # Analyze each mitigation strategy separately to understand its impact
    if mitigation_columns:  # Only proceed if we found valid strategy columns
        for column in mitigation_columns:
            # Prepare friendly name for display
            friendly_name = column.replace('CBN_GHG_MITIG_', '')
            
            # Calculate missing data statistics
            missing_count = analysis_df[column].isna().sum()
            missing_pct = 100 * missing_count / len(analysis_df) if len(analysis_df) > 0 else 0

            print(f"\n{'-' * 80}")
            print(f"ANALYSIS OF {friendly_name} (Missing: {missing_count} companies, {missing_pct:.2f}%)")
            print(f"{'-' * 80}")

            # Check if column has any non-null values before grouping
            if analysis_df[column].notna().any():
                # Group by the column, ensuring observed=False for older pandas compatibility if needed
                try:
                     # Use observed=True for modern pandas with Categorical data
                     # Use observed=False for older pandas or if column is not categorical
                     # Let's be safe and handle both for now
                     try:
                          grouped_data = analysis_df.groupby(column, observed=True)
                     except TypeError:
                          grouped_data = analysis_df.groupby(column) # Fallback for non-categorical or older pandas

                     # Check if grouping actually produced groups
                     if not grouped_data.groups:
                          print("  No groups formed for this strategy after grouping (column might have single value or all NaNs).")
                          continue

                     # Summary table for each individual strategy
                     results = pd.DataFrame({
                         'Count': grouped_data.size(), # More direct way to count
                         '%': 100 * grouped_data.size() / len(analysis_df),
                         '3Y Avg Trend': grouped_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean(),
                         'Avg Scope12 Int': grouped_data['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
                     }).sort_values('Count', ascending=False)

                     # Replace index with friendly names if mapping exists
                     results.index = [category_name_map.get(idx, str(idx)) for idx in results.index] # Ensure index is string
                     results = results.round(2)

                     # Check if results DataFrame is empty
                     if results.empty:
                         print("  No data available for this strategy after aggregation.")
                         continue # Skip to next strategy

                     # Determine column widths for pretty printing
                     max_category_width = max([len(str(idx)) for idx in results.index] + [15])
                     max_count_width = max([len(f"{val:.0f}" if not pd.isna(val) else "N/A") for val in results['Count']] + [6])
                     max_pct_width = max([len(f"{val:.2f}" if not pd.isna(val) else "N/A") for val in results['%']] + [5])
                     max_trend_width = max([len(f"{val:.2f}" if not pd.isna(val) else "N/A") for val in results['3Y Avg Trend']] + [12])
                     max_intensity_width = max([len(f"{val:.2f}" if not pd.isna(val) else "N/A") for val in results['Avg Scope12 Int']] + [15])

                     # Print table header
                     print(f"┌{'─' * (max_category_width + 2)}┬{'─' * (max_count_width + 2)}┬{'─' * (max_pct_width + 2)}┬{'─' * (max_trend_width + 2)}┬{'─' * (max_intensity_width + 2)}┐")
                     print(f"│ {'Category':<{max_category_width}} │ {'Count':>{max_count_width}} │ {'%':>{max_pct_width}} │ {'3Y Avg Trend':>{max_trend_width}} │ {'Avg Scope12 Int':>{max_intensity_width}} │")
                     print(f"├{'─' * (max_category_width + 2)}┼{'─' * (max_count_width + 2)}┼{'─' * (max_pct_width + 2)}┼{'─' * (max_trend_width + 2)}┼{'─' * (max_intensity_width + 2)}┤")

                     # Print each row of the summary table
                     for idx, row in results.iterrows():
                         count = f"{row['Count']:.0f}" if not pd.isna(row['Count']) else "N/A"
                         pct = f"{row['%']:.2f}" if not pd.isna(row['%']) else "N/A"
                         trend = f"{row['3Y Avg Trend']:.2f}" if not pd.isna(row['3Y Avg Trend']) else "N/A"
                         intensity = f"{row['Avg Scope12 Int']:.2f}" if not pd.isna(row['Avg Scope12 Int']) else "N/A"
                         print(f"│ {str(idx):<{max_category_width}} │ {count:>{max_count_width}} │ {pct:>{max_pct_width}} │ {trend:>{max_trend_width}} │ {intensity:>{max_intensity_width}} │") # Ensure idx is string

                     print(f"└{'─' * (max_category_width + 2)}┴{'─' * (max_count_width + 2)}┴{'─' * (max_pct_width + 2)}┴{'─' * (max_trend_width + 2)}┴{'─' * (max_intensity_width + 2)}┘")

                     # Key findings for individual strategies (customized per strategy) - Add checks for key existence
                     trends = {idx: val for idx, val in zip(results.index, results['3Y Avg Trend'])}
                     # Example check:
                     if friendly_name == 'DISTRIBUTION':
                          best_impl_key = category_name_map.get('All or most stores and distribution centers', 'All or most stores and distribution centers')
                          no_impl_key = category_name_map.get('No', 'No')
                          if best_impl_key in trends and no_impl_key in trends:
                               best_trend = trends[best_impl_key]
                               no_trend = trends[no_impl_key]
                               if not pd.isna(best_trend) and not pd.isna(no_trend):
                                    diff = abs(best_trend - no_trend)
                                    print(f"Key finding: '{best_impl_key}' implementation leads to {diff:.2f}% better emissions trend vs. '{no_impl_key}'.")
                          else:
                               print(f"Key finding: Could not compare best vs no implementation for {friendly_name} (missing categories in results).")
                     # ... Add similar checks for other friendly_name conditions ...


                except Exception as group_error:
                     print(f"  Error during grouping/aggregation for column '{column}': {group_error}")
                     continue # Skip to next strategy if grouping fails

            else:
                 print(f"  No non-null data found for strategy column: {column}")


    # --- Combined Mitigation Analysis ---
    # This section creates a composite score across all mitigation strategies
    print("\n--- Combined Mitigation Score Analysis ---")
    
    # Retrieve predefined score mappings for each strategy
    # These scores quantify the effectiveness/quality of different implementation levels
    print("Retrieving mitigation score mappings...")
    try:
        # Load score mappings from external function
        # Each strategy's implementation levels are mapped to numerical scores
        mitigation_score_maps = get_mitigation_score_maps()
    except NameError as ne:
        print(f"ERROR: Required function 'get_mitigation_score_maps' not found: {ne}")
        print("This function is needed to score different implementation levels.")
        return None
    except Exception as map_error:
        print(f"ERROR: Failed to load score mappings: {map_error}")
        return None


    # --- Prepare for Combined Analysis ---
    # Get unique companies that have both programs and emissions data
    unique_issuers = analysis_df['ISSUERID'].unique()
    if len(unique_issuers) == 0:
         print("No companies with both program and emissions data found. Cannot perform combined analysis.")
         return None

    # Initialize results dataframe with one row per company
    # This will store the combined scores across all strategies
    combined_df = pd.DataFrame({'ISSUERID': unique_issuers})
    
    # Filter out any mitigation columns that don't exist in the merged dataframe
    # This handles cases where expected columns might be missing
    valid_mitigation_columns = [col for col in mitigation_columns_base if col in analysis_df.columns]
    if not valid_mitigation_columns:
         print("No valid mitigation columns available for scoring.")
         print("Returning basic analysis results without combined scoring.")
         return analysis_df  # Return what we have without the combined scoring


    print(f"Calculating combined scores using columns: {valid_mitigation_columns}")
    # Calculate scores for each strategy by company
    for col in valid_mitigation_columns:
        # Map original categories to scores directly
        if col not in mitigation_score_maps:
             print(f"Warning: No score map found for column {col}. Skipping scoring for this column.")
             # Assign NaN or 0? Let's assign NaN so it doesn't skew the mean if mean(skipna=True) is used later.
             analysis_df[f'{col}_Score'] = np.nan
             continue

        score_map = mitigation_score_maps[col]
        # Apply map safely, defaulting missing keys/NaNs to 0 (or NaN if preferred)
        # Defaulting to 0 assumes missing info means no effort/score.
        analysis_df[f'{col}_Score'] = analysis_df[col].map(lambda x: score_map.get(x, 0) if pd.notna(x) else 0)

        # For each company, get the maximum score across all rows (in case of duplicates)
        # Group by ISSUERID and aggregate the max score
        try:
            company_scores = analysis_df.groupby('ISSUERID')[f'{col}_Score'].max()
            combined_df = pd.merge(combined_df, pd.DataFrame({f'{col}_Score': company_scores}),
                                  on='ISSUERID', how='left')
        except Exception as agg_error:
            print(f"Error aggregating scores for column {col}: {agg_error}")
            # Assign NaN to the score column in combined_df for robustness
            combined_df[f'{col}_Score'] = np.nan


    # --- Calculate Combined Scores ---
    # Collect all available score columns for the final calculation
    score_columns = [f'{col}_Score' for col in valid_mitigation_columns if f'{col}_Score' in combined_df.columns]
    if not score_columns:
         print("No score columns available for combined calculation.")
         print("This may indicate issues with the scoring process.")
         return analysis_df

    # Calculate the mean score across all strategies for each company
    # Using skipna=True ensures we still get a score even if some strategies are missing
    print(f"Calculating combined score from {len(score_columns)} strategies...")
    combined_df['Combined_Mitigation_Score'] = combined_df[score_columns].mean(axis=1, skipna=True)

    # Handle any remaining missing values (if all strategies were NaN)
    # Setting to 0 assumes no mitigation effort for missing data
    combined_df['Combined_Mitigation_Score'].fillna(0, inplace=True)

    # --- Categorize Scores ---
    # Convert numerical scores into interpretable categories
    print("Categorizing combined scores...")
    
    # Define bin edges and labels for score categorization
    # The bins are designed to create meaningful categories from the score distribution
    bins = [-0.1, 1.0, 2.0, 3.0, 4.5, 6.1]  # Adjusted to ensure proper binning
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    try:
        # Assign categories based on score ranges
        # Using right=True means bins are (left, right] except for the first bin which is [left, right]
        combined_df['Combined_Mitigation_Level'] = pd.cut(
            combined_df['Combined_Mitigation_Score'],
            bins=bins, 
            labels=labels, 
            right=True, 
            include_lowest=True  # Ensures the first bin includes its left edge
        )
    except Exception as bin_error:
        print(f"ERROR: Failed to categorize scores: {bin_error}")
        # Ensure we still have a level column even if binning fails
        combined_df['Combined_Mitigation_Level'] = 'Error'


    # --- Prepare Final Results ---
    # Combine the calculated scores with the original analysis data
    print("Merging combined scores with analysis data...")
    
    # Define essential columns to include in the final output
    # These are the key metrics we want to analyze alongside the mitigation scores
    essential_cols = [
        'ISSUERID',  # Primary key for merging
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',  # Emissions trend metric
        'CARBON_EMISSIONS_SCOPE_12_INTEN',  # Emissions intensity
        'Size_Category',  # Company size segmentation
        'Industry_Group'  # Industry classification
    ]
    
    # Include region if available (it's created during analysis)
    if 'Region' in analysis_df.columns:
        essential_cols.append('Region')  # For regional analysis

    # --- Prepare Reporting Data ---
    # Select only the columns that exist in the analysis dataframe
    # This handles cases where expected columns might be missing
    cols_to_select = [col for col in essential_cols if col in analysis_df.columns]
    
    # Ensure we always have the ISSUERID for merging
    if 'ISSUERID' not in cols_to_select: 
        cols_to_select.insert(0, 'ISSUERID')

    # Validate that we have the key emissions metrics for meaningful analysis
    required_metrics = ['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'CARBON_EMISSIONS_SCOPE_12_INTEN']
    if not all(col in analysis_df.columns for col in required_metrics):
        missing = [col for col in required_metrics if col not in analysis_df.columns]
        print(f"Warning: Missing key emissions metrics: {missing}")
        print("Some analyses may be limited or unavailable.")

    # Create a clean dataset for reporting with one row per company
    # Drop duplicates to ensure we don't have multiple rows per company
    reporting_data = analysis_df[cols_to_select].drop_duplicates('ISSUERID')

    # --- Combine All Results ---
    # Merge the calculated scores with the reporting data
    # Using left join to ensure we keep all companies with scores, even if some reporting data is missing
    combined_analysis = pd.merge(
        combined_df[['ISSUERID', 'Combined_Mitigation_Score', 'Combined_Mitigation_Level']],  # Left side: scores
        reporting_data,  # Right side: reporting metrics
        on='ISSUERID',   # Join key
        how='left'       # Keep all companies with scores, even if metrics are missing
    )

    # --- Present Results ---
    # Print a summary of the combined mitigation analysis
    # This gives a high-level view of the results
    print(f"\n{'=' * 80}")
    print("OVERALL COMBINED MITIGATION ANALYSIS")
    print(f"{'=' * 80}")
    print("This section shows how mitigation efforts correlate with emissions performance.")
    print("Higher mitigation scores indicate more comprehensive reduction programs.")

    # Check if the grouping column exists and has data
    if 'Combined_Mitigation_Level' in combined_analysis.columns and combined_analysis['Combined_Mitigation_Level'].notna().any():
        # Group by the combined level, ensuring observed=False/True for categorical compatibility
        try:
            # Use observed=True for modern pandas with Categorical data
            grouped_combined = combined_analysis.groupby('Combined_Mitigation_Level', observed=True)
        except TypeError:
            # Fallback for older pandas or non-categorical
            grouped_combined = combined_analysis.groupby('Combined_Mitigation_Level')

        # Check if grouping produced results before aggregation
        if grouped_combined.groups:
            agg_dict = {
                'Count': ('Combined_Mitigation_Level', 'size') # Use size for counting in groups
            }
            # Add aggregation for numeric columns only if they exist
            if 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' in combined_analysis.columns:
                agg_dict['3Y Avg Trend'] = ('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')
            if 'CARBON_EMISSIONS_SCOPE_12_INTEN' in combined_analysis.columns:
                agg_dict['Avg Scope12 Int'] = ('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')

            combined_summary = grouped_combined.agg(**agg_dict).round(2)

            print("\nSummary by Combined Mitigation Level:")
            # Use to_markdown for potentially better formatting if available, else to_string
            try:
                 print(combined_summary.to_markdown(numalign="left", stralign="left"))
            except ImportError:
                 print(combined_summary.to_string())
        else:
            print("\nNo groups formed for combined mitigation level summary.")

    else:
        print("\n'Combined_Mitigation_Level' column missing or empty, cannot generate summary.")


    # --- Segmentation Analysis ---
    # Break down results by different company characteristics
    # This helps identify patterns across different segments
    print("\n--- Segmentation Analysis for Combined Score ---")
    print("Analyzing results across different company segments...\n")
    
    # Define the segments to analyze
    # Each entry maps a display name to a column name in the data
    segments = {
        'Company Size': 'Size_Category',  # Small, Medium, Large companies
        'Region': 'Region' if 'Region' in combined_analysis.columns else None,  # Geographic regions
        'Industry Group': 'Industry_Group'  # Industry sectors
    }

    for segment_name, segment_col in segments.items():
        if segment_col is None or segment_col not in combined_analysis.columns:
            print(f"\nSkipping segmentation analysis for {segment_name} (column '{segment_col}' not available).")
            continue

        print(f"\n{'-' * 40}")
        print(f"{segment_name.upper()} ANALYSIS FOR COMBINED MITIGATION")
        print(f"{'-' * 40}")

        # Check if segment column and level column have data
        if combined_analysis[segment_col].isna().all() or 'Combined_Mitigation_Level' not in combined_analysis.columns or combined_analysis['Combined_Mitigation_Level'].isna().all():
             print(f"  No data available for segmentation: {segment_name}")
             continue

        # Implementation rates by combined mitigation level
        try:
            imp_rates = pd.crosstab(
                combined_analysis[segment_col],
                combined_analysis['Combined_Mitigation_Level'],
                normalize='index',
                dropna=False # Keep rows/cols with no data
            ) * 100
            print(f"\nImplementation Rates by {segment_name} (%):")
            print(imp_rates.round(1).to_markdown(numalign="left", stralign="left") if not imp_rates.empty else "  No implementation rate data.")
        except Exception as e:
            print(f"  Could not generate implementation rates for {segment_name}: {e}")


        # Emissions trend by segment and combined mitigation level
        if 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' in combined_analysis.columns:
            try:
                # Group by segment and level, ensuring observed=True/False
                try:
                     grouped_emissions = combined_analysis.groupby([segment_col, 'Combined_Mitigation_Level'], observed=True)
                except TypeError:
                     grouped_emissions = combined_analysis.groupby([segment_col, 'Combined_Mitigation_Level'])

                # Check if grouping produced results before aggregation
                if grouped_emissions.groups:
                    emissions_segment = grouped_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean().unstack()
                    print(f"\nAverage Emissions Trend by {segment_name} and Combined Mitigation Level (%):")
                    print(emissions_segment.round(2).to_markdown(numalign="left", stralign="left") if not emissions_segment.empty else "  No emissions trend data.")
                else:
                    print(f"\nNo groups formed for emissions trend analysis by {segment_name}.")

            except Exception as e:
                 print(f"  Could not generate emissions trends for {segment_name}: {e}")
        else:
            print(f"\nSkipping emissions trend analysis for {segment_name} ('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' column missing).")


    print("\nDetailed mitigation strategies analysis function finished.")
    return combined_analysis  # Return the combined analysis dataframe for potential further use

def main():
    """Main function to run the programs1 basic and mitigation strategies analysis."""
    print("="*40)
    print("Starting Programs1 Basic Analysis Script")
    print("="*40)

    try:
        programs1_df, company_emissions_df = load_data()

        if programs1_df is not None:
             analyze_basic_programs1(programs1_df)
             combined_results = analyze_mitigation_strategies_improved(programs1_df, company_emissions_df)

             if combined_results is not None and not combined_results.empty:
                  print("\nDetailed analysis produced results.")
             else:
                  print("\nDetailed analysis did not produce results or returned None (check warnings/errors above).")
        else:
             print("Essential data (programs1_df) could not be loaded. Aborting analysis.")

        print("\n" + "="*40)
        print("Analysis script finished.")
        print("="*40)

    except Exception as e:
        print(f"\n--- UNHANDLED ERROR IN MAIN EXECUTION ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error: {e}")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")

# ===== MAIN EXECUTION =====

def main():
    """Main function to run the programs1 basic and mitigation strategies analysis."""
    try:
        # Load the data
        programs1_df, company_emissions_df = load_data()
        
        # Perform basic analysis
        analyze_basic_programs1(programs1_df)
        
        # If we have emissions data, perform correlation analysis
        if company_emissions_df is not None and not company_emissions_df.empty:
            analyze_mitigation_strategies_improved(programs1_df, company_emissions_df)
        
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\nPlease check the error message above and try again.")
        return None

# ===== SCRIPT ENTRY POINT =====
if __name__ == "__main__":
    main()