"""
Detailed analysis of company reduction programs from the programs2 dataset.

Input DataFrames:
1. programs2_df: Contains program implementation details
2. company_emissions_df: Contains company emissions and metadata

Key Analyses Performed:
1. Program category distribution and characteristics
2. Implementation timelines and trends
3. Regional and industry patterns
4. Program oversight and governance structures

Outputs:
- Console output with detailed analysis results
- Visualizations of program characteristics and trends
- Markdown report with key findings and statistics
"""

# ===== IMPORTS =====
import os
import re
import sys
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.config import paths
from src.utils.general_utils import (
    categorize_size,
    categorize_region,
    categorize_industry,
    get_friendly_name_mapping,
)

warnings.filterwarnings('ignore')

# ===== FILE PATHS =====
# Update these paths if your data is stored elsewhere relative to the project root.
PROGRAMS2_DATA_PATH = "data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
COMPANY_EMISSIONS_PATH = "data/company_emissions_merged.csv"

# ===== DATA LOADING =====
def load_data():
    """
    Load and prepare all necessary dataframes.
    
    Processing Steps:
    1. Load programs2 data from CSV file
    2. Load company emissions data if available
    3. Print dataset information
    
    Returns:
        tuple: (programs2_df, company_emissions_df)
    
    Raises:
        FileNotFoundError: If required data files are not found
    """
    try:
        programs2_df = pd.read_csv(PROGRAMS2_DATA_PATH).dropna(axis=1, how='all')
        
        try:
            company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        except FileNotFoundError:
            print(f"Warning: Company emissions data not found at {COMPANY_EMISSIONS_PATH}")
            company_emissions_df = None
        
        print(f"\nPROGRAMS2 DATAFRAME: shape={programs2_df.shape}, unique companies={programs2_df['ISSUERID'].nunique():,}")
        if company_emissions_df is not None:
            print(f"COMPANY_EMISSIONS DATAFRAME: shape={company_emissions_df.shape}, unique companies={company_emissions_df['ISSUERID'].nunique():,}")
        
        return programs2_df, company_emissions_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data file is present at the specified path:")
        print(f"- {PROGRAMS2_DATA_PATH}")
        raise

# ===== ANALYSIS FUNCTIONS =====
def analyze_program_categories(programs2_df):
    """
    Analyze the distribution and characteristics of program categories.
    
    Parameters:
    1. programs2_df: Programs2 dataset with program details
    
    Processing Steps:
    1. Calculate category frequencies and overlaps
    2. Analyze category relationships
    3. Generate summary statistics
    
    Returns:
        DataFrame with program category analysis results
    """
    print("\n--- PROGRAM VALUES ANALYSIS ---")
    print("=" * 80)
    
    category_columns = [
        'CBN_PROG_LOW_CARB_RENEW',
        'CBN_EVIDENCE_TARG_ENERGY_IMPROV',
        'EXEC_BODY_ENV_ISSUES',
        'CBN_REG_ENERGY_AUDITS',
        'CBN_PROG_REDU_CARB_CORE_OP'
    ]
    
    print("\n1. DISTRIBUTION OF PROGRAM CATEGORIES")
    print("-" * 80)
    
    total_programs = len(programs2_df)
    print(f"Total number of programs: {total_programs:,}")
    
    category_counts = {}
    for col in category_columns:
        non_null_count = programs2_df[col].count()
        category_counts[col] = non_null_count
    
    print("Distribution of program categories:")
    for category, count in category_counts.items():
        percentage = count / total_programs * 100
        clean_name = category.replace('CBN_', '').replace('_', ' ').title()
        print(f"  - {clean_name}: {count:,} ({percentage:.2f}%)")
    
    print("\n2. VALUE DISTRIBUTION WITHIN CATEGORIES")
    print("-" * 80)
    
    for col in category_columns:
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        print(f"\nDistribution within {clean_name}:")
        
        value_counts = programs2_df[col].value_counts()
        total_non_null = category_counts[col]
        
        for value, count in value_counts.head(5).items():
            percentage = count / total_non_null * 100
            print(f"  - {value}: {count:,} ({percentage:.2f}%)")
        
        if len(value_counts) > 5:
            remaining_count = sum(value_counts.iloc[5:])
            remaining_pct = remaining_count / total_non_null * 100
            print(f"  - Other ({len(value_counts) - 5} more values): {remaining_count:,} ({remaining_pct:.2f}%)")
    
    if 'CBN_IMP_PROG_YEAR' in programs2_df.columns:
        print("\n3. PROGRAM CATEGORIES OVER TIME")
        print("-" * 80)
        
        time_df = programs2_df.dropna(subset=['CBN_IMP_PROG_YEAR'])
        
        print(f"Temporal analysis of program categories from {int(time_df['CBN_IMP_PROG_YEAR'].min())} to {int(time_df['CBN_IMP_PROG_YEAR'].max())}:")
        
        for col in category_columns:
            clean_name = col.replace('CBN_', '').replace('_', ' ').title()
            print(f"\n  Evolution of {clean_name}:")
            
            year_counts = time_df.groupby('CBN_IMP_PROG_YEAR')[col].count()
            
            recent_years = range(2013, 2024)
            
            for year in recent_years:
                if year in year_counts.index:
                    year_total = len(time_df[time_df['CBN_IMP_PROG_YEAR'] == year])
                    count = year_counts[year]
                    percentage = count / year_total * 100
                    print(f"    - {year}: {count:,} ({percentage:.1f}% of programs that year)")

def analyze_program_types(programs2_df):
    """
    Analyze the distribution and characteristics of program types.
    
    Parameters:
    1. programs2_df: Programs2 dataset with program details
    
    Processing Steps:
    1. Calculate distribution of program types
    2. Analyze relationships between program types and scopes
    3. Print summary statistics
    
    Returns:
        None: Results are printed to console
    """
    print("\n--- PROGRAM VALUES ANALYSIS ---")
    print("=" * 80)
    
    program_columns = [
        'CBN_PROG_LOW_CARB_RENEW',
        'CBN_PROG_REDU_CARB_CORE_OP'
    ]
    
    print("\n1. VALUE DISTRIBUTION FOR PROGRAM TYPES")
    print("-" * 80)
    
    for col in program_columns:
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        print(f"\nValues for {clean_name}:")
        
        value_counts = programs2_df[col].value_counts(dropna=False)
        total = len(programs2_df)
        
        for value, count in value_counts.items():
            percentage = count / total * 100
            if pd.isna(value):
                print(f"  - Missing: {count:,} ({percentage:.2f}%)")
            else:
                print(f"  - {value}: {count:,} ({percentage:.2f}%)")
    
    print("\n2. RELATIONSHIP BETWEEN PROGRAM TYPES")
    print("-" * 80)
    
    if len(program_columns) >= 2:
        crosstab = pd.crosstab(
            programs2_df[program_columns[0]], 
            programs2_df[program_columns[1]],
            margins=True
        )
        
        print(f"Relationship between {program_columns[0].replace('CBN_', '').replace('_', ' ').title()} and {program_columns[1].replace('CBN_', '').replace('_', ' ').title()}:")
        
        non_empty_count = programs2_df.dropna(subset=program_columns).shape[0]
        total_count = programs2_df.shape[0]
        
        print(f"Programs with data for both columns: {non_empty_count:,} ({non_empty_count/total_count*100:.2f}%)")
        
        most_common_combinations = []
        for val1 in crosstab.index:
            if val1 == 'All':
                continue
            for val2 in crosstab.columns:
                if val2 == 'All':
                    continue
                count = crosstab.loc[val1, val2]
                if count > 0:
                    most_common_combinations.append((val1, val2, count))
        
        most_common_combinations.sort(key=lambda x: x[2], reverse=True)
        
        print("\nTop 5 most common combinations:")
        for val1, val2, count in most_common_combinations[:5]:
            percentage = count / non_empty_count * 100
            print(f"  - {val1} + {val2}: {count:,} ({percentage:.2f}%)")

def analyze_program_oversight(programs2_df):
    """
    Analyze the oversight structure for carbon reduction programs.
    
    Parameters:
    1. programs2_df: Programs2 dataset with oversight details

    Processing Steps:
    1. Analyze oversight levels and their distribution
    2. Examine oversight frequency and responsibilities
    3. Print summary statistics
    
    Returns:
        None: Results are printed to console
    """
    print("\n--- PROGRAM OVERSIGHT ANALYSIS ---")
    print("=" * 80)
    
    oversight_column = 'EXEC_BODY_ENV_ISSUES'
    
    # 1. Basic distribution of oversight types
    print("\n1. DISTRIBUTION OF OVERSIGHT TYPES")
    print("-" * 80)
    
    # Calculate the missing percentage
    missing_oversight = programs2_df[oversight_column].isna().sum()
    missing_pct = missing_oversight / len(programs2_df) * 100
    print(f"Missing oversight data: {missing_oversight:,} ({missing_pct:.2f}%)")
    
    oversight_counts = programs2_df[oversight_column].value_counts(dropna=True)
    valid_programs = len(programs2_df) - missing_oversight
    
    print(f"Distribution of oversight types (among {valid_programs:,} programs with data):")
    for oversight, count in oversight_counts.items():
        percentage = count / valid_programs * 100
        print(f"  - {oversight}: {count:,} ({percentage:.2f}%)")
    
    # 2. Oversight by company
    print("\n2. COMPANIES BY OVERSIGHT TYPE")
    print("-" * 80)
    
    # Filter to programs with oversight data
    oversight_df = programs2_df.dropna(subset=[oversight_column])
    
    companies_per_oversight = oversight_df.groupby(oversight_column)['ISSUERID'].nunique()
    total_companies_with_oversight = oversight_df['ISSUERID'].nunique()
    
    print(f"Total companies with oversight data: {total_companies_with_oversight:,}")
    print("Companies per oversight type:")
    for oversight, count in companies_per_oversight.items():
        percentage = count / total_companies_with_oversight * 100
        print(f"  - {oversight}: {count:,} companies ({percentage:.2f}%)")
    
    # 3. Oversight trends over time
    if 'CBN_IMP_PROG_YEAR' in programs2_df.columns:
        print("\n3. OVERSIGHT TRENDS OVER TIME")
        print("-" * 80)
        
        # Filter for programs with both oversight and year data
        time_oversight_df = programs2_df.dropna(subset=[oversight_column, 'CBN_IMP_PROG_YEAR'])
        
        # Group by year and oversight
        year_oversight = pd.crosstab(
            time_oversight_df['CBN_IMP_PROG_YEAR'], 
            time_oversight_df[oversight_column]
        )
        
        # Get recent years
        available_years = sorted(time_oversight_df['CBN_IMP_PROG_YEAR'].unique())
        recent_years = [year for year in available_years if year >= 2013]  # Last ~10 years
        
        print(f"Oversight trends for years {min(recent_years)} to {max(recent_years)}:")
        
        # Get a subset for recent years
        if recent_years:
            year_oversight_subset = year_oversight.loc[recent_years]
            
            # Calculate percentages for each year
            year_totals = year_oversight_subset.sum(axis=1)
            year_oversight_pct = year_oversight_subset.div(year_totals, axis=0) * 100
            
            for year in recent_years:
                if year in year_oversight_subset.index:
                    print(f"\n  Year {int(year)}:")
                    for oversight in year_oversight_subset.columns:
                        count = year_oversight_subset.loc[year, oversight]
                        pct = year_oversight_pct.loc[year, oversight]
                        print(f"    - {oversight}: {count:.0f} ({pct:.1f}%)")

def analyze_geographical_distribution(programs2_df):
    """
    Analyze the geographical distribution of programs.
    
    Parameters:
    1. programs2_df: Programs2 dataset with location data

    Processing Steps:
    1. Calculate program distribution by country
    2. Count unique companies per country
    3. Calculate programs per company by country
    4. Print summary statistics
    
    Returns:
        None: Results are printed to console
    """
    print("\n--- GEOGRAPHICAL DISTRIBUTION ANALYSIS ---")
    print("=" * 80)
    
    if 'ISSUER_CNTRY_DOMICILE' not in programs2_df.columns:
        print("Country information not found in the dataset.")
        return
    
    print("\n1. PROGRAMS BY COUNTRY")
    print("-" * 80)
    
    country_counts = programs2_df['ISSUER_CNTRY_DOMICILE'].value_counts()
    total_programs = len(programs2_df)
    
    print("Top 15 countries by program count:")
    for country, count in country_counts.head(15).items():
        percentage = count / total_programs * 100
        print(f"  - {country}: {count:,} programs ({percentage:.2f}%)")
    
    print("\n2. COMPANIES BY COUNTRY")
    print("-" * 80)
    
    companies_per_country = programs2_df.groupby('ISSUER_CNTRY_DOMICILE')['ISSUERID'].nunique()
    total_companies = programs2_df['ISSUERID'].nunique()
    
    print("Top 15 countries by company count:")
    for country, count in companies_per_country.sort_values(ascending=False).head(15).items():
        percentage = count / total_companies * 100
        print(f"  - {country}: {count:,} companies ({percentage:.2f}%)")
    
    print("\n3. PROGRAMS PER COMPANY BY COUNTRY")
    print("-" * 80)
    
    programs_per_company = country_counts / companies_per_country
    
    print("Top 15 countries by programs per company:")
    for country, ratio in programs_per_company.sort_values(ascending=False).head(15).items():
        print(f"  - {country}: {ratio:.2f} programs per company")
    
    print("\n4. PROGRAM CATEGORIES BY REGION")
    print("-" * 80)
    
    programs2_df['Region'] = programs2_df['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
    
    region_category = pd.crosstab(
        programs2_df['Region'], 
        programs2_df['CBN_PROG_LOW_CARB_RENEW'],
        normalize='index'
    ) * 100
    
    for region in region_category.index:
        print(f"\nCategory distribution for {region}:")
        for category, percentage in region_category.loc[region].sort_values(ascending=False).items():
            print(f"  - {category}: {percentage:.1f}%")

def analyze_program_descriptions(programs2_df):
    """
    Analyze the textual content of program descriptions.
    
    Parameters:
    1. programs2_df: Programs2 dataset with description text
    
    Processing Steps:
    1. Preprocess and clean description text
    2. Perform text analysis (word frequency, key terms)
    3. Generate word clouds and visualizations
    
    Returns:
        None: Results are printed to console and visualizations are displayed
    """
    print("\n--- PROGRAM DESCRIPTIONS ANALYSIS ---")
    print("=" * 80)
    
    description_column = 'CBN_PROG_REDU_CARB_CORE_OP'
    
    if description_column not in programs2_df.columns:
        print("Program descriptions not found in the dataset.")
        return
    
    print("\n1. BASIC DESCRIPTION STATISTICS")
    print("-" * 80)
    
    desc_df = programs2_df.dropna(subset=[description_column])
    
    desc_lengths = desc_df[description_column].str.len()
    
    print(f"Programs with descriptions: {len(desc_df):,} ({len(desc_df)/len(programs2_df)*100:.2f}%)")
    print(f"Average description length: {desc_lengths.mean():.1f} characters")
    print(f"Median description length: {desc_lengths.median():.1f} characters")
    print(f"Shortest description: {desc_lengths.min()} characters")
    print(f"Longest description: {desc_lengths.max()} characters")
    
    print("\n2. MOST COMMON DESCRIPTIONS")
    print("-" * 80)
    
    description_counts = desc_df[description_column].value_counts().head(15)
    
    print("Top 15 most common descriptions:")
    for desc, count in description_counts.items():
        percentage = count / len(desc_df) * 100
        print(f"  - \"{desc}\": {count:,} occurrences ({percentage:.2f}%)")
    
    print("\n3. DESCRIPTION CATEGORIES")
    print("-" * 80)
    
    if desc_df[description_column].str.contains('No evidence').any():
        print("\nDistribution by evidence level:")
        for term in ['No evidence', 'Programs', 'General']:
            count = desc_df[desc_df[description_column].str.contains(term)].shape[0]
            pct = count / len(desc_df) * 100
            print(f"  - {term}: {count:,} ({pct:.2f}%)")

def analyze_implementation_timeline(programs2_df):
    """
    Analyze the implementation timeline of programs.
    
    Parameters:
    1. programs2_df: Programs2 dataset with date information

    Processing Steps:
    1. Parse and clean date fields
    2. Calculate implementation durations
    3. Analyze temporal patterns
    
    Returns:
        None: Results are printed to console
    """
    print("\n--- IMPLEMENTATION TIMELINE ANALYSIS ---")
    print("=" * 80)
    
    if 'CBN_IMP_PROG_YEAR' not in programs2_df.columns:
        print("Implementation year information not found in the dataset.")
        return
    
    print("\n1. IMPLEMENTATION YEAR DISTRIBUTION")
    print("-" * 80)
    
    year_df = programs2_df.dropna(subset=['CBN_IMP_PROG_YEAR'])
    
    year_df['CBN_IMP_PROG_YEAR'] = year_df['CBN_IMP_PROG_YEAR'].astype(int)
    
    year_counts = year_df['CBN_IMP_PROG_YEAR'].value_counts().sort_index()
    
    print(f"Programs with implementation year data: {len(year_df):,} ({len(year_df)/len(programs2_df)*100:.2f}%)")
    print(f"Earliest implementation year: {year_counts.index.min()}")
    print(f"Latest implementation year: {year_counts.index.max()}")
    
    recent_years = range(2010, 2025)
    print("\nImplementation year distribution (2010-2024):")
    for year in recent_years:
        if year in year_counts.index:
            count = year_counts[year]
            percentage = count / len(year_df) * 100
            print(f"  - {year}: {count:,} programs ({percentage:.2f}%)")
    
    print("\n2. PROGRAM TYPE IMPLEMENTATION BY YEAR")
    print("-" * 80)
    
    program_types = ['CBN_PROG_LOW_CARB_RENEW', 'CBN_PROG_REDU_CARB_CORE_OP']
    
    for prog_type in program_types:
        clean_name = prog_type.replace('CBN_', '').replace('_', ' ').title()
        print(f"\nImplementation years for {clean_name}:")
        
        type_year_df = year_df.dropna(subset=[prog_type])
        
        type_year_counts = type_year_df['CBN_IMP_PROG_YEAR'].value_counts().sort_index()
        
        for year in recent_years:
            if year in type_year_counts.index:
                count = type_year_counts[year]
                year_total = year_counts[year] if year in year_counts.index else 0
                if year_total > 0:
                    percentage = count / year_total * 100
                    print(f"  - {year}: {count:,} programs ({percentage:.2f}% of programs that year)")
    
    print("\n3. IMPLEMENTATION TIMELINE BY REGION")
    print("-" * 80)
    
    year_df['Region'] = year_df['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
    
    avg_year_by_region = year_df.groupby('Region')['CBN_IMP_PROG_YEAR'].agg(['mean', 'median', 'min', 'max', 'count'])
    
    print("Implementation timeline by region:")
    for region, stats in avg_year_by_region.sort_values('mean', ascending=False).iterrows():
        print(f"\n  {region}:")
        print(f"    - Average year: {stats['mean']:.1f}")
        print(f"    - Median year: {stats['median']:.1f}")
        print(f"    - Earliest: {stats['min']}")
        print(f"    - Latest: {stats['max']}")
        print(f"    - Program count: {stats['count']:,}")

def analyze_emissions_relationship(programs2_df, company_emissions_df):
    """
    Analyze the relationship between programs and company emissions.
    
    Parameters:
    1. programs2_df: Programs2 dataset with program details
    2. company_emissions_df: Company emissions data

    Processing Steps:
    1. Merge program and emissions data
    2. Calculate correlations between program features and emissions
    3. Analyze program effectiveness
    
    Returns:
        None: Results are printed to console
    """
    print("\n--- EMISSIONS RELATIONSHIP ANALYSIS ---")
    print("=" * 80)
    
    if company_emissions_df is None:
        print("Company emissions data not available for analysis.")
        return
    
    print("\n1. EMISSIONS VS PROGRAM COUNT")
    print("-" * 80)
    
    program_counts = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    
    emissions_cols = [col for col in company_emissions_df.columns if 'CARBON' in col or 'EMISSION' in col]
    key_emission_cols = [
        'ISSUERID',
        'CARBON_EMISSIONS_SCOPE_12',
        'CARBON_EMISSIONS_SCOPE_2',
        'CARBON_EMISSIONS_SCOPE_3',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',
        'CARBON_EMISSIONS_SCOPE_12_INTEN'
    ]
    
    available_cols = ['ISSUERID'] + [col for col in key_emission_cols[1:] if col in company_emissions_df.columns]
    
    emissions_metrics = company_emissions_df[available_cols].drop_duplicates()
    
    merged_df = pd.merge(program_counts, emissions_metrics, on='ISSUERID', how='inner')
    
    print(f"Companies with both programs and emissions data: {len(merged_df)}")
    
    correlations = {}
    
    if 'CARBON_EMISSIONS_SCOPE_12' in merged_df.columns:
        correlations['Scope 1+2 Emissions'] = merged_df['CARBON_EMISSIONS_SCOPE_12'].corr(merged_df['program_count'])
    
    if 'CARBON_EMISSIONS_SCOPE_2' in merged_df.columns:
        correlations['Scope 2 Emissions'] = merged_df['CARBON_EMISSIONS_SCOPE_2'].corr(merged_df['program_count'])
    
    if 'CARBON_EMISSIONS_SCOPE_3' in merged_df.columns:
        correlations['Scope 3 Emissions'] = merged_df['CARBON_EMISSIONS_SCOPE_3'].corr(merged_df['program_count'])
    
    if 'CARBON_EMISSIONS_SCOPE_12_INTEN' in merged_df.columns:
        correlations['Emissions Intensity'] = merged_df['CARBON_EMISSIONS_SCOPE_12_INTEN'].corr(merged_df['program_count'])
    
    if 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' in merged_df.columns:
        correlations['3-Year Trend'] = merged_df['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].corr(merged_df['program_count'])
    
    print("Correlation between program count and emissions metrics:")
    for metric, corr in correlations.items():
        if not np.isnan(corr):
            print(f"  - {metric}: {corr:.3f}")
        else:
            print(f"  - {metric}: N/A (insufficient data)")
    
    print("\n2. PROGRAM CATEGORIES VS EMISSION TRENDS")
    print("-" * 80)
    
    if 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' not in merged_df.columns:
        print("Emissions trend data not available. Skipping this analysis.")
        return merged_df
    
    category_companies = {}
    for category in programs2_df['CBN_PROG_LOW_CARB_RENEW'].unique():
        if pd.notna(category):
            companies = programs2_df[programs2_df['CBN_PROG_LOW_CARB_RENEW'] == category]['ISSUERID'].unique()
            category_companies[category] = set(companies)
    
    trend_by_category = {}
    for category, companies in category_companies.items():
        category_emissions = company_emissions_df[company_emissions_df['ISSUERID'].isin(companies)]
        
        if 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR' in category_emissions.columns:
            valid_trends = category_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].dropna()
            if len(valid_trends) > 0:
                avg_trend = valid_trends.mean()
                trend_by_category[category] = avg_trend
    
    print("Average emission trends by program category:")
    for category, trend in sorted(trend_by_category.items(), key=lambda x: x[1]):
        print(f"  - {category}: {trend:.2f}%")
    
    print("\n3. IMPLEMENTATION TIMELINE VS EMISSIONS TREND")
    print("-" * 80)
    
    if 'CBN_IMP_PROG_YEAR' in programs2_df.columns:
        year_df = programs2_df.dropna(subset=['CBN_IMP_PROG_YEAR'])
        avg_impl_year = year_df.groupby('ISSUERID')['CBN_IMP_PROG_YEAR'].mean()
        
        timeline_trend_df = pd.DataFrame({
            'ISSUERID': avg_impl_year.index,
            'avg_implementation_year': avg_impl_year.values
        })
        
        timeline_trend_df = pd.merge(
            timeline_trend_df,
            company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates(),
            on='ISSUERID',
            how='inner'
        )
        
        timeline_trend_df = timeline_trend_df.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
        
        if len(timeline_trend_df) > 0:
            year_trend_corr = timeline_trend_df['avg_implementation_year'].corr(timeline_trend_df['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
            
            print(f"Correlation between avg implementation year and emission trend: {year_trend_corr:.3f}")
            
            if len(timeline_trend_df) >= 10:
                try:
                    timeline_trend_df['year_group'] = pd.cut(
                        timeline_trend_df['avg_implementation_year'], 
                        bins=[2010, 2013, 2016, 2019, 2022, 2025],
                        labels=['2010-2012', '2013-2015', '2016-2018', '2019-2021', '2022-2024']
                    )
                    
                    trend_by_year_group = timeline_trend_df.groupby('year_group')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].agg(['mean', 'count'])
                    
                    print("\nEmission trends by program implementation timeline:")
                    for year_group, stats in trend_by_year_group.iterrows():
                        print(f"  - {year_group}: {stats['mean']:.2f}% trend ({stats['count']:,} companies)")
                except Exception as e:
                    print(f"Error in year group analysis: {e}")
        else:
            print("Insufficient data for timeline vs. trend analysis.")
    
    return merged_df

def analyze_programs2_comprehensive(programs2_df, company_emissions_df):
    """
    Run comprehensive analysis on the programs2 dataset.
    
    Parameters:
    1. programs2_df: Programs2 dataset with all program details
    2. company_emissions_df: Company emissions and financial data
    
    Processing Steps:
    1. Execute all analysis functions in sequence
    2. Generate summary statistics and visualizations
    3. Save results to file
    
    Returns:
        None: Results are printed to console and saved to file
    """
    print("\n====== PROGRAMS2 COMPREHENSIVE ANALYSIS ======")
    print("=" * 80)
    
    analyze_program_categories(programs2_df)
    analyze_program_types(programs2_df)
    analyze_program_oversight(programs2_df)
    analyze_geographical_distribution(programs2_df)
    analyze_program_descriptions(programs2_df)
    analyze_implementation_timeline(programs2_df)
    
    if company_emissions_df is not None:
        analyze_emissions_relationship(programs2_df, company_emissions_df)
    
    print("\n====== ANALYSIS COMPLETE ======")

def save_results_to_file(programs2_df, company_emissions_df):
    """
    Save analysis results to a markdown file.
    
    Parameters:
    1. programs2_df: Programs2 dataset with analysis results
    2. company_emissions_df: Company emissions data used in analysis
    
    Processing Steps:
    1. Generate markdown content with analysis results
    2. Create reports directory if it doesn't exist
    3. Save to a timestamped file in the reports directory
    
    Returns:
        str: Path to the saved markdown file
    """
    original_stdout = sys.stdout
    
    results_dir = paths.REPORTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "programs2_detailed_analysis.md"), 'w') as f:
        sys.stdout = f
        
        analyze_programs2_comprehensive(programs2_df, company_emissions_df)
        
        sys.stdout = original_stdout
    
    print(f"Analysis results saved to {os.path.join(results_dir, 'programs2_detailed_analysis.md')}")

if __name__ == "__main__":
    programs2_df, company_emissions_df = load_data()
    
    analyze_programs2_comprehensive(programs2_df, company_emissions_df)
    
    save_results_to_file(programs2_df, company_emissions_df) 