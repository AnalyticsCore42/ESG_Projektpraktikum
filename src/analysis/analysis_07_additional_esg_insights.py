"""
Additional ESG Insights - Advanced Analysis Script

This script provides in-depth analysis of various ESG (Environmental, Social, and Governance) metrics,
with a focus on uncovering patterns and insights not covered in the standard analysis scripts.
It includes analysis of program effectiveness, regional emissions patterns, industry code patterns,
and regulatory influences on emissions performance.

Key Features:
- Analysis of carbon reduction program effectiveness
- Regional emissions pattern analysis
- Industry sector emissions profiling using NACE codes
- Regulatory framework impact assessment

Note: This script is part of the ESG Analysis Project and is designed to be run after the
core analysis scripts have been executed.
"""

# Additional ESG Insights - Advanced Analysis Script
# This script explores dimensions of ESG data not covered in previous analyses

# ===== IMPORTS =====
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils.general_utils import (
    categorize_size,
    categorize_region,
    categorize_industry,
)

warnings.filterwarnings('ignore')

# --- Data Loading ---
PROGRAMS1_DATA_PATH = "data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
PROGRAMS2_DATA_PATH = "data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
COMPANY_EMISSIONS_PATH = "data/company_emissions_merged.csv"
TARGETS_DATA_PATH = "data/Reduktionsziele Results - 20241212 15_49_29.csv"

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all necessary dataframes for ESG analysis.
    
    This function loads four key datasets required for the analysis:
    - Programs1: Initial carbon reduction program data
    - Programs2: Detailed carbon reduction program data
    - Company Emissions: Emissions data by company
    - Targets: Emission reduction targets set by companies
    
    Returns:
        Tuple containing four DataFrames in order:
            programs1_df: Initial program data
            programs2_df: Detailed program data
            company_emissions_df: Company emissions data
            
    Raises:
        FileNotFoundError: If any required data file is missing
    """
    try:
        # Load the necessary dataframes
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        programs2_df = pd.read_csv(PROGRAMS2_DATA_PATH).dropna(axis=1, how='all')
        company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        targets_df = pd.read_csv(TARGETS_DATA_PATH).dropna(axis=1, how='all')
        
        # Output dataset information
        print(f"\nPROGRAMS1 DATAFRAME: shape={programs1_df.shape}, unique companies={programs1_df['ISSUERID'].nunique():,}")
        print(f"PROGRAMS2 DATAFRAME: shape={programs2_df.shape}, unique companies={programs2_df['ISSUERID'].nunique():,}")
        print(f"COMPANY_EMISSIONS DATAFRAME: shape={company_emissions_df.shape}, unique companies={company_emissions_df['ISSUERID'].nunique():,}")
        print(f"TARGETS DATAFRAME: shape={targets_df.shape}, unique companies={targets_df['ISSUERID'].nunique():,}")
        
        return programs1_df, programs2_df, company_emissions_df, targets_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are present at the specified paths")
        raise

def analyze_program_effectiveness(programs2_df, company_emissions_df):
    """
    Analyze the effectiveness of different carbon reduction programs.
    
    Args:
        programs2_df (pd.DataFrame): DataFrame containing program implementation data
        company_emissions_df (pd.DataFrame): DataFrame with company emissions data
        
    Returns:
        dict: Analysis results including program effectiveness metrics
    """
    print("\n--- PROGRAM TYPE EFFECTIVENESS ANALYSIS ---")
    print("=" * 80)
    
    # 1. Identify program type columns
    program_columns = [
        'CBN_PROG_REDU_CARB_CORE_OP',
        'CBN_PROG_LOW_CARB_RENEW',
        'CBN_EVIDENCE_TARG_ENERGY_IMPROV',
        'EXEC_BODY_ENV_ISSUES',
        'CBN_REG_ENERGY_AUDITS'
    ]
    
    # 2. Prepare consolidated dataset
    print("\n1. PROGRAM TYPE DISTRIBUTION")
    print("-" * 80)
    
    # Get count of non-null values for each program type
    program_distribution = {col: programs2_df[col].count() for col in program_columns}
    total_programs = sum(program_distribution.values())
    
    # Print distribution
    print(f"Total program entries: {total_programs:,}")
    for col, count in sorted(program_distribution.items(), key=lambda x: x[1], reverse=True):
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        percentage = count / len(programs2_df) * 100
        print(f"  - {clean_name}: {count:,} entries ({percentage:.1f}%)")
    
    # 3. For each program type, analyze the distribution of values/categories
    print("\n2. PROGRAM TYPE CATEGORY DISTRIBUTIONS")
    print("-" * 80)
    
    for col in program_columns:
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        print(f"\n{clean_name} Categories:")
        
        value_counts = programs2_df[col].value_counts()
        total = value_counts.sum()
        
        for val, count in value_counts.items():
            if pd.notna(val):  # Skip NaN values
                percentage = count / total * 100
                print(f"  - {val}: {count:,} ({percentage:.1f}%)")
    
    # 4. Company-level program types analysis
    print("\n3. COMPANY-LEVEL PROGRAM TYPES ANALYSIS")
    print("-" * 80)
    
    # Aggregate programs by company
    company_programs = programs2_df.groupby('ISSUERID')[program_columns].apply(
        lambda df: {col: df[col].value_counts().to_dict() for col in program_columns}
    )
    
    # Count companies with each program type
    companies_with_program = {}
    for col in program_columns:
        has_program = sum(1 for _, programs in company_programs.items() 
                         if programs[col] and any(pd.notna(k) for k in programs[col].keys()))
        percentage = has_program / len(company_programs) * 100
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        companies_with_program[clean_name] = (has_program, percentage)
    
    print("Companies implementing each program type:")
    for prog_type, (count, pct) in sorted(companies_with_program.items(), key=lambda x: x[1][1], reverse=True):
        print(f"  - {prog_type}: {count:,} companies ({pct:.1f}%)")
    
    # 5. Emissions performance by program type
    print("\n4. EMISSIONS PERFORMANCE BY PROGRAM TYPE")
    print("-" * 80)
    
    # Merge with emissions data
    common_companies = set(programs2_df['ISSUERID'].unique()) & set(company_emissions_df['ISSUERID'].unique())
    print(f"Companies with both program and emissions data: {len(common_companies):,}")
    
    # For each program type, calculate average emissions metrics
    for col in program_columns:
        clean_name = col.replace('CBN_', '').replace('_', ' ').title()
        print(f"\n{clean_name} Impact on Emissions:")
        
        # Extract dominant program type per company
        company_dominant_program = defaultdict(str)
        for company_id, programs in company_programs.items():
            if company_id in common_companies and col in programs and programs[col]:
                # Get most frequent program value (excluding NaN)
                valid_programs = {k: v for k, v in programs[col].items() if pd.notna(k)}
                if valid_programs:
                    dominant_program = max(valid_programs.items(), key=lambda x: x[1])[0]
                    company_dominant_program[company_id] = dominant_program
        
        # Create temporary DataFrame with company IDs and dominant program
        temp_df = pd.DataFrame({
            'ISSUERID': list(company_dominant_program.keys()),
            'dominant_program': list(company_dominant_program.values())
        })
        
        # Merge with emissions data
        merged_df = pd.merge(
            temp_df, 
            company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_12_INTEN', 
                                 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates(),
            on='ISSUERID', 
            how='inner'
        )
        
        # Calculate emissions metrics by program category
        if not merged_df.empty:
            emissions_by_program = merged_df.groupby('dominant_program').agg({
                'ISSUERID': 'count',
                'CARBON_EMISSIONS_SCOPE_12': 'mean',
                'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
                'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
            })
            
            # Sort by emissions trend (improving to worsening)
            emissions_by_program = emissions_by_program.sort_values('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR')
            
            # Print results
            for program, metrics in emissions_by_program.iterrows():
                company_count = metrics['ISSUERID']
                intensity = metrics['CARBON_EMISSIONS_SCOPE_12_INTEN']
                trend = metrics['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']
                
                print(f"  - {program} ({company_count:,} companies):")
                print(f"    * Emissions Intensity: {intensity:.2f}")
                print(f"    * 3-Year Trend: {trend:.2f}% (negative is improving)")
        else:
            print("  Insufficient data for analysis")
    
    return company_programs

def analyze_regional_emissions_patterns(company_emissions_df):
    """
    Analyze emissions patterns across different geographic regions.
    
    Args:
        company_emissions_df (pd.DataFrame): DataFrame with company emissions data
        
    Returns:
        tuple: A tuple containing two DataFrames:
            - Country-level metrics
            - Region-level metrics
    """
    print("\n--- DETAILED GEOGRAPHIC EMISSIONS ANALYSIS ---")
    print("=" * 80)
    
    # 1. Country-level analysis
    print("\n1. COUNTRY-LEVEL EMISSIONS ANALYSIS")
    print("-" * 80)
    
    # Prepare country data
    country_df = company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_emissions', 
                                     'CARBON_EMISSIONS_SCOPE_12', 
                                     'CARBON_EMISSIONS_SCOPE_12_INTEN',
                                     'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()
    
    # Filter to companies with emissions data
    country_df = country_df.dropna(subset=['CARBON_EMISSIONS_SCOPE_12'])
    
    # Calculate metrics by country
    country_metrics = country_df.groupby('ISSUER_CNTRY_DOMICILE_emissions').agg({
        'ISSUERID': 'count',
        'CARBON_EMISSIONS_SCOPE_12': ['sum', 'mean'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
    })
    
    # Calculate percentage of total emissions
    total_emissions = country_df['CARBON_EMISSIONS_SCOPE_12'].sum()
    country_metrics['emissions_percentage'] = country_metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')] / total_emissions * 100
    
    # Sort by total emissions
    country_metrics = country_metrics.sort_values(('CARBON_EMISSIONS_SCOPE_12', 'sum'), ascending=False)
    
    # Print top countries by emissions
    print("Top 15 countries by total emissions:")
    for i, (country, metrics) in enumerate(country_metrics.head(15).iterrows(), 1):
        company_count = float(metrics[('ISSUERID', 'count')])
        total_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')])
        avg_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'mean')])
        intensity = float(metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')])
        trend = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')])
        pct = float(metrics['emissions_percentage'])
        
        print(f"  {i}. {country} ({company_count:,.0f} companies):")
        print(f"     - Total Emissions: {total_emissions:,.0f} tons CO2e ({pct:.1f}% of global)")
        print(f"     - Avg per Company: {avg_emissions:,.0f} tons CO2e")
        print(f"     - Intensity: {intensity:.2f}")
        print(f"     - 3-Year Trend: {trend:.2f}%")
    
    # 2. Regional trends
    print("\n2. REGIONAL EMISSION TRENDS")
    print("-" * 80)
    
    # Add region categorization
    country_df['Region'] = country_df['ISSUER_CNTRY_DOMICILE_emissions'].apply(categorize_region)
    
    # Calculate metrics by region
    region_metrics = country_df.groupby('Region').agg({
        'ISSUERID': 'count',
        'CARBON_EMISSIONS_SCOPE_12': ['sum', 'mean'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
    })
    
    # Calculate percentage of total emissions
    region_metrics['emissions_percentage'] = region_metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')] / total_emissions * 100
    
    # Sort by emissions trend (improving to worsening)
    region_metrics = region_metrics.sort_values(('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean'))
    
    print("Regional emission trends (sorted by improving to worsening):")
    for region, metrics in region_metrics.iterrows():
        company_count = float(metrics[('ISSUERID', 'count')])
        total_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')])
        pct = float(metrics['emissions_percentage'])
        intensity = float(metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')])
        trend = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')])
        
        print(f"\n  {region} ({company_count:,.0f} companies):")
        print(f"    - Total Emissions: {total_emissions:,.0f} tons CO2e ({pct:.1f}% of global)")
        print(f"    - Emissions Intensity: {intensity:.2f}")
        print(f"    - 3-Year Trend: {trend:.2f}%")
    
    # 3. Country improvement leaders
    print("\n3. COUNTRY IMPROVEMENT LEADERS")
    print("-" * 80)
    
    # Filter to countries with enough companies for statistical significance
    significant_countries = country_metrics[country_metrics[('ISSUERID', 'count')] >= 20].index
    filtered_country_df = country_df[country_df['ISSUER_CNTRY_DOMICILE_emissions'].isin(significant_countries)]
    
    # Calculate metrics focusing on improvement
    improvement_metrics = filtered_country_df.groupby('ISSUER_CNTRY_DOMICILE_emissions').agg({
        'ISSUERID': 'count',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': ['mean', 'median', 'std']
    })
    
    # Sort by improvement (lowest/negative trend first)
    improvement_metrics = improvement_metrics.sort_values(('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean'))
    
    print("Top 10 countries by emissions improvement rate:")
    for i, (country, metrics) in enumerate(improvement_metrics.head(10).iterrows(), 1):
        company_count = metrics[('ISSUERID', 'count')]
        mean_trend = metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')]
        median_trend = metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'median')]
        
        print(f"  {i}. {country} ({company_count:,} companies):")
        print(f"     - Avg 3-Year Trend: {mean_trend:.2f}%")
        print(f"     - Median 3-Year Trend: {median_trend:.2f}%")
    
    return country_metrics, region_metrics

def analyze_industry_code_patterns(company_emissions_df):
    """
    Analyze emissions patterns across different industry sectors using NACE codes.
    
    Args:
        company_emissions_df (pd.DataFrame): DataFrame with company emissions data
        
    Returns:
        tuple: A tuple containing two DataFrames:
            - Section-level industry metrics
            - Class-level industry metrics
    """
    print("\n--- NACE INDUSTRY CODE ANALYSIS ---")
    print("=" * 80)
    
    # 1. Prepare industry data
    industry_df = company_emissions_df[['ISSUERID', 'NACE_CLASS_CODE', 'NACE_CLASS_DESCRIPTION',
                                      'CARBON_EMISSIONS_SCOPE_12', 
                                      'CARBON_EMISSIONS_SCOPE_12_INTEN',
                                      'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()
    
    # Filter to companies with emissions data
    industry_df = industry_df.dropna(subset=['CARBON_EMISSIONS_SCOPE_12', 'NACE_CLASS_CODE'])
    
    # Extract NACE section (first letter) from class code
    industry_df['NACE_SECTION'] = industry_df['NACE_CLASS_CODE'].astype(str).str[0]
    
    # Map NACE sections to descriptive names
    nace_section_names = {
        'A': 'Agriculture, Forestry and Fishing',
        'B': 'Mining and Quarrying',
        'C': 'Manufacturing',
        'D': 'Electricity, Gas, Steam and Air Conditioning Supply',
        'E': 'Water Supply; Sewerage, Waste Management',
        'F': 'Construction',
        'G': 'Wholesale and Retail Trade',
        'H': 'Transportation and Storage',
        'I': 'Accommodation and Food Service Activities',
        'J': 'Information and Communication',
        'K': 'Financial and Insurance Activities',
        'L': 'Real Estate Activities',
        'M': 'Professional, Scientific and Technical Activities',
        'N': 'Administrative and Support Service Activities',
        'O': 'Public Administration and Defence',
        'P': 'Education',
        'Q': 'Human Health and Social Work Activities',
        'R': 'Arts, Entertainment and Recreation',
        'S': 'Other Service Activities',
        'T': 'Activities of Households as Employers',
        'U': 'Activities of Extraterritorial Organizations'
    }
    
    industry_df['NACE_SECTION_NAME'] = industry_df['NACE_SECTION'].map(nace_section_names)
    
    # 1. NACE section analysis
    print("\n1. EMISSIONS BY NACE SECTION")
    print("-" * 80)
    
    # Calculate metrics by NACE section
    section_metrics = industry_df.groupby('NACE_SECTION_NAME').agg({
        'ISSUERID': 'count',
        'CARBON_EMISSIONS_SCOPE_12': ['sum', 'mean'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
    })
    
    # Calculate percentage of total emissions
    total_emissions = industry_df['CARBON_EMISSIONS_SCOPE_12'].sum()
    section_metrics['emissions_percentage'] = section_metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')] / total_emissions * 100
    
    # Sort by total emissions
    section_metrics = section_metrics.sort_values(('CARBON_EMISSIONS_SCOPE_12', 'sum'), ascending=False)
    
    print("NACE sections by total emissions:")
    for i, (section, metrics) in enumerate(section_metrics.iterrows(), 1):
        if pd.notna(section):
            company_count = float(metrics[('ISSUERID', 'count')])
            total_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')])
            avg_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'mean')])
            intensity = float(metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')])
            trend = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')])
            pct = float(metrics['emissions_percentage'])
            
            print(f"\n  {i}. {section} ({company_count:,.0f} companies):")
            print(f"     - Total Emissions: {total_emissions:,.0f} tons CO2e ({pct:.1f}% of global)")
            print(f"     - Avg per Company: {avg_emissions:,.0f} tons CO2e")
            print(f"     - Intensity: {intensity:.2f}")
            print(f"     - 3-Year Trend: {trend:.2f}%")
    
    # 2. NACE class analysis (more detailed)
    print("\n2. TOP NACE CLASSES BY EMISSIONS")
    print("-" * 80)
    
    # Calculate metrics by NACE class
    class_metrics = industry_df.groupby(['NACE_CLASS_CODE', 'NACE_CLASS_DESCRIPTION']).agg({
        'ISSUERID': 'count',
        'CARBON_EMISSIONS_SCOPE_12': ['sum', 'mean'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
    })
    
    # Calculate percentage of total emissions
    class_metrics['emissions_percentage'] = class_metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')] / total_emissions * 100
    
    # Sort by total emissions
    class_metrics = class_metrics.sort_values(('CARBON_EMISSIONS_SCOPE_12', 'sum'), ascending=False)
    
    print("Top 15 NACE classes by total emissions:")
    for i, (class_info, metrics) in enumerate(class_metrics.head(15).iterrows(), 1):
        nace_code, nace_desc = class_info
        
        company_count = float(metrics[('ISSUERID', 'count')])
        total_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')])
        avg_emissions = float(metrics[('CARBON_EMISSIONS_SCOPE_12', 'mean')])
        intensity = float(metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')])
        trend = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')])
        pct = float(metrics['emissions_percentage'])
        
        print(f"\n  {i}. {nace_code}: {nace_desc} ({company_count:,.0f} companies):")
        print(f"     - Total Emissions: {total_emissions:,.0f} tons CO2e ({pct:.1f}% of global)")
        print(f"     - Avg per Company: {avg_emissions:,.0f} tons CO2e")
        print(f"     - Intensity: {intensity:.2f}")
        print(f"     - 3-Year Trend: {trend:.2f}%")
    
    # 3. Improvement leaders by NACE section
    print("\n3. NACE SECTION IMPROVEMENT LEADERS")
    print("-" * 80)
    
    # Sort sections by improvement rate (lowest/negative trend first)
    improvement_sections = section_metrics.sort_values(('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean'))
    
    print("NACE sections by emissions improvement rate:")
    for i, (section, metrics) in enumerate(improvement_sections.iterrows(), 1):
        if pd.notna(section):
            company_count = float(metrics[('ISSUERID', 'count')])
            if company_count >= 10:  # Only include sections with sufficient data
                trend = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')])
                intensity = float(metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')])
                
                print(f"  {i}. {section} ({company_count:,.0f} companies):")
                print(f"     - 3-Year Trend: {trend:.2f}%")
                print(f"     - Emissions Intensity: {intensity:.2f}")
    
    return section_metrics, class_metrics

def analyze_regulatory_influence(programs2_df, company_emissions_df):
    """
    Analyze how regulatory frameworks influence emissions performance.
    
    Args:
        programs2_df (pd.DataFrame): DataFrame containing regulatory program data
        company_emissions_df (pd.DataFrame): DataFrame with company emissions data
        
    Returns:
        tuple: A tuple containing two DataFrames:
            - Status metrics by regulatory status
            - Regional regulatory status metrics
    """
    print("\n--- REGULATORY ENVIRONMENT INFLUENCE ANALYSIS ---")
    print("=" * 80)
    
    # 1. Prepare regulatory indicator data
    # Use energy audits as proxy for regulatory compliance
    reg_column = 'CBN_REG_ENERGY_AUDITS'
    
    # Group by company to determine dominant regulatory status
    company_reg_status = programs2_df.groupby('ISSUERID')[reg_column].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan
    ).to_dict()
    
    # Create temporary DataFrame with company IDs and regulatory status
    reg_df = pd.DataFrame({
        'ISSUERID': list(company_reg_status.keys()),
        'reg_status': list(company_reg_status.values())
    })
    
    # Drop rows with missing regulatory status
    reg_df = reg_df.dropna(subset=['reg_status'])
    
    # Add country information
    country_info = programs2_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE']].drop_duplicates()
    reg_df = pd.merge(reg_df, country_info, on='ISSUERID', how='left')
    
    # Add region categorization
    reg_df['Region'] = reg_df['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
    
    # 1. Regulatory status distribution
    print("\n1. REGULATORY STATUS DISTRIBUTION")
    print("-" * 80)
    
    # Calculate overall distribution
    status_counts = reg_df['reg_status'].value_counts()
    total = status_counts.sum()
    
    print("Overall distribution of regulatory energy audit status:")
    for status, count in status_counts.items():
        percentage = count / total * 100
        print(f"  - {status}: {count:,} companies ({percentage:.1f}%)")
    
    # 2. Regulatory status by region
    print("\n2. REGULATORY STATUS BY REGION")
    print("-" * 80)
    
    # Calculate status distribution by region
    region_status = pd.crosstab(reg_df['Region'], reg_df['reg_status'], normalize='index') * 100
    
    print("Energy audit regulatory compliance by region (%):")
    for region, row in region_status.iterrows():
        print(f"\n  {region}:")
        for status, percentage in row.items():
            print(f"    - {status}: {percentage:.1f}%")
    
    # 3. Emissions performance by regulatory status
    print("\n3. EMISSIONS PERFORMANCE BY REGULATORY STATUS")
    print("-" * 80)
    
    # Merge with emissions data
    emissions_df = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12', 
                                       'CARBON_EMISSIONS_SCOPE_12_INTEN',
                                       'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()
    
    merged_df = pd.merge(reg_df, emissions_df, on='ISSUERID', how='inner')
    
    # Calculate emissions metrics by regulatory status
    status_metrics = merged_df.groupby('reg_status').agg({
        'ISSUERID': 'count',
        'CARBON_EMISSIONS_SCOPE_12': 'mean',
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'mean',
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean'
    })
    
    # Sort by emissions trend (improving to worsening)
    status_metrics = status_metrics.sort_values('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR')
    
    print("Emissions performance by regulatory energy audit status:")
    for status, metrics in status_metrics.iterrows():
        company_count = float(metrics['ISSUERID'])
        emissions = float(metrics['CARBON_EMISSIONS_SCOPE_12'])
        intensity = float(metrics['CARBON_EMISSIONS_SCOPE_12_INTEN'])
        trend = float(metrics['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
        
        print(f"\n  {status} ({company_count:,.0f} companies):")
        print(f"    - Avg Emissions: {emissions:,.0f} tons CO2e")
        print(f"    - Emissions Intensity: {intensity:.2f}")
        print(f"    - 3-Year Trend: {trend:.2f}%")
    
    # 4. Analysis by specific regulatory environments (top countries)
    print("\n4. TOP COUNTRIES BY REGULATORY COMPLIANCE")
    print("-" * 80)
    
    # Count companies with mandatory energy audits by country
    reg_compliance = reg_df[reg_df['reg_status'] == 'Mandatory energy audits'].groupby('ISSUER_CNTRY_DOMICILE').size()
    total_by_country = reg_df.groupby('ISSUER_CNTRY_DOMICILE').size()
    
    # Calculate compliance percentage
    compliance_pct = (reg_compliance / total_by_country * 100).reset_index()
    compliance_pct.columns = ['Country', 'Compliance_Percentage']
    
    # Filter to countries with at least 10 companies
    compliance_pct = compliance_pct[compliance_pct.index.isin(total_by_country[total_by_country >= 10].index)]
    
    # Sort by compliance percentage
    compliance_pct = compliance_pct.sort_values('Compliance_Percentage', ascending=False)
    
    print("Top 10 countries by mandatory energy audit compliance:")
    for i, (country, percentage) in enumerate(zip(compliance_pct['Country'], compliance_pct['Compliance_Percentage']), 1):
        if i <= 10:
            total = float(total_by_country[country])
            compliant = float(reg_compliance.get(country, 0))
            percentage = float(percentage)
            print(f"  {i}. {country}: {percentage:.1f}% ({compliant:,.0f} of {total:,.0f} companies)")
    
    return status_metrics, region_status

def run_additional_analysis(programs1_df, programs2_df, company_emissions_df, targets_df):
    """
    Execute all additional ESG analysis functions and compile results.
    
    Args:
        programs1_df: DataFrame containing initial program data
        programs2_df: DataFrame containing detailed program data
        company_emissions_df: DataFrame with company emissions data
        targets_df: DataFrame containing emission reduction targets
        
    Returns:
        dict: Dictionary containing results from all analysis modules
    """
    print("\n====== ADDITIONAL ESG INSIGHTS ======")
    print("=" * 80)
    
    # Analyze program type effectiveness
    company_programs = analyze_program_effectiveness(programs2_df, company_emissions_df)
    
    # Analyze regional emissions patterns
    country_metrics, region_metrics = analyze_regional_emissions_patterns(company_emissions_df)
    
    # Analyze industry code patterns
    section_metrics, class_metrics = analyze_industry_code_patterns(company_emissions_df)
    
    # Analyze regulatory influence
    status_metrics, region_status = analyze_regulatory_influence(programs2_df, company_emissions_df)
    
    print("\n====== ANALYSIS COMPLETE ======")
    
    return {
        'company_programs': company_programs,
        'country_metrics': country_metrics,
        'region_metrics': region_metrics,
        'section_metrics': section_metrics,
        'class_metrics': class_metrics,
        'status_metrics': status_metrics,
        'region_status': region_status
    }

def save_results_to_file(programs1_df, programs2_df, company_emissions_df, targets_df):
    """
    Execute the analysis and save the results to a markdown file.
    
    Args:
        programs1_df: DataFrame containing initial program data
        programs2_df: DataFrame containing detailed program data
        company_emissions_df: DataFrame with company emissions data
        targets_df: DataFrame containing emission reduction targets
        
    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes analysis results to a markdown file
        - Prints status messages to console
    """
    # Redirect stdout to capture all print output
    import sys
    original_stdout = sys.stdout
    
    # Create output directory if it doesn't exist
    results_dir = "output/reports"
    os.makedirs(results_dir, exist_ok=True)
    
    # Write analysis output to file
    with open(f"{results_dir}/additional_esg_insights.md", 'w') as f:
        sys.stdout = f
        
        # Run the analysis
        run_additional_analysis(programs1_df, programs2_df, company_emissions_df, targets_df)
        
        # Restore original stdout
        sys.stdout = original_stdout
    
    print(f"Analysis results saved to {results_dir}/additional_esg_insights.md")

if __name__ == "__main__":
    import sys
    
    # Load the data
    programs1_df, programs2_df, company_emissions_df, targets_df = load_data()
    
    # Run the additional analysis and print to console
    results = run_additional_analysis(programs1_df, programs2_df, company_emissions_df, targets_df)
    
    # Also save to file
    save_results_to_file(programs1_df, programs2_df, company_emissions_df, targets_df) 