"""
Advanced ESG analysis for carbon reduction programs and company characteristics.

Input DataFrames:
1. programs1_df:
2. programs2_df:
3. company_emissions_df:

Key Analyses Performed:
1. Market cap relationships with emissions and programs
2. Sales to emissions ratio analysis
3. Industry peer comparison using GICS industry benchmarks
4. Source transparency and disclosure analysis

Outputs:
- Console output with detailed analysis results
- Visualizations of relationships and trends
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
import urllib.parse

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
# Update these paths if your data is stored elsewhere relative to the project root
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
    
    Returns:
        tuple: (programs1_df, programs2_df, company_emissions_df)
    
    Raises:
        FileNotFoundError: If required data files are not found
    """
    try:
        # Load the necessary dataframes
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        programs2_df = pd.read_csv(PROGRAMS2_DATA_PATH).dropna(axis=1, how='all')
        company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        
        # Output dataset information
        print(f"\nPROGRAMS1 DATAFRAME: shape={programs1_df.shape}, unique companies={programs1_df['ISSUERID'].nunique():,}")
        print(f"PROGRAMS2 DATAFRAME: shape={programs2_df.shape}, unique companies={programs2_df['ISSUERID'].nunique():,}")
        print(f"COMPANY_EMISSIONS DATAFRAME: shape={company_emissions_df.shape}, unique companies={company_emissions_df['ISSUERID'].nunique():,}")
        
        return programs1_df, programs2_df, company_emissions_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are present at the specified paths")
        raise

# ===== ANALYSIS FUNCTIONS =====
def analyze_market_cap_relationships(programs2_df, company_emissions_df):
    """
    Analyze how company size (market cap) relates to emissions and program implementation.
    
    Parameters:
    1. programs2_df: Programs2 dataset with program details
    2. company_emissions_df: Company emissions and financial data

    
    Processing Steps:
    1. Merge program and emissions data
    2. Categorize companies by market cap quartiles
    3. Analyze emissions intensity by size
    4. Analyze program implementation by size
    
    Returns:
        DataFrame with market cap categories and analysis results
    """
    print("\n--- MARKET CAP vs EMISSIONS & PROGRAMS ANALYSIS ---")
    print("=" * 80)
    
    # 1. Prepare the data
    # Get program counts per company
    program_counts = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    
    # Get market cap and emissions data
    market_cap_emissions = company_emissions_df[['ISSUERID', 'MarketCap_USD', 
                                                'CARBON_EMISSIONS_SCOPE_12',
                                                'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()
    
    # Merge the datasets
    merged_df = pd.merge(program_counts, market_cap_emissions, on='ISSUERID', how='inner')
    
    # Filter out missing values
    analysis_df = merged_df.dropna(subset=['MarketCap_USD', 'CARBON_EMISSIONS_SCOPE_12'])
    
    print(f"Companies with complete market cap, programs, and emissions data: {len(analysis_df)}")
    
    # 2. Categorize companies by market cap size
    analysis_df['Size_Category'] = pd.qcut(
        analysis_df['MarketCap_USD'], 
        q=4, 
        labels=['Small', 'Medium-Small', 'Medium-Large', 'Large'],
        duplicates='drop'
    )
    
    # 3. Analyze emissions metrics by size category
    print("\n1. EMISSIONS METRICS BY COMPANY SIZE")
    print("-" * 80)
    
    size_metrics = analysis_df.groupby('Size_Category').agg({
        'MarketCap_USD': ['mean', 'median'],
        'CARBON_EMISSIONS_SCOPE_12': ['mean', 'median', 'sum'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': ['mean', 'median'],
        'program_count': ['mean', 'median', 'count']
    })
    
    # Calculate percentage of total emissions by size category
    total_emissions = analysis_df['CARBON_EMISSIONS_SCOPE_12'].sum()
    
    # Print results in a structured way
    for size_cat in ['Small', 'Medium-Small', 'Medium-Large', 'Large']:
        if size_cat in size_metrics.index:
            metrics = size_metrics.loc[size_cat]
            emission_pct = metrics[('CARBON_EMISSIONS_SCOPE_12', 'sum')] / total_emissions * 100
            
            print(f"\n  {size_cat} Companies:")
            print(f"    - Count: {metrics[('program_count', 'count')]:,}")
            print(f"    - Avg Market Cap: ${metrics[('MarketCap_USD', 'mean')]:,.0f}")
            print(f"    - Avg Emissions (Scope 1+2): {metrics[('CARBON_EMISSIONS_SCOPE_12', 'mean')]:,.1f}")
            print(f"    - Emissions Intensity: {metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')]:,.2f}")
            print(f"    - Avg Program Count: {metrics[('program_count', 'mean')]:,.1f}")
            print(f"    - Share of Total Emissions: {emission_pct:.1f}%")
    
    # 4. Correlation analysis
    print("\n2. CORRELATIONS WITH MARKET CAP")
    print("-" * 80)
    
    # Calculate correlations with market cap
    corr_emissions = analysis_df['MarketCap_USD'].corr(analysis_df['CARBON_EMISSIONS_SCOPE_12'])
    corr_intensity = analysis_df['MarketCap_USD'].corr(analysis_df['CARBON_EMISSIONS_SCOPE_12_INTEN'])
    corr_programs = analysis_df['MarketCap_USD'].corr(analysis_df['program_count'])
    
    print("Correlations with company market cap:")
    print(f"  - Absolute Emissions (Scope 1+2): {corr_emissions:.3f}")
    print(f"  - Emissions Intensity: {corr_intensity:.3f}")
    print(f"  - Program Count: {corr_programs:.3f}")
    
    # 5. Efficiency analysis (programs per $ market cap)
    print("\n3. PROGRAM EFFICIENCY BY SIZE")
    print("-" * 80)
    
    # Calculate programs per billion $ market cap
    analysis_df['programs_per_billion'] = analysis_df['program_count'] / (analysis_df['MarketCap_USD'] / 1e9)
    
    size_efficiency = analysis_df.groupby('Size_Category')['programs_per_billion'].agg(['mean', 'median'])
    
    print("Programs per billion $ market cap by company size:")
    for size_cat, metrics in size_efficiency.iterrows():
        print(f"  - {size_cat}: {metrics['mean']:.2f} programs per $B (median: {metrics['median']:.2f})")
    
    return analysis_df

def analyze_sales_emissions_relationship(company_emissions_df):
    """
    Analyze relationship between company sales and emissions metrics.
    
    Parameters:
    1. company_emissions_df: Company emissions and financial data
    
    Processing Steps:
    1. Calculate sales to emissions ratio
    2. Analyze distribution across industries
    3. Identify outliers and trends
    
    Returns:
        DataFrame with calculated sales to emissions metrics
    """
    print("\n--- SALES vs EMISSIONS INTENSITY ANALYSIS ---")
    print("=" * 80)
    
    # 1. Prepare the data
    # Filter to companies with both sales and emissions data
    analysis_df = company_emissions_df[['ISSUERID', 'SALES_USD_RECENT', 
                                      'CARBON_EMISSIONS_SCOPE_12', 
                                      'CARBON_EMISSIONS_SCOPE_12_INTEN',
                                      'NACE_CLASS_DESCRIPTION']].drop_duplicates()
    
    # Filter out missing values
    analysis_df = analysis_df.dropna(subset=['SALES_USD_RECENT', 'CARBON_EMISSIONS_SCOPE_12_INTEN'])
    
    # Add industry group
    analysis_df['Industry_Group'] = analysis_df['NACE_CLASS_DESCRIPTION'].apply(categorize_industry)
    
    print(f"Companies with both sales and emissions intensity data: {len(analysis_df)}")
    
    # 2. Sales quartile analysis
    print("\n1. EMISSIONS INTENSITY BY SALES QUARTILE")
    print("-" * 80)
    
    # Categorize companies by sales quartiles
    analysis_df['Sales_Category'] = pd.qcut(
        analysis_df['SALES_USD_RECENT'], 
        q=4, 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
        duplicates='drop'
    )
    
    # Analyze emissions intensity by sales category
    sales_metrics = analysis_df.groupby('Sales_Category').agg({
        'SALES_USD_RECENT': ['mean', 'median'],
        'CARBON_EMISSIONS_SCOPE_12_INTEN': ['mean', 'median'],
        'ISSUERID': 'count'
    })
    
    # Print results
    for sales_cat in ['Low', 'Medium-Low', 'Medium-High', 'High']:
        if sales_cat in sales_metrics.index:
            metrics = sales_metrics.loc[sales_cat]
            
            print(f"\n  {sales_cat} Sales Companies:")
            print(f"    - Count: {metrics[('ISSUERID', 'count')]:,}")
            print(f"    - Avg Sales: ${metrics[('SALES_USD_RECENT', 'mean')]:,.0f}")
            print(f"    - Emissions Intensity: {metrics[('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')]:,.2f}")
    
    # 3. Correlation analysis
    print("\n2. CORRELATION BETWEEN SALES AND EMISSIONS")
    print("-" * 80)
    
    # Calculate correlations
    corr_intensity = analysis_df['SALES_USD_RECENT'].corr(analysis_df['CARBON_EMISSIONS_SCOPE_12_INTEN'])
    corr_absolute = analysis_df['SALES_USD_RECENT'].corr(analysis_df['CARBON_EMISSIONS_SCOPE_12'])
    
    print("Correlations with company sales:")
    print(f"  - Absolute Emissions (Scope 1+2): {corr_absolute:.3f}")
    print(f"  - Emissions Intensity: {corr_intensity:.3f}")
    
    # 4. Industry-specific analysis
    print("\n3. SALES-EMISSIONS RELATIONSHIP BY INDUSTRY")
    print("-" * 80)
    
    # Calculate correlations by industry group
    industry_corrs = {}
    top_industries = analysis_df['Industry_Group'].value_counts().head(5).index
    
    for industry in top_industries:
        industry_df = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_df) >= 10:  # Only calculate if we have enough data
            corr = industry_df['SALES_USD_RECENT'].corr(industry_df['CARBON_EMISSIONS_SCOPE_12_INTEN'])
            industry_corrs[industry] = corr
    
    print("Sales-Emissions Intensity correlation by industry:")
    for industry, corr in industry_corrs.items():
        count = len(analysis_df[analysis_df['Industry_Group'] == industry])
        print(f"  - {industry} ({count:,} companies): {corr:.3f}")
    
    # 5. Carbon efficiency metric (sales per unit of emissions)
    print("\n4. CARBON EFFICIENCY (SALES PER EMISSION)")
    print("-" * 80)
    
    # Calculate sales per unit of emissions (carbon efficiency), handling division by zero
    analysis_df['sales_per_emission'] = analysis_df.apply(
        lambda row: row['SALES_USD_RECENT'] / row['CARBON_EMISSIONS_SCOPE_12'] 
        if row['CARBON_EMISSIONS_SCOPE_12'] > 0 else np.nan, 
        axis=1
    )
    
    # Drop infinite values and convert to numeric
    analysis_df['sales_per_emission'] = pd.to_numeric(analysis_df['sales_per_emission'], errors='coerce')
    
    # Group by industry and calculate average carbon efficiency
    industry_efficiency = analysis_df.groupby('Industry_Group')['sales_per_emission'].agg(['mean', 'median', 'count'])
    
    # Sort by mean efficiency, handling NaN values
    industry_efficiency = industry_efficiency.sort_values('mean', ascending=False, na_position='last')
    
    print("Sales per unit of emissions ($ per ton CO2e) by industry:")
    for industry, metrics in industry_efficiency.iterrows():
        if metrics['count'] >= 10:  # Only show industries with enough data
            mean_value = metrics['mean']
            if pd.isna(mean_value):
                print(f"  - {industry} ({metrics['count']:,} companies): insufficient data")
            else:
                print(f"  - {industry} ({metrics['count']:,} companies): ${mean_value:,.0f} per ton CO2e")
    
    return analysis_df

def analyze_industry_peer_comparison(programs2_df, company_emissions_df):
    """
    Compare companies to their industry peers based on GICS industry classification.
    
    Parameters:
    1. programs2_df: Programs2 dataset with program details
    2. company_emissions_df: Company emissions and industry data
    
    Processing Steps:
    1. Group companies by GICS industry
    2. Calculate industry benchmarks for emissions and program metrics
    3. Compare companies to their industry peers
    
    Returns:
        DataFrame with industry benchmarks and company comparisons
    """
    print("\n--- INDUSTRY PEER COMPARISON ANALYSIS ---")
    print("=" * 80)
    
    # 1. Prepare the data
    # Focus on the peer comparison ratio
    peer_df = company_emissions_df[['ISSUERID', 'ISSUER_NAME_company', 'ISSUER_CNTRY_DOMICILE_emissions',
                                  'NACE_CLASS_DESCRIPTION', 'CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO',
                                  'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()
    
    # Filter to companies with peer ratio data
    peer_df = peer_df.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO'])
    
    # Add region and industry group
    peer_df['Region'] = peer_df['ISSUER_CNTRY_DOMICILE_emissions'].apply(categorize_region)
    peer_df['Industry_Group'] = peer_df['NACE_CLASS_DESCRIPTION'].apply(categorize_industry)
    
    print(f"Companies with industry peer comparison data: {len(peer_df)}")
    
    # 2. Distribution of peer ratios
    print("\n1. DISTRIBUTION OF INDUSTRY PEER RATIOS")
    print("-" * 80)
    
    # Calculate percentiles
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    peer_stats = peer_df['CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO'].describe(percentiles=percentiles)
    
    print("Distribution of peer comparison ratios:")
    print(f"  - Average: {peer_stats['mean']:.2f}")
    print(f"  - Median: {peer_stats['50%']:.2f}")
    print(f"  - 10th percentile (best performers): {peer_stats['10%']:.2f}")
    print(f"  - 90th percentile (worst performers): {peer_stats['90%']:.2f}")
    
    # 3. Performance categories
    print("\n2. COMPANIES BY PERFORMANCE CATEGORY")
    print("-" * 80)
    
    # Categorize companies by their peer ratio
    peer_df['Performance_Category'] = pd.cut(
        peer_df['CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO'],
        bins=[0, 0.5, 0.8, 1.2, 2, float('inf')],
        labels=['Industry Leader', 'Better than Peers', 'Average', 'Below Average', 'Significant Underperformer']
    )
    
    # Count companies in each category
    performance_counts = peer_df['Performance_Category'].value_counts().sort_index()
    
    print("Companies by peer performance category:")
    for category, count in performance_counts.items():
        percentage = count / len(peer_df) * 100
        print(f"  - {category}: {count:,} companies ({percentage:.1f}%)")
    
    # 4. Regional comparison
    print("\n3. PEER PERFORMANCE BY REGION")
    print("-" * 80)
    
    # First, create a helper column for leaders
    peer_df['is_leader'] = peer_df['Performance_Category'] == 'Industry Leader'
    
    # Calculate average peer ratio and leaders by region
    region_metrics = peer_df.groupby('Region').agg({
        'CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO': ['mean', 'median', 'count'],
        'is_leader': 'sum'  # Count of leaders
    })
    
    # Calculate percentage of leaders
    region_metrics['pct_leaders'] = region_metrics[('is_leader', 'sum')] / region_metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')] * 100
    
    # Sort by average performance
    region_metrics = region_metrics.sort_values(('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'mean'))
    
    print("Peer performance by region (sorted from best to worst):")
    for region, metrics in region_metrics.iterrows():
        if metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')] >= 10:
            company_count = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')])
            avg_ratio = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'mean')])
            leader_count = float(metrics[('is_leader', 'sum')])
            leader_pct = float(metrics['pct_leaders'])
            
            print(f"\n  {region}:")
            print(f"    - Companies: {company_count:,.0f}")
            print(f"    - Avg Peer Ratio: {avg_ratio:.2f}")
            print(f"    - Industry Leaders: {leader_count:.0f} companies ({leader_pct:.1f}%)")
    
    # 5. Industry comparison
    print("\n4. PEER PERFORMANCE BY INDUSTRY")
    print("-" * 80)
    
    # Calculate average peer ratio and leaders by industry
    industry_metrics = peer_df.groupby('Industry_Group').agg({
        'CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO': ['mean', 'median', 'count'],
        'is_leader': 'sum'  # Count of leaders
    })
    
    # Calculate percentage of leaders
    industry_metrics['pct_leaders'] = industry_metrics[('is_leader', 'sum')] / industry_metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')] * 100
    
    # Sort by average performance
    industry_metrics = industry_metrics.sort_values(('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'mean'))
    
    print("Peer performance by industry (sorted from best to worst):")
    for industry, metrics in industry_metrics.iterrows():
        if metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')] >= 10:
            company_count = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'count')])
            avg_ratio = float(metrics[('CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO', 'mean')])
            leader_count = float(metrics[('is_leader', 'sum')])
            leader_pct = float(metrics['pct_leaders'])
            
            print(f"\n  {industry}:")
            print(f"    - Companies: {company_count:,.0f}")
            print(f"    - Avg Peer Ratio: {avg_ratio:.2f}")
            print(f"    - Industry Leaders: {leader_count:.0f} companies ({leader_pct:.1f}%)")
    
    return peer_df

def analyze_source_transparency(programs2_df):
    """
    Analyze the transparency of emissions sources and program disclosures.
    
    Parameters:
    1. programs2_df: Programs2 dataset with transparency indicators
    
    Processing Steps:
    1. Analyze disclosure levels by program type
    2. Examine verification status and methods
    3. Identify transparency gaps and patterns
    4. Generate visualizations of transparency metrics
    
    Returns:
        DataFrame with transparency analysis results
    """
    print("\n--- SOURCE URL TRANSPARENCY ANALYSIS ---")
    print("=" * 80)
    
    # 1. Identify source columns and their availability
    source_columns = [col for col in programs2_df.columns if 'SOURCE' in col]
    
    # Calculate availability rate for each source column
    print("\n1. SOURCE DATA AVAILABILITY")
    print("-" * 80)
    
    source_availability = {}
    for col in source_columns:
        non_null = programs2_df[col].count()
        availability_pct = non_null / len(programs2_df) * 100
        clean_name = col.replace('CBN_', '').replace('_SOURCE', '').replace('SOURCE_', '').replace('_', ' ').title()
        source_availability[clean_name] = (non_null, availability_pct)
    
    print("Source URL availability by program type:")
    for source_type, (count, pct) in sorted(source_availability.items(), key=lambda x: x[1][1], reverse=True):
        print(f"  - {source_type}: {count:,} programs ({pct:.1f}%)")
    
    # 2. Company-level source transparency analysis
    print("\n2. COMPANY-LEVEL SOURCE TRANSPARENCY")
    print("-" * 80)
    
    # For each company, calculate the percentage of programs with sources
    company_sources = programs2_df.groupby('ISSUERID').apply(
        lambda x: {col: x[col].count() / len(x) * 100 for col in source_columns}
    )
    
    # Calculate overall transparency score (average of all source percentages)
    transparency_scores = []
    for _, sources in company_sources.items():
        if sources:  # Skip empty dictionaries
            score = sum(sources.values()) / len(sources)
            transparency_scores.append(score)
    
    # Calculate distribution of transparency scores
    if transparency_scores:
        print("Distribution of company transparency scores (percentage of programs with sources):")
        transparency_pcts = [10, 25, 50, 75, 90]
        for pct in transparency_pcts:
            score = np.percentile(transparency_scores, pct)
            print(f"  - {pct}th percentile: {score:.1f}%")
        
        full_transparency = sum(1 for score in transparency_scores if score >= 95)
        full_pct = full_transparency / len(transparency_scores) * 100
        print(f"  - Companies with ≥95% transparency: {full_transparency:,} ({full_pct:.1f}%)")
    
    # 3. Domain analysis
    print("\n3. SOURCE URL DOMAIN ANALYSIS")
    print("-" * 80)
    
    # Extract domains from source URLs
    all_domains = []
    for col in source_columns:
        urls = programs2_df[col].dropna()
        
        # Extract domain from each URL
        for url in urls:
            try:
                domain = urllib.parse.urlparse(url).netloc
                if domain:
                    all_domains.append(domain)
            except:
                pass  # Skip invalid URLs
    
    # Count domain frequencies
    domain_counts = Counter(all_domains)
    
    print(f"Total URLs analyzed: {len(all_domains):,}")
    print(f"Unique domains found: {len(domain_counts):,}")
    
    print("\nTop 15 domains used for source URLs:")
    for domain, count in domain_counts.most_common(15):
        percentage = count / len(all_domains) * 100
        print(f"  - {domain}: {count:,} occurrences ({percentage:.1f}%)")
    
    # 4. Regional transparency comparison
    print("\n4. TRANSPARENCY BY REGION")
    print("-" * 80)
    
    # Add region to programs2_df
    programs2_with_region = programs2_df.copy()
    programs2_with_region['Region'] = programs2_with_region['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
    
    # For each region, calculate the percentage of programs with sources
    region_transparency = {}
    for region in programs2_with_region['Region'].unique():
        region_df = programs2_with_region[programs2_with_region['Region'] == region]
        
        # Calculate source availability for this region
        source_pcts = {}
        for col in source_columns:
            non_null = region_df[col].count()
            total = len(region_df)
            if total > 0:
                source_pcts[col] = non_null / total * 100
        
        # Calculate average transparency
        if source_pcts:
            avg_transparency = sum(source_pcts.values()) / len(source_pcts)
            region_transparency[region] = (avg_transparency, len(region_df))
    
    print("Source transparency by region:")
    for region, (transparency, count) in sorted(region_transparency.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  - {region}: {transparency:.1f}% source availability ({count:,} programs)")
    
    return source_availability, domain_counts

def run_advanced_esg_analysis(programs1_df, programs2_df, company_emissions_df):
    """
    Run comprehensive advanced ESG analysis.
    
    Parameters:
    1. programs1_df: First program dataset with mitigation strategies
    2. programs2_df: Second program dataset with implementation details
    3. company_emissions_df: Company emissions and financial data
    
    Processing Steps:
    1. Run all analysis components in sequence
    2. Generate visualizations for each analysis
    3. Compile results into a structured format
    4. Save analysis outputs to files
    
    Returns:
        dict with all analysis results and visualizations
    """
    print("\n====== ADVANCED ESG ANALYSIS ======")
    print("=" * 80)
    
    # Run the market cap analysis
    market_cap_df = analyze_market_cap_relationships(programs2_df, company_emissions_df)
    
    # Run the sales vs emissions analysis
    sales_emissions_df = analyze_sales_emissions_relationship(company_emissions_df)
    
    # Run the industry peer comparison analysis
    peer_comparison_df = analyze_industry_peer_comparison(programs2_df, company_emissions_df)
    
    # Run the source URL transparency analysis
    source_availability, domain_counts = analyze_source_transparency(programs2_df)
    
    print("\n====== ANALYSIS COMPLETE ======")
    
    return {
        'market_cap_analysis': market_cap_df,
        'sales_emissions_analysis': sales_emissions_df,
        'peer_comparison_analysis': peer_comparison_df,
        'source_availability': source_availability,
        'domain_counts': domain_counts
    }

def save_analysis_results(results, output_dir='output/reports'):
    """
    Save advanced ESG analysis results to files.
    
    Parameters:
    1. results: Dictionary containing analysis results from run_advanced_esg_analysis
    2. output_dir: Directory to save results (default: 'output/reports')
    
    Returns:
        str: Path to saved results directory
    """
    import os
    import pandas as pd
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each result to a separate CSV file
    for key, df in results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(f"{output_dir}/{key}.csv", index=False)
    
    # Save a markdown report
    with open(f"{output_dir}/advanced_esg_analysis.md", 'w') as f:
        f.write("# Advanced ESG Analysis Report\n\n")
        f.write("## Summary of Results\n\n")
        
        # Add a section for each result
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                f.write(f"### {key.replace('_', ' ').title()}\n\n")
                f.write(f"Shape: {value.shape}\n\n")
                f.write("First 5 rows:\n\n")
                f.write(value.head().to_markdown() + "\n\n")
    
    print(f"Analysis results saved to {output_dir}/")
    return output_dir

if __name__ == "__main__":
    import sys
    
    # Load the data
    programs1_df, programs2_df, company_emissions_df = load_data()
    
    # Run the advanced analysis and print to console
    results = run_advanced_esg_analysis(programs1_df, programs2_df, company_emissions_df)
    
    # Also save to file
    save_analysis_results(results, output_dir='output/reports')