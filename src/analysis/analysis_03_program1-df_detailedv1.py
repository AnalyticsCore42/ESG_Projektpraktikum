"""
Comprehensive visualization and analysis of corporate carbon mitigation strategies.

Input DataFrames:
1. programs1_df: Contains carbon reduction program data with mitigation strategies
2. company_emissions_df: Contains company emissions data and metadata
3. company_data: Basic company information and characteristics
4. emissions_data: Detailed emissions metrics
5. targets_data: Carbon reduction targets information
6. programs2_data: Additional program data for cross-referencing

Key Analyses Performed:
1. Distribution of mitigation strategies across different company segments
2. Effectiveness analysis of various carbon reduction approaches
3. Impact of company characteristics (size, region, industry) on strategy implementation
4. Combined mitigation score calculation and visualization
5. Detailed breakdown of specific mitigation categories (distribution, raw materials, manufacturing, transport, carbon capture)

Outputs:
- Generates 24 visualization plots saved in the output directory
- Console output with analysis summaries and statistics
- Combined analysis results as a structured dictionary
"""

# ===== IMPORTS =====
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

from src.config import paths
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

# ===== VISUALIZATION SETUP =====
print("\n--- VISUALIZATION SETUP ---")
print("=" * 80)
print("Configuring visualization styles and fonts...")

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
print("✓ Visualization styles configured")

# ===== FILE PATHS =====
print("\n--- DATA CONFIGURATION ---")
print("=" * 80)
print("Setting up data file paths...")

# Data file paths
COMPANY_DATA_PATH = "data/Unternehmensdaten Results - 20241212 15_41_41.csv"
EMISSIONS_DATA_PATH = "data/Treibhausgasemissionen Results - 20241212 15_44_03.csv"
TARGETS_DATA_PATH = "data/Reduktionsziele Results - 20241212 15_49_29.csv"
PROGRAMS1_DATA_PATH = "data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
PROGRAMS2_DATA_PATH = "data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
COMPANY_EMISSIONS_PATH = "data/company_emissions_merged.csv"
print("✓ Data file paths configured")

# ===== VISUALIZATION STYLES =====
print("\n--- COLOR CONFIGURATION ---")
print("=" * 80)
print("Setting up color schemes for visualizations...")

# Industry colors - consistent across all X.3 plots
industry_colors = {
    'Manufacturing': '#1f77b4',  # blue
    'Technology': '#2ca02c',     # green
    'Financial Services': '#ff7f0e',  # orange
    'Energy & Utilities': '#d62728',  # red
    'Retail & Wholesale': '#e377c2',  # pink
    'Transportation': '#9467bd',  # purple
    'Extractive': '#17becf'      # cyan
}
print(f"✓ Defined {len(industry_colors)} industry colors")

# Region colors - consistent across all X.2 plots
region_colors = {
    'Europe': '#006400',         # dark green
    'North America': '#00008B',  # dark blue
    'Asia-Pacific': '#8c564b',   # brown
    'South America': '#bcbd22',  # olive
    'Africa': '#FFD700',         # yellow
    'Middle East': '#555555',    # darker grey
    'Others': '#ff7f0e'          # orange
}
print(f"✓ Defined {len(region_colors)} region colors")

# ===== ANALYSIS FUNCTIONS =====
print("\n--- ANALYSIS FUNCTIONS ---")
print("=" * 80)

def analyze_comprehensive_mitigation_data(programs1_df, company_emissions_df):
    """
    Analyze and structure mitigation data for comprehensive visualization.
    
    Parameters:
    1. programs1_df (DataFrame): Contains program 1 data with mitigation strategies
       - Required columns: ISSUERID, CBN_GHG_MITIG_* (various strategy columns)
    2. company_emissions_df (DataFrame): Contains company emissions and metadata
       - Required columns: ISSUERID, CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR, 
         CARBON_EMISSIONS_SCOPE_12_INTEN, MarketCap_USD, NACE_CLASS_DESCRIPTION
    
    Processing Steps:
    1. Merge program and emissions data
    2. Apply company segmentation (size, region, industry)
    3. Map mitigation strategies to standardized categories
    4. Calculate combined mitigation scores
    5. Structure results for visualization
    
    Returns:
        dict: Nested dictionary with analysis results, structured by:
            - Category (e.g., 'distribution', 'raw_materials')
            - Breakdown (e.g., 'by_industry', 'by_region')
            - Data for visualization
    """
    
    # Merge the datasets
    analysis_df = pd.merge(
        programs1_df,
        company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'CARBON_EMISSIONS_SCOPE_12_INTEN', 
                            'MarketCap_USD', 'NACE_CLASS_DESCRIPTION']],
        on='ISSUERID',
        how='inner'
    )
    
    # Ensure country data is present
    if 'ISSUER_CNTRY_DOMICILE' not in analysis_df.columns:
        if 'ISSUER_CNTRY_DOMICILE' in programs1_df.columns:
            analysis_df['ISSUER_CNTRY_DOMICILE'] = programs1_df.set_index('ISSUERID').loc[analysis_df['ISSUERID'], 'ISSUER_CNTRY_DOMICILE'].values
        elif 'ISSUER_CNTRY_DOMICILE' in company_emissions_df.columns:
            country_data = company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE']].drop_duplicates()
            analysis_df = pd.merge(analysis_df, country_data, on='ISSUERID', how='left')
    
    # Apply segmentation using our external functions
    analysis_df['Size_Category'] = analysis_df['MarketCap_USD'].apply(categorize_size)
    if 'ISSUER_CNTRY_DOMICILE' in analysis_df.columns:
        analysis_df['Region'] = analysis_df['ISSUER_CNTRY_DOMICILE'].apply(categorize_region)
    analysis_df['Industry_Group'] = analysis_df['NACE_CLASS_DESCRIPTION'].apply(categorize_industry)
    
    # Define mitigation strategy columns
    mitigation_columns = [
        'CBN_GHG_MITIG_DISTRIBUTION', 
        'CBN_GHG_MITIG_RAW_MAT', 
        'CBN_GHG_MITIG_MFG', 
        'CBN_GHG_MITIG_TRANSPORT', 
        'CBN_GHG_MITIG_CAPTURE'
    ]
    
    # Create simple category mappings for analysis
    analysis_df['Distribution_Simple'] = analysis_df['CBN_GHG_MITIG_DISTRIBUTION'].apply(map_distribution_category)
    analysis_df['RAW_MAT_Simple'] = analysis_df['CBN_GHG_MITIG_RAW_MAT'].apply(map_raw_materials_category)
    analysis_df['MFG_Simple'] = analysis_df['CBN_GHG_MITIG_MFG'].apply(map_manufacturing_category)
    analysis_df['TRANSPORT_Simple'] = analysis_df['CBN_GHG_MITIG_TRANSPORT'].apply(map_transport_category)
    analysis_df['CAPTURE_Simple'] = analysis_df['CBN_GHG_MITIG_CAPTURE'].apply(map_capture_category)
    
    # Calculate combined mitigation score
    mitigation_score_maps = get_mitigation_score_maps()
    
    # Extract unique ISSUERIDs
    unique_issuers = analysis_df['ISSUERID'].unique()
    
    # Create a new dataframe for combined analysis (one row per company)
    combined_df = pd.DataFrame({'ISSUERID': unique_issuers})
    
    # Calculate scores for each strategy by company
    for col in mitigation_columns:
        # Map original categories to scores directly
        score_map = mitigation_score_maps[col]
        analysis_df[f'{col}_Score'] = analysis_df[col].map(lambda x: score_map.get(x, 0))
        
        # For each company, get the maximum score across all rows (in case of duplicates)
        company_scores = analysis_df.groupby('ISSUERID')[f'{col}_Score'].max()
        combined_df = pd.merge(combined_df, pd.DataFrame({f'{col}_Score': company_scores}), 
                              on='ISSUERID', how='left')
    
    # Calculate the combined mitigation score
    score_columns = [f'{col}_Score' for col in mitigation_columns]
    combined_df['Combined_Mitigation_Score'] = combined_df[score_columns].mean(axis=1)
    
    # Bin the combined score into 5 categories with adjusted thresholds
    bins = [0, 1.0, 2.0, 3.0, 4.5, 6.1]  # 5 categories from 0 to 6
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    combined_df['Combined_Mitigation_Level'] = pd.cut(combined_df['Combined_Mitigation_Score'], 
                                                    bins=bins, labels=labels)
    
    # Merge combined scores back with other data
    analysis_df = pd.merge(
        analysis_df,
        combined_df[['ISSUERID', 'Combined_Mitigation_Score', 'Combined_Mitigation_Level']],
        on='ISSUERID',
        how='left'
    )
    
    # Create the structure for our comprehensive results
    results = {
        'combined': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        },
        'distribution': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        },
        'raw_materials': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        },
        'transport': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        },
        'capture': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        },
        'manufacturing': {
            'base': {},
            'by_size': {},
            'by_region': {},
            'by_industry': {}
        }
    }
    
    # --- COMBINED MITIGATION ANALYSIS ---
    # 1.1 Base analysis
    combined_summary = analysis_df.groupby('Combined_Mitigation_Level').agg(
        count=('ISSUERID', 'nunique'),
        emission_trend=('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean'),
        intensity=('CARBON_EMISSIONS_SCOPE_12_INTEN', 'mean')
    )
    
    results['combined']['base'] = {
        'categories': combined_summary.index.tolist(),
        'emission_trends': combined_summary['emission_trend'].tolist(),
        'counts': combined_summary['count'].tolist(),
        'intensities': combined_summary['intensity'].tolist()
    }
    
    # 1.2 By company size
    size_order = ['Mega-cap', 'Large-cap', 'Mid-cap', 'Small-cap', 'Micro-cap']
    size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            size_summary = size_data.groupby('Combined_Mitigation_Level').agg(
                count=('ISSUERID', 'nunique'),
                emission_trend=('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')
            )
            if not size_summary.empty:
                size_trends[size] = {
                    'categories': size_summary.index.tolist(),
                    'emission_trends': size_summary['emission_trend'].tolist(),
                    'counts': size_summary['count'].tolist()
                }
    
    results['combined']['by_size'] = size_trends
    
    # 1.3 By region
    region_list = ['Europe', 'North America', 'Asia-Pacific', 'South America', 'Africa', 'Middle East']
    region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            region_summary = region_data.groupby('Combined_Mitigation_Level').agg(
                count=('ISSUERID', 'nunique'),
                emission_trend=('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')
            )
            if not region_summary.empty:
                region_trends[region] = {
                    'categories': region_summary.index.tolist(),
                    'emission_trends': region_summary['emission_trend'].tolist(),
                    'counts': region_summary['count'].tolist()
                }
    
    results['combined']['by_region'] = region_trends
    
    # 1.4 By industry
    industry_list = ['Manufacturing', 'Technology', 'Financial Services', 'Energy & Utilities', 
                    'Retail & Wholesale', 'Transportation', 'Extractive']
    industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            industry_summary = industry_data.groupby('Combined_Mitigation_Level').agg(
                count=('ISSUERID', 'nunique'),
                emission_trend=('CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 'mean')
            )
            if not industry_summary.empty:
                industry_trends[industry] = {
                    'categories': industry_summary.index.tolist(),
                    'emission_trends': industry_summary['emission_trend'].tolist(),
                    'counts': industry_summary['count'].tolist()
                }
    
    results['combined']['by_industry'] = industry_trends
    
    # --- DISTRIBUTION STRATEGIES ANALYSIS ---
    # 2.1 Base analysis
    distribution_categories = ['No', 'General statement', 'Some stores/distribution centers (anecdotal cases)', 
                              'All or most stores and distribution centers']
    display_distribution_categories = ['No', 'General statement', 'Partial', 'Comprehensive']
    
    distribution_summary = pd.DataFrame()
    for cat, display_cat in zip(distribution_categories, display_distribution_categories):
        subset = analysis_df[analysis_df['CBN_GHG_MITIG_DISTRIBUTION'] == cat]
        if len(subset) > 0:
            distribution_summary.loc[display_cat, 'count'] = len(subset)
            distribution_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            distribution_summary.loc[display_cat, 'intensity'] = subset['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
    
    results['distribution']['base'] = {
        'categories': display_distribution_categories,
        'emission_trends': distribution_summary['emission_trend'].tolist(),
        'counts': distribution_summary['count'].tolist(),
        'intensities': distribution_summary['intensity'].tolist()
    }
    
    # 2.2 By company size
    dist_size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            dist_size_summary = pd.DataFrame()
            for cat, display_cat in zip(distribution_categories, display_distribution_categories):
                subset = size_data[size_data['CBN_GHG_MITIG_DISTRIBUTION'] == cat]
                if len(subset) > 0:
                    dist_size_summary.loc[display_cat, 'count'] = len(subset)
                    dist_size_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not dist_size_summary.empty:
                dist_size_trends[size] = {
                    'categories': dist_size_summary.index.tolist(),
                    'emission_trends': dist_size_summary['emission_trend'].tolist(),
                    'counts': dist_size_summary['count'].tolist()
                }
    
    results['distribution']['by_size'] = dist_size_trends
    
    # 2.3 By region
    dist_region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            dist_region_summary = pd.DataFrame()
            for cat, display_cat in zip(distribution_categories, display_distribution_categories):
                subset = region_data[region_data['CBN_GHG_MITIG_DISTRIBUTION'] == cat]
                if len(subset) > 0:
                    dist_region_summary.loc[display_cat, 'count'] = len(subset)
                    dist_region_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not dist_region_summary.empty:
                dist_region_trends[region] = {
                    'categories': dist_region_summary.index.tolist(),
                    'emission_trends': dist_region_summary['emission_trend'].tolist(),
                    'counts': dist_region_summary['count'].tolist()
                }
    
    results['distribution']['by_region'] = dist_region_trends
    
    # 2.4 By industry
    dist_industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            dist_industry_summary = pd.DataFrame()
            for cat, display_cat in zip(distribution_categories, display_distribution_categories):
                subset = industry_data[industry_data['CBN_GHG_MITIG_DISTRIBUTION'] == cat]
                if len(subset) > 0:
                    dist_industry_summary.loc[display_cat, 'count'] = len(subset)
                    dist_industry_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not dist_industry_summary.empty:
                dist_industry_trends[industry] = {
                    'categories': dist_industry_summary.index.tolist(),
                    'emission_trends': dist_industry_summary['emission_trend'].tolist(),
                    'counts': dist_industry_summary['count'].tolist()
                }
    
    results['distribution']['by_industry'] = dist_industry_trends
    
    # --- RAW MATERIALS STRATEGIES ANALYSIS ---
    # 3.1 Base analysis
    raw_mat_categories = ['No', 'Some products (anecdotal cases)', 'General statement', 'All or core products']
    display_raw_mat_categories = ['No', 'Some products', 'General statement', 'All or core products']
    
    raw_mat_summary = pd.DataFrame()
    for cat, display_cat in zip(raw_mat_categories, display_raw_mat_categories):
        subset = analysis_df[analysis_df['CBN_GHG_MITIG_RAW_MAT'] == cat]
        if len(subset) > 0:
            raw_mat_summary.loc[display_cat, 'count'] = len(subset)
            raw_mat_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            raw_mat_summary.loc[display_cat, 'intensity'] = subset['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
    
    results['raw_materials']['base'] = {
        'categories': display_raw_mat_categories,
        'emission_trends': raw_mat_summary['emission_trend'].tolist(),
        'counts': raw_mat_summary['count'].tolist(),
        'intensities': raw_mat_summary['intensity'].tolist()
    }
    
    # 3.2 By company size
    raw_size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            raw_size_summary = pd.DataFrame()
            for cat, display_cat in zip(raw_mat_categories, display_raw_mat_categories):
                subset = size_data[size_data['CBN_GHG_MITIG_RAW_MAT'] == cat]
                if len(subset) > 0:
                    raw_size_summary.loc[display_cat, 'count'] = len(subset)
                    raw_size_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not raw_size_summary.empty:
                raw_size_trends[size] = {
                    'categories': raw_size_summary.index.tolist(),
                    'emission_trends': raw_size_summary['emission_trend'].tolist(),
                    'counts': raw_size_summary['count'].tolist()
                }
    
    results['raw_materials']['by_size'] = raw_size_trends
    
    # 3.3 By region
    raw_region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            raw_region_summary = pd.DataFrame()
            for cat, display_cat in zip(raw_mat_categories, display_raw_mat_categories):
                subset = region_data[region_data['CBN_GHG_MITIG_RAW_MAT'] == cat]
                if len(subset) > 0:
                    raw_region_summary.loc[display_cat, 'count'] = len(subset)
                    raw_region_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not raw_region_summary.empty:
                raw_region_trends[region] = {
                    'categories': raw_region_summary.index.tolist(),
                    'emission_trends': raw_region_summary['emission_trend'].tolist(),
                    'counts': raw_region_summary['count'].tolist()
                }
    
    results['raw_materials']['by_region'] = raw_region_trends
    
    # 3.4 By industry
    raw_industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            raw_industry_summary = pd.DataFrame()
            for cat, display_cat in zip(raw_mat_categories, display_raw_mat_categories):
                subset = industry_data[industry_data['CBN_GHG_MITIG_RAW_MAT'] == cat]
                if len(subset) > 0:
                    raw_industry_summary.loc[display_cat, 'count'] = len(subset)
                    raw_industry_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not raw_industry_summary.empty:
                raw_industry_trends[industry] = {
                    'categories': raw_industry_summary.index.tolist(),
                    'emission_trends': raw_industry_summary['emission_trend'].tolist(),
                    'counts': raw_industry_summary['count'].tolist()
                }
    
    results['raw_materials']['by_industry'] = raw_industry_trends
    
    # --- TRANSPORT STRATEGIES ANALYSIS ---
    # 4.1 Base analysis
    transport_categories = ['No', 'General statement', 
                          'Improvements in fleet, routes, OR load/packaging optimization', 
                          'Improvements in fleet, routes, AND load/packaging optimization']
    display_transport_categories = ['No', 'General statement', 'Partial', 'Comprehensive']
    
    transport_summary = pd.DataFrame()
    for cat, display_cat in zip(transport_categories, display_transport_categories):
        subset = analysis_df[analysis_df['CBN_GHG_MITIG_TRANSPORT'] == cat]
        if len(subset) > 0:
            transport_summary.loc[display_cat, 'count'] = len(subset)
            transport_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            transport_summary.loc[display_cat, 'intensity'] = subset['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
    
    results['transport']['base'] = {
        'categories': display_transport_categories,
        'emission_trends': transport_summary['emission_trend'].tolist(),
        'counts': transport_summary['count'].tolist(),
        'intensities': transport_summary['intensity'].tolist()
    }
    
    # 4.2 By company size
    transport_size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            transport_size_summary = pd.DataFrame()
            for cat, display_cat in zip(transport_categories, display_transport_categories):
                subset = size_data[size_data['CBN_GHG_MITIG_TRANSPORT'] == cat]
                if len(subset) > 0:
                    transport_size_summary.loc[display_cat, 'count'] = len(subset)
                    transport_size_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not transport_size_summary.empty:
                transport_size_trends[size] = {
                    'categories': transport_size_summary.index.tolist(),
                    'emission_trends': transport_size_summary['emission_trend'].tolist(),
                    'counts': transport_size_summary['count'].tolist()
                }
    
    results['transport']['by_size'] = transport_size_trends
    
    # 4.3 By region
    transport_region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            transport_region_summary = pd.DataFrame()
            for cat, display_cat in zip(transport_categories, display_transport_categories):
                subset = region_data[region_data['CBN_GHG_MITIG_TRANSPORT'] == cat]
                if len(subset) > 0:
                    transport_region_summary.loc[display_cat, 'count'] = len(subset)
                    transport_region_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not transport_region_summary.empty:
                transport_region_trends[region] = {
                    'categories': transport_region_summary.index.tolist(),
                    'emission_trends': transport_region_summary['emission_trend'].tolist(),
                    'counts': transport_region_summary['count'].tolist()
                }
    
    results['transport']['by_region'] = transport_region_trends
    
    # 4.4 By industry
    transport_industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            transport_industry_summary = pd.DataFrame()
            for cat, display_cat in zip(transport_categories, display_transport_categories):
                subset = industry_data[industry_data['CBN_GHG_MITIG_TRANSPORT'] == cat]
                if len(subset) > 0:
                    transport_industry_summary.loc[display_cat, 'count'] = len(subset)
                    transport_industry_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not transport_industry_summary.empty:
                transport_industry_trends[industry] = {
                    'categories': transport_industry_summary.index.tolist(),
                    'emission_trends': transport_industry_summary['emission_trend'].tolist(),
                    'counts': transport_industry_summary['count'].tolist()
                }
    
    results['transport']['by_industry'] = transport_industry_trends
    
    # --- CARBON CAPTURE STRATEGIES ANALYSIS ---
    # 5.1 Base analysis
    capture_categories = ['No evidence', 'Limited efforts / information', 'Some efforts', 'Aggressive efforts']
    
    capture_summary = pd.DataFrame()
    for cat in capture_categories:
        subset = analysis_df[analysis_df['CBN_GHG_MITIG_CAPTURE'] == cat]
        if len(subset) > 0:
            capture_summary.loc[cat, 'count'] = len(subset)
            capture_summary.loc[cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            capture_summary.loc[cat, 'intensity'] = subset['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
    
    results['capture']['base'] = {
        'categories': capture_categories,
        'emission_trends': capture_summary['emission_trend'].tolist(),
        'counts': capture_summary['count'].tolist(),
        'intensities': capture_summary['intensity'].tolist()
    }
    
    # 5.2 By company size
    capture_size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            capture_size_summary = pd.DataFrame()
            for cat in capture_categories:
                subset = size_data[size_data['CBN_GHG_MITIG_CAPTURE'] == cat]
                if len(subset) > 0:
                    capture_size_summary.loc[cat, 'count'] = len(subset)
                    capture_size_summary.loc[cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not capture_size_summary.empty:
                capture_size_trends[size] = {
                    'categories': capture_size_summary.index.tolist(),
                    'emission_trends': capture_size_summary['emission_trend'].tolist(),
                    'counts': capture_size_summary['count'].tolist()
                }
    
    results['capture']['by_size'] = capture_size_trends
    
    # 5.3 By region
    capture_region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            capture_region_summary = pd.DataFrame()
            for cat in capture_categories:
                subset = region_data[region_data['CBN_GHG_MITIG_CAPTURE'] == cat]
                if len(subset) > 0:
                    capture_region_summary.loc[cat, 'count'] = len(subset)
                    capture_region_summary.loc[cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not capture_region_summary.empty:
                capture_region_trends[region] = {
                    'categories': capture_region_summary.index.tolist(),
                    'emission_trends': capture_region_summary['emission_trend'].tolist(),
                    'counts': capture_region_summary['count'].tolist()
                }
    
    results['capture']['by_region'] = capture_region_trends
    
    # 5.4 By industry
    capture_industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            capture_industry_summary = pd.DataFrame()
            for cat in capture_categories:
                subset = industry_data[industry_data['CBN_GHG_MITIG_CAPTURE'] == cat]
                if len(subset) > 0:
                    capture_industry_summary.loc[cat, 'count'] = len(subset)
                    capture_industry_summary.loc[cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not capture_industry_summary.empty:
                capture_industry_trends[industry] = {
                    'categories': capture_industry_summary.index.tolist(),
                    'emission_trends': capture_industry_summary['emission_trend'].tolist(),
                    'counts': capture_industry_summary['count'].tolist()
                }
    
    results['capture']['by_industry'] = capture_industry_trends
    
    # --- MANUFACTURING STRATEGIES ANALYSIS ---
    # 6.1 Base analysis
    mfg_categories = ['No', 'General statement', 'Some facilities (anecdotal cases)', 
                     'All or core production facilities']
    display_mfg_categories = ['No', 'General statement', 'Some facilities', 'All/Core facilities']
    
    mfg_summary = pd.DataFrame()
    for cat, display_cat in zip(mfg_categories, display_mfg_categories):
        subset = analysis_df[analysis_df['CBN_GHG_MITIG_MFG'] == cat]
        if len(subset) > 0:
            mfg_summary.loc[display_cat, 'count'] = len(subset)
            mfg_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            mfg_summary.loc[display_cat, 'intensity'] = subset['CARBON_EMISSIONS_SCOPE_12_INTEN'].mean()
    
    results['manufacturing']['base'] = {
        'categories': display_mfg_categories,
        'emission_trends': mfg_summary['emission_trend'].tolist(),
        'counts': mfg_summary['count'].tolist(),
        'intensities': mfg_summary['intensity'].tolist()
    }
    
    # 6.2 By company size
    mfg_size_trends = {}
    
    for size in size_order:
        size_data = analysis_df[analysis_df['Size_Category'] == size]
        if len(size_data) > 10:  # Ensure enough data
            mfg_size_summary = pd.DataFrame()
            for cat, display_cat in zip(mfg_categories, display_mfg_categories):
                subset = size_data[size_data['CBN_GHG_MITIG_MFG'] == cat]
                if len(subset) > 0:
                    mfg_size_summary.loc[display_cat, 'count'] = len(subset)
                    mfg_size_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not mfg_size_summary.empty:
                mfg_size_trends[size] = {
                    'categories': mfg_size_summary.index.tolist(),
                    'emission_trends': mfg_size_summary['emission_trend'].tolist(),
                    'counts': mfg_size_summary['count'].tolist()
                }
    
    results['manufacturing']['by_size'] = mfg_size_trends
    
    # 6.3 By region
    mfg_region_trends = {}
    
    for region in region_list:
        region_data = analysis_df[analysis_df['Region'] == region]
        if len(region_data) > 10:  # Ensure enough data
            mfg_region_summary = pd.DataFrame()
            for cat, display_cat in zip(mfg_categories, display_mfg_categories):
                subset = region_data[region_data['CBN_GHG_MITIG_MFG'] == cat]
                if len(subset) > 0:
                    mfg_region_summary.loc[display_cat, 'count'] = len(subset)
                    mfg_region_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not mfg_region_summary.empty:
                mfg_region_trends[region] = {
                    'categories': mfg_region_summary.index.tolist(),
                    'emission_trends': mfg_region_summary['emission_trend'].tolist(),
                    'counts': mfg_region_summary['count'].tolist()
                }
    
    results['manufacturing']['by_region'] = mfg_region_trends
    
    # 6.4 By industry
    mfg_industry_trends = {}
    
    for industry in industry_list:
        industry_data = analysis_df[analysis_df['Industry_Group'] == industry]
        if len(industry_data) > 10:  # Ensure enough data
            mfg_industry_summary = pd.DataFrame()
            for cat, display_cat in zip(mfg_categories, display_mfg_categories):
                subset = industry_data[industry_data['CBN_GHG_MITIG_MFG'] == cat]
                if len(subset) > 0:
                    mfg_industry_summary.loc[display_cat, 'count'] = len(subset)
                    mfg_industry_summary.loc[display_cat, 'emission_trend'] = subset['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean()
            
            if not mfg_industry_summary.empty:
                mfg_industry_trends[industry] = {
                    'categories': mfg_industry_summary.index.tolist(),
                    'emission_trends': mfg_industry_summary['emission_trend'].tolist(),
                    'counts': mfg_industry_summary['count'].tolist()
                }
    
    results['manufacturing']['by_industry'] = mfg_industry_trends
    
    return results, analysis_df

# ===== HELPER FUNCTIONS =====
# Helper function for fancy text with custom background
def add_fancy_text(ax, x, y, text, fontsize=11, ha='center', va='center', fontweight='bold', 
                 color='black', bbox_color='white', bbox_alpha=0.9, zorder=10, transform=None):
    """
    Add text with a styled background box for better visibility.
    
    Parameters:
    1. ax (matplotlib.axes.Axes): The axes to draw on
    2. x, y (float): Text position in data coordinates
    3. text (str): Text to display
    4. fontsize (int): Text font size
    5. ha, va (str): Horizontal and vertical alignment
    6. fontweight (str): Text font weight
    7. color (str): Text color
    8. bbox_color (str): Background box color
    9. bbox_alpha (float): Background box transparency
    10. zorder (int): Drawing order
    11. transform: Coordinate system transformation
    
    Returns:
        matplotlib.text.Text: The created text object
    """
    txt = ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, fontweight=fontweight, 
                 color=color, zorder=zorder, transform=transform,
                 bbox=dict(boxstyle="round,pad=0.4", fc=bbox_color, ec='gray', alpha=bbox_alpha, lw=1))
    return txt

# ===== MAIN FUNCTION =====
def main():
    """
    Execute the complete analysis and visualization pipeline.
    
    Processing Steps:
    1. Load and validate input data
    2. Perform comprehensive mitigation analysis
    3. Generate and configure visualization figures
    4. Create and save all plots
    5. Handle any errors during execution
    
    Returns:
        None: Results are saved to disk and printed to console
    """
    # Load the data
    print("Loading data...")
    try:
        programs1_df = pd.read_csv(PROGRAMS1_DATA_PATH).dropna(axis=1, how='all')
        company_emissions_df = pd.read_csv(COMPANY_EMISSIONS_PATH)
        
        print(f"Programs1 DataFrame: {programs1_df.shape}")
        print(f"Company Emissions DataFrame: {company_emissions_df.shape}")
        
        # Generate visualizations
        print("\nGenerating visualization data...")
        visualization_data, analysis_df = analyze_comprehensive_mitigation_data(programs1_df, company_emissions_df)
        
        # Create the figure and visualizations
        print("Creating visualization grid...")
        
        # Define improved color palette for implementation quality
        implementation_colors = ["#e15759", "#f28e2b", "#edc948", "#59a14f"]
        implementation_cmap = LinearSegmentedColormap.from_list('implementation_cmap', implementation_colors)

        # Create a figure with 6x4 grid of subplots for all 24 plots
        fig = plt.figure(figsize=(28, 36))
        gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1], 
                              hspace=0.7, wspace=0.3)
                              
        # Function to create a basic bar chart for strategy implementation levels
        def create_implementation_bar_chart(ax, data, title, show_annotation=True):
            if not data:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, 
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
                return
            
            categories = data['categories']
            emission_trends = data['emission_trends']
            counts = data['counts']
            
            # Create the bar chart
            x_pos = np.arange(len(categories))
            colors = implementation_cmap(np.linspace(0, 1, len(categories)))
            bars = ax.bar(x_pos, emission_trends, color=colors, edgecolor='black', linewidth=0.8, zorder=3)
            
            # Add gridlines
            ax.grid(axis='y', which='major', alpha=0.3, linestyle='-', zorder=0)
            ax.minorticks_on()
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5, zorder=2)
            
            # Add labels and title
            ax.set_xlabel('Implementation Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('3-Year Emissions Trend (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=14)
            
            # Add emission values on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                color = 'black' if height > 0 else 'white'
                bgColor = 'white' if height > 0 else 'black'
                bgAlpha = 0.7 if height > 0 else 0.5
                
                y_pos = height + 0.15 if height > 0 else height - 0.15
                ax.text(i, y_pos, f"{height:.2f}%", ha='center', va='center', 
                        fontsize=10, fontweight='bold', color=color, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc=bgColor, ec='none', alpha=bgAlpha))
            
            # Add counts below bars
            y_min = min(min(emission_trends) * 1.1, -1)
            for i, count in enumerate(counts):
                ax.text(i, y_min * 0.93, f"n={count:.0f}", ha='center', fontsize=9, 
                        fontweight='bold', color='#333333', zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='#cccccc', alpha=0.8))
            
            # Add annotation comparing best vs worst if applicable
            if show_annotation and len(categories) >= 2:
                first_idx, last_idx = 0, -1
                if 'No' in categories:
                    first_idx = categories.index('No')
                if 'Comprehensive' in categories:
                    last_idx = categories.index('Comprehensive')
                elif 'All/Core facilities' in categories:
                    last_idx = categories.index('All/Core facilities')
                elif 'All or core products' in categories:
                    last_idx = categories.index('All or core products')
                elif 'Aggressive efforts' in categories:
                    last_idx = categories.index('Aggressive efforts')
                
                if last_idx >= 0 and first_idx >= 0:
                    # Add arrow connecting bars
                    fancy_arrow = FancyArrowPatch(
                        (first_idx, emission_trends[first_idx]), 
                        (last_idx, emission_trends[last_idx]),
                        connectionstyle=f"arc3,rad=-0.3", 
                        arrowstyle='->', color='black', lw=1.5, alpha=0.8, zorder=5
                    )
                    ax.add_patch(fancy_arrow)
                    
                    # Add annotation with improvement calculation
                    improvement = abs(emission_trends[last_idx] - emission_trends[first_idx])
                    
                    # Get the correct implementation description
                    impl_desc = "comprehensive implementation"
                    if categories[last_idx] == 'All/Core facilities':
                        impl_desc = "comprehensive facility implementation"
                    elif categories[last_idx] == 'All or core products':
                        impl_desc = "full implementation"
                    elif categories[last_idx] == 'Aggressive efforts':
                        impl_desc = "aggressive capture efforts"
                    
                    annotation_text = f"Companies with {impl_desc} have on average\n{improvement:.2f}% better emission trends than those with no implementation"
                    add_fancy_text(ax, 0.5, 0.85, annotation_text, transform=ax.transAxes, fontsize=10, zorder=10)
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            if len(categories[0]) > 10:  # If labels are long
                ax.set_xticklabels(categories, rotation=15, ha='right', fontweight='bold', fontsize=10)
            else:
                ax.set_xticklabels(categories, fontweight='bold', fontsize=10)
            
            # Set y-axis limits
            y_min = min(min(emission_trends) * 1.1, -1)
            y_max = max(max(emission_trends) * 1.1, 1)
            ax.set_ylim(y_min, y_max)
            
            return ax

        # Modify the create_comparison_line_chart function to use consistent colors
        def create_comparison_line_chart(ax, comparison_data, title, comparison_type="size", category_order=None):
            if not comparison_data:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, 
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
                return
            
            # Use the provided category_order if available, otherwise get all categories from the data
            if category_order:
                unique_categories = [cat for cat in category_order if any(cat in comparison_data[key]['categories'] for key in comparison_data)]
            else:
                # Get categories (implementation levels)
                all_categories = []
                for key in comparison_data:
                    all_categories.extend(comparison_data[key]['categories'])
                unique_categories = sorted(list(set(all_categories)))
            
            # Get the comparison keys based on comparison type
            if comparison_type == "size":
                size_order = ['Mega-cap', 'Large-cap', 'Mid-cap', 'Small-cap', 'Micro-cap'] 
                comparison_keys = [key for key in size_order if key in comparison_data]
                # Use a consistent color scheme for size
                colors = plt.cm.viridis(np.linspace(0, 0.9, len(comparison_keys)))
            elif comparison_type == "region":
                region_list = ['Europe', 'North America', 'Asia-Pacific', 'South America', 'Africa', 'Middle East', 'Others']
                comparison_keys = [key for key in region_list if key in comparison_data]
                # Use our predefined region colors
                colors = [region_colors.get(key, '#999999') for key in comparison_keys]
            else:  # industry
                industry_list = ['Manufacturing', 'Technology', 'Financial Services', 'Energy & Utilities', 
                                 'Retail & Wholesale', 'Transportation', 'Extractive']
                comparison_keys = [key for key in industry_list if key in comparison_data]
                # Use our predefined industry colors
                colors = [industry_colors.get(key, '#999999') for key in comparison_keys]
            
            # Plot each comparison group as a line
            for i, key in enumerate(comparison_keys):
                data = comparison_data[key]
                
                # Create a full set of x-values for all possible categories
                x_values = []
                y_values = []
                
                for cat in unique_categories:
                    if cat in data['categories']:
                        cat_idx = data['categories'].index(cat)
                        x_values.append(unique_categories.index(cat))
                        y_values.append(data['emission_trends'][cat_idx])
                
                # Plot the line if we have data
                if x_values and y_values:
                    color = colors[i] if isinstance(colors[0], str) else colors[i]
                    ax.plot(x_values, y_values, marker='o', color=color, label=key, linewidth=2, markersize=8)
            
            # Add labels and title
            ax.set_xlabel('Implementation Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('3-Year Emissions Trend (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=14)
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add gridlines
            ax.grid(True, alpha=0.3)
            
            # Set x-axis ticks and labels
            ax.set_xticks(range(len(unique_categories)))
            if len(unique_categories) > 0 and len(unique_categories[0]) > 10:  # If labels are long
                ax.set_xticklabels(unique_categories, rotation=15, ha='right', fontweight='bold', fontsize=9)
            else:
                ax.set_xticklabels(unique_categories, fontweight='bold', fontsize=9)
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
            
            return ax

        # Create a figure with 6x4 grid of subplots with reduced spacing at the top
        fig = plt.figure(figsize=(28, 36))
        gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1], 
                              hspace=0.7, wspace=0.3)

        # 1. Combined Mitigation Strategies
        combined_category_order = visualization_data['combined']['base']['categories']

        # 1.0 Combined Mitigation Strategies - Base 
        ax_combined_base = fig.add_subplot(gs[0, 0])
        create_implementation_bar_chart(
            ax_combined_base, 
            visualization_data['combined']['base'],
            "1.0 Combined Mitigation Strategies Effectiveness"
        )

        # 1.1 Combined Mitigation by Company Size
        ax_combined_size = fig.add_subplot(gs[0, 1])
        create_comparison_line_chart(
            ax_combined_size,
            visualization_data['combined']['by_size'],
            "1.1 Combined Mitigation by Company Size",
            "size",
            combined_category_order
        )

        # 1.2 Combined Mitigation by Region
        ax_combined_region = fig.add_subplot(gs[0, 2])
        create_comparison_line_chart(
            ax_combined_region,
            visualization_data['combined']['by_region'],
            "1.2 Combined Mitigation by Region",
            "region",
            combined_category_order
        )

        # 1.3 Combined Mitigation by Industry
        ax_combined_industry = fig.add_subplot(gs[0, 3])
        create_comparison_line_chart(
            ax_combined_industry,
            visualization_data['combined']['by_industry'],
            "1.3 Combined Mitigation by Industry",
            "industry",
            combined_category_order
        )

        # 2. Distribution Strategies
        dist_category_order = visualization_data['distribution']['base']['categories']

        # 2.0 Distribution Strategies - Base
        ax_dist_base = fig.add_subplot(gs[1, 0])
        create_implementation_bar_chart(
            ax_dist_base, 
            visualization_data['distribution']['base'],
            "2.0 Distribution Strategies Effectiveness"
        )

        # 2.1 Distribution by Company Size
        ax_dist_size = fig.add_subplot(gs[1, 1])
        create_comparison_line_chart(
            ax_dist_size,
            visualization_data['distribution']['by_size'],
            "2.1 Distribution by Company Size",
            "size",
            dist_category_order
        )

        # 2.2 Distribution by Region
        ax_dist_region = fig.add_subplot(gs[1, 2])
        create_comparison_line_chart(
            ax_dist_region,
            visualization_data['distribution']['by_region'],
            "2.2 Distribution by Region",
            "region",
            dist_category_order
        )

        # 2.3 Distribution by Industry
        ax_dist_industry = fig.add_subplot(gs[1, 3])
        create_comparison_line_chart(
            ax_dist_industry,
            visualization_data['distribution']['by_industry'],
            "2.3 Distribution by Industry",
            "industry",
            dist_category_order
        )

        # 3. Raw Materials Strategies
        raw_category_order = visualization_data['raw_materials']['base']['categories']

        # 3.0 Raw Materials Strategies - Base
        ax_raw_base = fig.add_subplot(gs[2, 0])
        create_implementation_bar_chart(
            ax_raw_base, 
            visualization_data['raw_materials']['base'],
            "3.0 Raw Materials Strategies Effectiveness"
        )

        # 3.1 Raw Materials by Company Size
        ax_raw_size = fig.add_subplot(gs[2, 1])
        create_comparison_line_chart(
            ax_raw_size,
            visualization_data['raw_materials']['by_size'],
            "3.1 Raw Materials by Company Size",
            "size",
            raw_category_order
        )

        # 3.2 Raw Materials by Region
        ax_raw_region = fig.add_subplot(gs[2, 2])
        create_comparison_line_chart(
            ax_raw_region,
            visualization_data['raw_materials']['by_region'],
            "3.2 Raw Materials by Region",
            "region",
            raw_category_order
        )

        # 3.3 Raw Materials by Industry
        ax_raw_industry = fig.add_subplot(gs[2, 3])
        create_comparison_line_chart(
            ax_raw_industry,
            visualization_data['raw_materials']['by_industry'],
            "3.3 Raw Materials by Industry",
            "industry",
            raw_category_order
        )

        # 4. Transport Strategies
        transport_category_order = visualization_data['transport']['base']['categories']

        # 4.0 Transport Strategies - Base
        ax_transport_base = fig.add_subplot(gs[3, 0])
        create_implementation_bar_chart(
            ax_transport_base, 
            visualization_data['transport']['base'],
            "4.0 Transport Strategies Effectiveness"
        )

        # 4.1 Transport by Company Size
        ax_transport_size = fig.add_subplot(gs[3, 1])
        create_comparison_line_chart(
            ax_transport_size,
            visualization_data['transport']['by_size'],
            "4.1 Transport by Company Size",
            "size",
            transport_category_order
        )

        # 4.2 Transport by Region
        ax_transport_region = fig.add_subplot(gs[3, 2])
        create_comparison_line_chart(
            ax_transport_region,
            visualization_data['transport']['by_region'],
            "4.2 Transport by Region",
            "region",
            transport_category_order
        )

        # 4.3 Transport by Industry
        ax_transport_industry = fig.add_subplot(gs[3, 3])
        create_comparison_line_chart(
            ax_transport_industry,
            visualization_data['transport']['by_industry'],
            "4.3 Transport by Industry",
            "industry",
            transport_category_order
        )

        # 5. Carbon Capture Strategies
        capture_category_order = visualization_data['capture']['base']['categories']

        # 5.0 Carbon Capture Strategies - Base
        ax_capture_base = fig.add_subplot(gs[4, 0])
        create_implementation_bar_chart(
            ax_capture_base, 
            visualization_data['capture']['base'],
            "5.0 Carbon Capture Strategies Effectiveness"
        )

        # 5.1 Carbon Capture by Company Size
        ax_capture_size = fig.add_subplot(gs[4, 1])
        create_comparison_line_chart(
            ax_capture_size,
            visualization_data['capture']['by_size'],
            "5.1 Carbon Capture by Company Size",
            "size",
            capture_category_order
        )

        # 5.2 Carbon Capture by Region
        ax_capture_region = fig.add_subplot(gs[4, 2])
        create_comparison_line_chart(
            ax_capture_region,
            visualization_data['capture']['by_region'],
            "5.2 Carbon Capture by Region",
            "region",
            capture_category_order
        )

        # 5.3 Carbon Capture by Industry
        ax_capture_industry = fig.add_subplot(gs[4, 3])
        create_comparison_line_chart(
            ax_capture_industry,
            visualization_data['capture']['by_industry'],
            "5.3 Carbon Capture by Industry",
            "industry",
            capture_category_order
        )

        # 6. Manufacturing Strategies
        mfg_category_order = visualization_data['manufacturing']['base']['categories']

        # 6.0 Manufacturing Strategies - Base
        ax_mfg_base = fig.add_subplot(gs[5, 0])
        create_implementation_bar_chart(
            ax_mfg_base, 
            visualization_data['manufacturing']['base'],
            "6.0 Manufacturing Strategies Effectiveness"
        )

        # 6.1 Manufacturing by Company Size
        ax_mfg_size = fig.add_subplot(gs[5, 1])
        create_comparison_line_chart(
            ax_mfg_size,
            visualization_data['manufacturing']['by_size'],
            "6.1 Manufacturing by Company Size",
            "size",
            mfg_category_order
        )

        # 6.2 Manufacturing by Region
        ax_mfg_region = fig.add_subplot(gs[5, 2])
        create_comparison_line_chart(
            ax_mfg_region,
            visualization_data['manufacturing']['by_region'],
            "6.2 Manufacturing by Region",
            "region",
            mfg_category_order
        )

        # 6.3 Manufacturing by Industry
        ax_mfg_industry = fig.add_subplot(gs[5, 3])
        create_comparison_line_chart(
            ax_mfg_industry,
            visualization_data['manufacturing']['by_industry'],
            "6.3 Manufacturing by Industry",
            "industry",
            mfg_category_order
        )

        # Add an overall title for the figure with better prominence
        fig.suptitle('Carbon Emission Reduction Strategies Effectiveness - Comprehensive Analysis', 
                     fontsize=24, fontweight='bold', y=0.99)

        # Calculate data completeness
        missing_distribution = 100 * programs1_df['CBN_GHG_MITIG_DISTRIBUTION'].isna().sum() / len(programs1_df)
        missing_raw_mat = 100 * programs1_df['CBN_GHG_MITIG_RAW_MAT'].isna().sum() / len(programs1_df)
        missing_mfg = 100 * programs1_df['CBN_GHG_MITIG_MFG'].isna().sum() / len(programs1_df)
        missing_transport = 100 * programs1_df['CBN_GHG_MITIG_TRANSPORT'].isna().sum() / len(programs1_df)
        missing_capture = 100 * programs1_df['CBN_GHG_MITIG_CAPTURE'].isna().sum() / len(programs1_df)

        # Add information about data completeness with enhanced styling
        fig.text(0.5, 0.005, f'Source: Analysis of Reduction Programs 1 dataframe (n={len(programs1_df)} companies)',
                ha='center', fontsize=12, fontweight='bold')

        # Use tighter layout to maximize chart area and reduce whitespace at the top
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        # Further reduce the whitespace at top by adjusting the top parameter
        plt.subplots_adjust(top=0.95)
        
        # Use the save_figure function from visualization_utils
        from src.utils.visualization_utils import save_figure
        
        # Save the figure in both PNG and PDF formats
        filename = 'fig03_programs1_detailed_visualization'
        print(f"\nSaving visualization as {filename} in output_graphs/png/ and output_graphs/pdf/ directories")
        
        # Save the figure
        save_figure(plt.gcf(), filename, formats=['png', 'pdf'], dpi=300)
        
        print("Visualizations saved successfully!")
        
        # Comment/uncomment the next line to enable/disable interactive display
        # This may cause issues in some environments (e.g., segmentation faults)
        # plt.show()
        
        print("\nAnalysis complete! Visualizations have been saved to:")
        print(f"  - {paths.OUTPUT_DIR}/png/")
        print(f"  - {paths.OUTPUT_DIR}/pdf/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
# ===== SCRIPT EXECUTION =====
if __name__ == "__main__":
    print("Starting Programs1 Detailed Visualization Analysis...")
    main()