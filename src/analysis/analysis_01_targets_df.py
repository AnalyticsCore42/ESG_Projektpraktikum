"""
This script analyzes the relationship between corporate emissions and carbon reduction targets.

Input DataFrames:
1. company_emissions_df: Contains company emissions data (Scope 1+2, Scope 3)
2. targets_df: Contains carbon reduction targets and progress metrics

Key Analyses Performed:
1. Scale of Emissions and Target-Setting Behavior (fig1):
   - Correlation between company emission levels and number of targets set
   - Relationship between company size/emissions and target ambition
   
2. Types of Targets and Their Effectiveness (fig2):
   - Comparison of target types (absolute vs. intensity-based)
   - Analysis of progress toward different target types
   
3. Geographic and Industry Context (fig3):
   - Country-level patterns in target setting and emissions
   - Industry-specific relationships between targets and emissions
   
4. Counterintuitive Relationships (fig4):
   - Analysis of unexpected patterns in the data
   - Exploration of weak or non-existent relationships

Outputs:
- Generates 4 main visualization figures (fig01-04) saved in output_graphs/
- Each figure contains multiple subplots with detailed visual analysis
- Visualizations include scatter plots, bar charts, and correlation analyses
"""

# ===== IMPORTS =====
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import paths
from src.utils.visualization_utils import (
    setup_plot_style, save_figure, add_correlation_annotation,
    plot_trendline, create_scatter_with_correlation, create_bar_chart
)
from src.utils.analysis_utils import winsorize

warnings.filterwarnings('ignore')

# ===== FILE PATHS =====
MERGED_PATH = paths.DATA_DIR / "company_emissions_merged.csv"
TARGETS_PATH = paths.DATA_DIR / "Reduktionsziele Results - 20241212 15_49_29.csv"

# ===== FILE VALIDATION =====
for path, name in [(MERGED_PATH, "Merged data"), (TARGETS_PATH, "Targets data")]:
    if not path.is_file():
        print(f"ERROR: {name} file not found at {path}")
        sys.exit(1)

# ===== DATA LOADING =====
def load_data():
    """
    Load and validate emissions and targets data.
    
    Returns:
        tuple: (company_emissions_df, targets_df) containing loaded and validated data
    """
    print("Loading data...")
    try:
        company_emissions_df = pd.read_csv(MERGED_PATH)
        targets_df = pd.read_csv(TARGETS_PATH).dropna(axis=1, how='all')

        print(f"Loaded company_emissions_df: shape={company_emissions_df.shape}, unique_ISSUERIDs={company_emissions_df['ISSUERID'].nunique():,}")
        print(f"Loaded targets_df: shape={targets_df.shape}, unique_ISSUERIDs={targets_df['ISSUERID'].nunique():,}")
        return company_emissions_df, targets_df
    except FileNotFoundError as e:
         print(f"Error loading data file: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        sys.exit(1)

# ===== DATA PREPARATION =====
print("\n--- DATA PREPARATION ---")
print("=" * 80)
print("Loading and preparing data...")
try:
    company_emissions_df, targets_df = load_data()
    print("✓ Data loaded successfully")
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"- Companies with emissions data: {company_emissions_df['ISSUERID'].nunique():,}")
    print(f"- Companies with targets: {targets_df['ISSUERID'].nunique():,}")
    print(f"- Total targets: {len(targets_df):,}")
    
    print("\nSetting up plot style...")
except Exception as e:
    print(f"\n❌ Error during data preparation: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
setup_plot_style()
print("Setup complete. Ready for analysis.")

print("\n--- EMISSIONS AND TARGET SETTING ANALYSIS ---")
print("=" * 80)
print("All results for this analysis are saved as visualizations (figures) in the output directory. No detailed results are printed to the console.")

# ===== ANALYSIS =====
fig1 = plt.figure(figsize=(18, 20))
gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.4, wspace=0.3)

target_level_analysis = targets_df.copy()

emissions_columns = ['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 
                    'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_3',
                    'ISSUER_CNTRY_DOMICILE_company', 'NACE_CLASS_DESCRIPTION']

target_level_analysis = pd.merge(
    target_level_analysis,
    company_emissions_df[emissions_columns].drop_duplicates('ISSUERID'),
    on='ISSUERID',
    how='left'
)

print("\n1. TARGET COUNT VS EMISSIONS RELATIONSHIP")
print("-" * 80)
print("Generating and saving plots for target count vs emissions relationship...")

ax1 = fig1.add_subplot(gs1[0, 0])

target_counts = target_level_analysis.groupby('ISSUERID').size().reset_index(name='target_count')

company_emissions = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12']].drop_duplicates()

company_target_data = pd.merge(company_emissions, target_counts, on='ISSUERID', how='inner')

company_target_data = company_target_data[
    (company_target_data['CARBON_EMISSIONS_SCOPE_12'] > 0) & 
    (company_target_data['target_count'] > 0)
]

title = '1.1 Higher-Emitting Companies Set More Targets'
x_data = np.log10(company_target_data['CARBON_EMISSIONS_SCOPE_12'])
y_data = company_target_data['target_count']
xlabel = 'Log₁₀ Absolute Emissions (Scope 1+2)'
ylabel = 'Number of Carbon Targets'
annotation = "Companies with higher absolute emissions tend\nto establish more carbon reduction targets"

create_scatter_with_correlation(
    ax1, x_data, y_data, title, xlabel, ylabel, 
    log_y=True, annotation_text=annotation
)

ax2 = fig1.add_subplot(gs1[0, 1])

company_scope3 = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_3']].drop_duplicates()

company_scope3_targets = pd.merge(company_scope3, target_counts, on='ISSUERID', how='inner')

company_scope3_targets = company_scope3_targets[
    (company_scope3_targets['CARBON_EMISSIONS_SCOPE_3'] > 0) & 
    (company_scope3_targets['target_count'] > 0)
]

title = '1.2 Scope 3 Emissions and Target Count'
x_data = np.log10(company_scope3_targets['CARBON_EMISSIONS_SCOPE_3'])
y_data = company_scope3_targets['target_count']
xlabel = 'Log₁₀ Scope 3 Emissions'
ylabel = 'Number of Carbon Targets (log scale)'
annotation = "Companies with higher value chain emissions also\ntend to set more carbon reduction targets"

create_scatter_with_correlation(
    ax2, x_data, y_data, title, xlabel, ylabel, 
    log_y=True, c='#31a354', annotation_text=annotation
)

print("\n2. TARGET AMBITION ANALYSIS")
print("-" * 80)
print("Generating and saving plots for target ambition analysis...")

ax3 = fig1.add_subplot(gs1[1, 0])

company_ambition = target_level_analysis.groupby('ISSUERID')['CBN_TARGET_REDUC_PCT'].mean().reset_index()

company_ambition_emissions = pd.merge(
    company_ambition,
    company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12']],
    on='ISSUERID',
    how='inner'
)

company_ambition_emissions = company_ambition_emissions[
    (company_ambition_emissions['CARBON_EMISSIONS_SCOPE_12'] > 0) &
    (company_ambition_emissions['CBN_TARGET_REDUC_PCT'].notna())
]

company_ambition_emissions['log_emissions'] = np.log10(company_ambition_emissions['CARBON_EMISSIONS_SCOPE_12'])

hb = ax3.hexbin(
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT'],
    gridsize=30, cmap='Oranges', mincnt=1, bins='log'
)

corr, p_value = add_correlation_annotation(
    ax3, 
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT']
)

plot_trendline(
    ax3,
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT']
)

ax3.set_xlabel('Log₁₀ Absolute Emissions (Scope 1+2)', fontsize=12)
ax3.set_ylabel('Target Reduction Percentage (%)', fontsize=12)
ax3.set_title('1.3 Target Ambition vs. Absolute Emissions\nDensity Plot Shows Relationship Pattern', fontweight='bold', fontsize=14)
ax3.grid(alpha=0.3)

cb = plt.colorbar(hb, ax=ax3)
cb.set_label('Log Number of Companies')

if corr < 0:
    annotation = "Higher-emitting companies tend to set\nless ambitious reduction targets"
else:
    annotation = "Higher-emitting companies tend to set\nmore ambitious reduction targets"
    
ax3.text(0.5, 0.05, annotation, 
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 1.4 Industry Analysis: Correlation Between Target Ambition and Emissions by Sector
ax4 = fig1.add_subplot(gs1[1, 1])

industry_data = company_emissions_df[['ISSUERID', 'NACE_CLASS_DESCRIPTION']].drop_duplicates()

company_industry_ambition = pd.merge(
    company_ambition_emissions, 
    industry_data,
    on='ISSUERID',
    how='inner'
)

industry_stats = []
for industry, group in company_industry_ambition.groupby('NACE_CLASS_DESCRIPTION'):
    if len(group) >= 10:
        corr, p_value = stats.pearsonr(
            group['log_emissions'],
            group['CBN_TARGET_REDUC_PCT']
        )
        industry_stats.append({
            'Industry': industry,
            'Correlation': corr,
            'P_Value': p_value,
            'Count': len(group)
        })

industry_correlations = pd.DataFrame(industry_stats)

significant_industries = industry_correlations[industry_correlations['P_Value'] < 0.05]

if len(significant_industries) > 10:
    significant_industries = significant_industries.copy()
    significant_industries['Abs_Corr'] = significant_industries['Correlation'].abs()
    significant_industries = significant_industries.nlargest(10, 'Abs_Corr')

significant_industries = significant_industries.sort_values('Correlation', ascending=False)

# Fix SettingWithCopyWarning for Short_Industry
significant_industries = significant_industries.copy()
significant_industries['Short_Industry'] = significant_industries['Industry'].str.replace(' and ', ' & ')
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.slice(0, 30)
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.replace('Manufacture of ', '')
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.replace('Manufacturing ', '')

colors = ['#1a9850' if c >= 0 else '#d73027' for c in significant_industries['Correlation']]

bars = ax4.barh(
    significant_industries['Short_Industry'], 
    significant_industries['Correlation'], 
    color=colors, 
    height=0.7
)

ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)

for i, (_, row) in enumerate(significant_industries.iterrows()):
    height = row['Correlation']
    ha = 'left' if height >= 0 else 'right'
    offset = 0.02 if height >= 0 else -0.02
    ax4.text(
        height + offset,
        i,
        f"p={row['P_Value']:.3f}, n={row['Count']}",
        ha=ha, va='center', fontsize=9,
        weight='bold'
    )

ax4.set_ylabel('Industry Sector', fontsize=12)
ax4.set_xlabel('Correlation: Target Ambition & Emissions', fontsize=12)
ax4.set_title('1.4 Significant Industry-Specific Correlations\nBetween Target Ambition and Emissions', fontweight='bold', fontsize=14)
ax4.grid(axis='x', alpha=0.3)

if len(significant_industries) > 0:
    most_positive = significant_industries.iloc[0]['Short_Industry'] if significant_industries.iloc[0]['Correlation'] > 0 else "No industries"
    most_negative = significant_industries.iloc[-1]['Short_Industry'] if significant_industries.iloc[-1]['Correlation'] < 0 else "No industries"
    
    ax4.text(0.5, 0.02, 
             f"Different industries show opposite patterns:\n{most_positive}: higher emissions → more ambitious targets\n{most_negative}: higher emissions → less ambitious targets", 
             transform=ax4.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

fig1.suptitle('1. Scale of Emissions and Target-Setting Behavior', 
              fontsize=18, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
save_figure(fig1, 'fig01_emissions_and_targets')

use_winsorization = False

fig2 = plt.figure(figsize=(18, 18))
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.3)

emissions_intensity_data = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()

target_types = target_level_analysis[['ISSUERID', 'TARGET_CARBON_TYPE']].dropna(subset=['TARGET_CARBON_TYPE'])

target_type_emissions = pd.merge(
    target_types,
    emissions_intensity_data,
    on='ISSUERID',
    how='inner'
)

# 2.1 Comparing Emissions: Absolute vs. Intensity-based Target Approaches
ax1 = fig2.add_subplot(gs2[0, 0])

target_type_emissions = target_type_emissions[target_type_emissions['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]

def winsorize(data, lower_percentile=5, upper_percentile=95):
    """
    Limit extreme values by capping at specified percentiles.
    
    Args:
        data: Input array of values
        lower_percentile: Bottom percentile to cap (default: 5th)
        upper_percentile: Top percentile to cap (default: 95th)
        
    Returns:
        Array with extreme values capped
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)

emissions_by_type = []
for type_name, group in target_type_emissions.groupby('TARGET_CARBON_TYPE'):
    if len(group) >= 30:
        if use_winsorization:
            processed_emissions = winsorize(group['CARBON_EMISSIONS_SCOPE_12_INTEN'].values)
        else:
            processed_emissions = group['CARBON_EMISSIONS_SCOPE_12_INTEN'].values
        
        emissions_by_type.append({
            'TARGET_CARBON_TYPE': type_name,
            'mean': np.mean(processed_emissions),
            'median': np.median(group['CARBON_EMISSIONS_SCOPE_12_INTEN']),
            'count': len(group),
            'std': np.std(processed_emissions)
        })

emissions_by_type = pd.DataFrame(emissions_by_type)

emissions_by_type = emissions_by_type.sort_values('count', ascending=False)

target_type_colors = {
    'Absolute': '#1a9850',
    'Production intensity': '#A63603',
    'Sales intensity': '#74add1',
    'Others?': '#984ea3'
}

colors = [target_type_colors.get(t, '#999999') for t in emissions_by_type['TARGET_CARBON_TYPE']]

bars = ax1.bar(
    emissions_by_type['TARGET_CARBON_TYPE'], 
    emissions_by_type['median'],
    yerr=emissions_by_type['std'] / np.sqrt(emissions_by_type['count']),
    color=colors, 
    edgecolor=colors,
    width=0.6,
    capsize=5
)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,  
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Median Emissions Intensity (Scope 1+2)', fontsize=12)
title_suffix = " (with Winsorization)" if use_winsorization else " (No Winsorization)"
ax1.set_title(f'2.1 Emissions Intensity by Target Type{title_suffix}', fontweight='bold', fontsize=14)
ax1.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

ax1.text(0.5, 0.05, "Companies using absolute targets have lower emissions\nintensity levels than those using production intensity targets", 
         transform=ax1.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 2.2 Emission Performance by Target Category and Type
ax2 = fig2.add_subplot(gs2[0, 1])

count_pivot = pd.pivot_table(
    target_level_analysis.dropna(subset=['CBN_TARGET_CATEGORY', 'TARGET_CARBON_TYPE', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']),
    values='ISSUERID',
    index='CBN_TARGET_CATEGORY',
    columns='TARGET_CARBON_TYPE',
    aggfunc='count'
)

pivot_data = pd.pivot_table(
    target_level_analysis.dropna(subset=['CBN_TARGET_CATEGORY', 'TARGET_CARBON_TYPE', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']),
    values='CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',
    index='CBN_TARGET_CATEGORY',
    columns='TARGET_CARBON_TYPE',
    aggfunc='mean'
)

if not pivot_data.empty and not pivot_data.isna().all().all():
    min_sample_size = 10
    
    mask = count_pivot < min_sample_size
    
    annot_text = pivot_data.copy().round(1).astype(str)
    for i in range(pivot_data.shape[0]):
        for j in range(pivot_data.shape[1]):
            if not pd.isna(pivot_data.iloc[i, j]) and not pd.isna(count_pivot.iloc[i, j]):
                count = count_pivot.iloc[i, j]
                value = pivot_data.iloc[i, j]
                annot_text.iloc[i, j] = f"{value:.1f}\n(n={count})"
            else:
                annot_text.iloc[i, j] = ""
    
    category_order = target_level_analysis['CBN_TARGET_CATEGORY'].value_counts().index.tolist()
    type_order = target_level_analysis['TARGET_CARBON_TYPE'].value_counts().index.tolist()
    
    valid_categories = [cat for cat in category_order if cat in pivot_data.index]
    valid_types = [typ for typ in type_order if typ in pivot_data.columns]
    
    pivot_data = pivot_data.reindex(valid_categories, axis=0)
    pivot_data = pivot_data.reindex(valid_types, axis=1)
    
    count_pivot = count_pivot.reindex(valid_categories, axis=0)
    count_pivot = count_pivot.reindex(valid_types, axis=1)
    mask = mask.reindex(valid_categories, axis=0)
    mask = mask.reindex(valid_types, axis=1)
    
    annot_text = annot_text.reindex(valid_categories, axis=0)
    annot_text = annot_text.reindex(valid_types, axis=1)
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")
    
    heatmap = sns.heatmap(
        pivot_data,
        annot=annot_text,
        fmt="",
        cmap=cmap,
        center=0,
        linewidths=0.7,
        ax=ax2,
        cbar_kws={"shrink": 0.8, "label": "Emission Trend (%)"},
        mask=mask,
    )
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask.iloc[i, j] and not pd.isna(pivot_data.iloc[i, j]):
                x = j
                y = i
                ax2.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, hatch='////', alpha=0.5))
    
    ax2.set_title('2.2 Emission Performance by Target Category and Type', 
                 fontweight='bold', fontsize=14)
    
    ax2.set_xlabel('Target Carbon Type', fontsize=12)
    ax2.set_ylabel('Target Category', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    if not pivot_data.isna().all().all():
        masked_pivot = pivot_data.mask(mask)
        
        best_val = masked_pivot.min().min()
        worst_val = masked_pivot.max().max()
        
        best_idx = masked_pivot.stack().idxmin()
        worst_idx = masked_pivot.stack().idxmax()
        
        best_n = count_pivot.loc[best_idx[0], best_idx[1]]
        worst_n = count_pivot.loc[worst_idx[0], worst_idx[1]]
        
        ax2.text(0.5, -0.25, 
                f"Best performance: {best_idx[0]} with {best_idx[1]} target type shows strongest emissions reduction\n"
                f"({best_val:.1f}%, n={best_n}) - Companies using 'other' emissions metrics with production-based targets\n"
                f"Worst performance: {worst_idx[0]} with {worst_idx[1]} target type shows emissions increase\n"
                f"({worst_val:.1f}%, n={worst_n}) - Energy consumption targets using production metrics need improvement",
                transform=ax2.transAxes, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.9))
        
        ax2.text(0.5, -0.42, 
                f"Note: Cells with fewer than {min_sample_size} companies are hatched and excluded from 'best/worst' calculations",
                transform=ax2.transAxes, fontsize=10, ha='center', style='italic',
                bbox=dict(boxstyle="round,pad=0.2", fc='#f0f0f0', alpha=0.7))
else:
    ax2.text(0.5, 0.5, "Insufficient data for target category vs. type analysis", 
             transform=ax2.transAxes, fontsize=14, ha='center')
    ax2.axis('off')

# 2.3 Target Progress and Emission Trends - IMPROVED with better color scale
ax3 = fig2.add_subplot(gs2[1, 0])

progress_trend_data = target_level_analysis.dropna(subset=['TARGET_CARBON_PROGRESS_PCT', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])

progress_trend_data = progress_trend_data[
    (progress_trend_data['TARGET_CARBON_PROGRESS_PCT'] >= 0) & 
    (progress_trend_data['TARGET_CARBON_PROGRESS_PCT'] <= 200)
]

if use_winsorization:
    progress_trend_data['robust_trend'] = progress_trend_data.groupby(
        pd.qcut(progress_trend_data['TARGET_CARBON_PROGRESS_PCT'], 8, duplicates='drop')
    )['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].transform(
        lambda x: winsorize(x, 5, 95)
    )
else:
    progress_trend_data['robust_trend'] = progress_trend_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']

progress_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]
bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%', '100-125%', '125-150%', '150-175%', '175-200%']

progress_trend_data['progress_bin'] = pd.cut(
    progress_trend_data['TARGET_CARBON_PROGRESS_PCT'],
    bins=progress_bins,
    labels=bin_labels,
    include_lowest=True
)

# Fix FutureWarning for groupby observed
trend_by_progress = progress_trend_data.groupby('progress_bin', observed=False)['robust_trend'].agg(
    ['mean', 'median', 'count', 'std']
).reset_index()

greens = plt.cm.Greens(np.linspace(0.3, 0.9, len(trend_by_progress)))
colors = greens

pos = np.arange(len(trend_by_progress))
bars = ax3.bar(
    pos, 
    trend_by_progress['mean'],
    yerr=trend_by_progress['std'] / np.sqrt(trend_by_progress['count']),
    color=colors,
    edgecolor=colors,
    width=0.7,
    capsize=5
)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height - 0.3 if height < 0 else height + 0.3,
            f'{height:.1f}%',
            ha='center', va='top' if height < 0 else 'bottom', 
            fontweight='bold', color='white' if height < -5 else 'black')

ax3.set_xlabel('TARGET_CARBON_PROGRESS_PCT\n(% progress toward reduction target, >100% = target exceeded)', fontsize=12)
ax3.set_ylabel('3-Year Emissions Trend (%) (Scope 1+2)', fontsize=12)
title_suffix = " (with Winsorization)" if use_winsorization else " (No Winsorization)"
ax3.set_title(f'2.3 Target Progress and Emissions Trends{title_suffix}', fontweight='bold', fontsize=14)
ax3.grid(axis='y', alpha=0.3)

ax3.set_xticks(pos)
ax3.set_xticklabels(trend_by_progress['progress_bin'], rotation=45, ha='right')

best_idx = trend_by_progress['mean'].idxmin()
best_progress = trend_by_progress.iloc[best_idx]

ax3.text(0.5, 0.1, 
         f"Targets with {best_progress['progress_bin']} progress show best trend\n({best_progress['mean']:.1f}%, n={best_progress['count']})\nHigher values indicate targets exceeded by a larger margin", 
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

fig2.suptitle('Group 2: Types of Targets and Their Effectiveness', 
              fontsize=18, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
save_figure(fig2, 'fig02_target_types_effectiveness')

# Create a figure with 4 subplots for Group 3
fig3 = plt.figure(figsize=(18, 20))
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.4, wspace=0.3)

# 3.1 Country-Level Target Count vs. Emission Trends
ax1 = fig3.add_subplot(gs3[0, 0])

country_emissions = company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()

country_trends = country_emissions.groupby('ISSUER_CNTRY_DOMICILE_company')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean().reset_index()

company_targets = target_level_analysis.groupby(['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company']).size().reset_index(name='target_count')

country_avg_targets = company_targets.groupby('ISSUER_CNTRY_DOMICILE_company').agg({
    'target_count': 'mean',
    'ISSUERID': 'nunique'
}).reset_index()
country_avg_targets.rename(columns={'ISSUERID': 'company_count'}, inplace=True)

country_data = pd.merge(country_trends, country_avg_targets, on='ISSUER_CNTRY_DOMICILE_company', how='inner')

country_data = country_data[country_data['company_count'] >= 10]

scatter = ax1.scatter(
    country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
    country_data['target_count'],
    s=country_data['company_count'] * 3,  # Size by company count
    c=country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],  # Color by trend
    cmap='RdYlGn_r',  # Red for positive (bad), green for negative (good)
    edgecolor='black',
    alpha=0.7
)

# Add country labels
for i, row in country_data.iterrows():
    ax1.annotate(
        row['ISSUER_CNTRY_DOMICILE_company'], 
        xy=(row['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], row['target_count']),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        weight='bold'
    )

# Calculate correlation
corr, p_value = stats.pearsonr(
    country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
    country_data['target_count']
)

# Add correlation information
ax1.text(0.05, 0.95, f"r = {corr:.3f}, p = {p_value:.3f}", 
         transform=ax1.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

# Add trendline
m, b = np.polyfit(
    country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
    country_data['target_count'], 
    1
)
x_line = np.linspace(
    min(country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']),
    max(country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']), 
    100
)
ax1.plot(x_line, m*x_line + b, color='red', linestyle='--', linewidth=2)

# Set labels and title
ax1.set_xlabel('3-Year Emissions Trend (%)', fontsize=12)
ax1.set_ylabel('Average Number of Targets per Company', fontsize=12)
ax1.set_title(f'3.1 Country-Level Target Count vs. Emission Trends\nBased on {len(country_data)} Countries with ≥10 Companies', fontweight='bold', fontsize=14)
ax1.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Emissions Trend (%)', rotation=270, labelpad=20)

# Add annotation based on correlation direction
if corr < 0:
    annotation = "Countries with better emissions trends tend to have\ncompanies that set more targets"
else:
    annotation = "Countries with better emissions trends tend to have\ncompanies that set fewer targets"
    
ax1.text(0.5, 0.05, annotation, 
         transform=ax1.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 3.2 Industry-specific correlations between target ambition and emissions
ax2 = fig3.add_subplot(gs3[0, 1])

# Add industry information to emissions data
industry_emissions_data = company_emissions_df[['ISSUERID', 'NACE_CLASS_DESCRIPTION', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()

# Get target ambition data
target_ambition = target_level_analysis.groupby('ISSUERID')['CBN_TARGET_REDUC_PCT'].mean().reset_index()

# Merge ambition with industry and emissions data
industry_target_emissions = pd.merge(
    target_ambition,
    industry_emissions_data,
    on='ISSUERID',
    how='inner'
)

# Filter out zero or missing values
industry_target_emissions = industry_target_emissions[
    (industry_target_emissions['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0) &
    (industry_target_emissions['CBN_TARGET_REDUC_PCT'].notna())
]

# Create log transformed x-values
industry_target_emissions['log_emissions'] = np.log10(industry_target_emissions['CARBON_EMISSIONS_SCOPE_12_INTEN'])

# Calculate correlation by industry
industry_stats = []
for industry, group in industry_target_emissions.groupby('NACE_CLASS_DESCRIPTION'):
    # Only include industries with enough companies
    if len(group) >= 10:
        # Calculate correlation for this industry
        corr, p_value = stats.pearsonr(
            group['log_emissions'],
            group['CBN_TARGET_REDUC_PCT']
        )
        industry_stats.append({
            'Industry': industry,
            'Correlation': corr,
            'P_Value': p_value,
            'Count': len(group)
        })

# Convert to DataFrame
industry_correlations = pd.DataFrame(industry_stats)

# Filter to significant correlations (p < 0.05)
significant_industries = industry_correlations[industry_correlations['P_Value'] < 0.05]

# Limit to top 10 by correlation strength if needed
if len(significant_industries) > 10:
    significant_industries = significant_industries.copy()
    significant_industries['Abs_Corr'] = significant_industries['Correlation'].abs()
    significant_industries = significant_industries.nlargest(10, 'Abs_Corr')

# Sort from most positive to most negative
significant_industries = significant_industries.sort_values('Correlation', ascending=False)

# Fix SettingWithCopyWarning for Short_Industry
significant_industries = significant_industries.copy()
significant_industries['Short_Industry'] = significant_industries['Industry'].str.replace(' and ', ' & ')
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.slice(0, 30)
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.replace('Manufacture of ', '')
significant_industries['Short_Industry'] = significant_industries['Short_Industry'].str.replace('Manufacturing ', '')

# Create color scheme based on correlation direction
colors = ['#1a9850' if c >= 0 else '#d73027' for c in significant_industries['Correlation']]

# Create horizontal bar chart for better readability
bars = ax2.barh(
    significant_industries['Short_Industry'], 
    significant_industries['Correlation'], 
    color=colors, 
    height=0.7
)

# Add a vertical line at x=0
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Annotate with p-values
for i, (_, row) in enumerate(significant_industries.iterrows()):
    height = row['Correlation']
    ha = 'left' if height >= 0 else 'right'
    offset = 0.02 if height >= 0 else -0.02
    ax2.text(
        height + offset,
        i,
        f"p={row['P_Value']:.3f}, n={row['Count']}",
        ha=ha, va='center', fontsize=9,
        weight='bold'
    )

# Set labels and title
ax2.set_ylabel('Industry Sector', fontsize=12)
ax2.set_xlabel('Correlation: Target Ambition & Emissions Intensity', fontsize=12)
ax2.set_title('3.2 Industry-Specific Correlations\nBetween Target Ambition and Emissions', fontweight='bold', fontsize=14)
ax2.grid(axis='x', alpha=0.3)

# Add annotation explaining the finding
if len(significant_industries) > 0:
    most_positive = significant_industries.iloc[0]['Short_Industry'] if significant_industries.iloc[0]['Correlation'] > 0 else "No industries"
    most_negative = significant_industries.iloc[-1]['Short_Industry'] if significant_industries.iloc[-1]['Correlation'] < 0 else "No industries"
    
    ax2.text(0.5, 0.02, 
             f"Different industries show opposite patterns:\n{most_positive}: higher emissions → more ambitious targets\n{most_negative}: higher emissions → less ambitious targets", 
             transform=ax2.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 3.3 Timeline vs. Emissions Intensity Correlation at Different Levels
ax3 = fig3.add_subplot(gs3[1, 0])

# First calculate the timeline for each target (years between base and implementation year)
target_timeline = target_level_analysis.copy()
target_timeline = target_timeline.dropna(subset=['CBN_TARGET_BASE_YEAR', 'CBN_TARGET_IMP_YEAR'])
target_timeline['timeline'] = target_timeline['CBN_TARGET_IMP_YEAR'] - target_timeline['CBN_TARGET_BASE_YEAR']

# Filter out negative or unusually large timelines
target_timeline = target_timeline[
    (target_timeline['timeline'] >= 0) &
    (target_timeline['timeline'] <= 30)  # Cap at 30 years to remove outliers
]

# Individual level: Calculate correlation between timeline and emissions intensity
# Merge timeline data with emissions intensity
individual_data = pd.merge(
    target_timeline[['ISSUERID', 'timeline']],
    company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']],
    on='ISSUERID',
    how='inner'
)

# Filter out zero or missing emissions
individual_data = individual_data[individual_data['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]

# Calculate log emission intensity for better distribution
individual_data['log_intensity'] = np.log10(individual_data['CARBON_EMISSIONS_SCOPE_12_INTEN'])

# Calculate correlation at individual level
individual_corr, ind_p = stats.pearsonr(individual_data['log_intensity'], individual_data['timeline'])

# Industry level: Calculate correlation between average timeline and average emissions by industry
# First get industry data
industry_data = pd.merge(
    individual_data,
    company_emissions_df[['ISSUERID', 'NACE_CLASS_DESCRIPTION']].drop_duplicates(),
    on='ISSUERID',
    how='inner'
)

# Calculate average by industry
industry_agg = industry_data.groupby('NACE_CLASS_DESCRIPTION').agg({
    'timeline': 'mean',
    'log_intensity': 'mean',
    'ISSUERID': 'count'
}).reset_index()

# Filter to industries with enough companies
industry_agg = industry_agg[industry_agg['ISSUERID'] >= 10]

# Calculate correlation at industry level
industry_corr, ind_trend_p = stats.pearsonr(industry_agg['log_intensity'], industry_agg['timeline'])

# Country level: Calculate correlation between average timeline and average emissions by country
# First get country data
country_data = pd.merge(
    individual_data,
    company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company']].drop_duplicates(),
    on='ISSUERID',
    how='inner'
)

# Calculate average by country
country_agg = country_data.groupby('ISSUER_CNTRY_DOMICILE_company').agg({
    'timeline': 'mean',
    'log_intensity': 'mean',
    'ISSUERID': 'count'
}).reset_index()

# Filter to countries with enough companies
country_agg = country_agg[country_agg['ISSUERID'] >= 10]

# Calculate correlation at country level
country_corr, cntry_p = stats.pearsonr(country_agg['log_intensity'], country_agg['timeline'])

# Create data for the bar chart
levels = ['Individual', 'Industry', 'Country']
correlations = [individual_corr, industry_corr, country_corr]
p_values = [f"p<{0.001}" if p < 0.001 else f"p={p:.3f}" for p in [ind_p, ind_trend_p, cntry_p]]
colors = ['#3182bd', '#e6550d', '#31a354']

# Create the bar chart
bars = ax3.bar(levels, correlations, color=colors, width=0.6)

# Add correlation values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'r = {height:.3f}',
            ha='center', va='bottom', fontweight='bold')
    
    # Add p-values below the bars
    ax3.text(bar.get_x() + bar.get_width()/2., -0.05, 
             p_values[i], ha='center', fontsize=10)

# Set labels and title
ax3.set_ylabel('Correlation Coefficient (r)', fontsize=12)
ax3.set_title('3.3 Timeline vs. Emissions Intensity\nCorrelation Strengthens at Industry Level', fontweight='bold', fontsize=14)
ax3.set_ylim(-0.1, 0.6)  # Set y-axis limits
ax3.grid(axis='y', alpha=0.3)

# Add annotation explaining the finding
ax3.text(0.5, 0.05, "Companies in higher-emitting industries\ntend to set longer-term targets", 
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 3.4 Target Aggressiveness vs. Emissions Trend at Different Levels
ax4 = fig3.add_subplot(gs3[1, 1])

# Individual level: Calculate correlation between aggressiveness and emissions trend
# Merge target ambition with emissions trend
individual_agg_trend = pd.merge(
    target_level_analysis[['ISSUERID', 'CBN_TARGET_REDUC_PCT']],
    company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']],
    on='ISSUERID',
    how='inner'
)

# Filter out missing values
individual_agg_trend = individual_agg_trend.dropna(subset=['CBN_TARGET_REDUC_PCT', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])

# Calculate correlation at individual level
ind_agg_corr, ind_agg_p = stats.pearsonr(individual_agg_trend['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], 
                                       individual_agg_trend['CBN_TARGET_REDUC_PCT'])

# Industry level: Calculate correlation between average aggressiveness and average trend by industry
# First get industry data
industry_agg_trend = pd.merge(
    individual_agg_trend,
    company_emissions_df[['ISSUERID', 'NACE_CLASS_DESCRIPTION']].drop_duplicates(),
    on='ISSUERID',
    how='inner'
)

# Calculate average by industry
industry_trend_agg = industry_agg_trend.groupby('NACE_CLASS_DESCRIPTION').agg({
    'CBN_TARGET_REDUC_PCT': 'mean',
    'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean',
    'ISSUERID': 'count'
}).reset_index()

# Filter to industries with enough companies
industry_trend_agg = industry_trend_agg[industry_trend_agg['ISSUERID'] >= 10]

# Calculate correlation at industry level
ind_trend_corr, ind_trend_p = stats.pearsonr(industry_trend_agg['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], 
                                           industry_trend_agg['CBN_TARGET_REDUC_PCT'])

# Country level: Calculate correlation between average aggressiveness and average trend by country
# First get country data
country_agg_trend = pd.merge(
    individual_agg_trend,
    company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company']].drop_duplicates(),
    on='ISSUERID',
    how='inner'
)

# Calculate average by country
country_trend_agg = country_agg_trend.groupby('ISSUER_CNTRY_DOMICILE_company').agg({
    'CBN_TARGET_REDUC_PCT': 'mean',
    'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'mean',
    'ISSUERID': 'count'
}).reset_index()

# Filter to countries with enough companies
country_trend_agg = country_trend_agg[country_trend_agg['ISSUERID'] >= 10]

# Calculate correlation at country level
country_trend_corr, country_trend_p = stats.pearsonr(country_trend_agg['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], 
                                                   country_trend_agg['CBN_TARGET_REDUC_PCT'])

# Create data for the bar chart
levels = ['Individual', 'Industry', 'Country']
correlations = [ind_agg_corr, ind_trend_corr, country_trend_corr]
p_values = [f"p<{0.001}" if p < 0.001 else f"p={p:.3f}" for p in [ind_agg_p, ind_trend_p, country_trend_p]]
colors = ['#3182bd', '#e6550d', '#31a354']

# Create the bar chart
bars = ax4.bar(levels, correlations, color=colors, width=0.6)

# Add correlation values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height - 0.03,
            f'r = {height:.3f}',
            ha='center', va='top', fontweight='bold', color='white' if height < -0.15 else 'black')
    
    # Add p-values below the bars
    ax4.text(bar.get_x() + bar.get_width()/2., 0.01, 
             p_values[i], ha='center', fontsize=10)

# Set labels and title
ax4.set_ylabel('Correlation Coefficient (r)', fontsize=12)
ax4.set_title('3.4 Target Aggressiveness vs. Emissions Trend\nCorrelation Strengthens at Country Level', fontweight='bold', fontsize=14)
ax4.set_ylim(-0.35, 0.05)  # Set y-axis limits
ax4.grid(axis='y', alpha=0.3)

# Add annotation explaining the finding
ax4.text(0.5, 0.8, "Companies in countries with better emissions trends\ntend to set more aggressive targets", 
         transform=ax4.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# Add an overall title for this group
fig3.suptitle('Group 3: Geographic and Industry Context Matters', 
              fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
# ===== VISUALIZATION =====
print("\n--- VISUALIZATION ---")
print("=" * 80)
print("Generating visualizations...")

# Set up the figure with GridSpec for better layout control

# ===== OUTPUT =====
print("\nSaving visualizations...")

# Save the figures
save_figure(fig3, 'fig03_geographic_industry_context')
print("✓ Saved: Geographic and Industry Context visualization")

# Create a figure with 4 subplots for Group 4
fig4 = plt.figure(figsize=(18, 20))
gs4 = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.4, wspace=0.3)

# 4.1 Target Ambition Weakly Linked to Emissions Intensity
ax1 = fig4.add_subplot(gs4[0, 0])

ambition_intensity = pd.merge(
    target_level_analysis[['ISSUERID', 'CBN_TARGET_REDUC_PCT']].dropna(subset=['CBN_TARGET_REDUC_PCT']),
    company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].dropna(subset=['CARBON_EMISSIONS_SCOPE_12_INTEN']),
    on='ISSUERID',
    how='inner'
)

ambition_intensity = ambition_intensity[ambition_intensity['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]
ambition_intensity = ambition_intensity.groupby('ISSUERID').agg({
    'CBN_TARGET_REDUC_PCT': 'mean',
    'CARBON_EMISSIONS_SCOPE_12_INTEN': 'first'
}).reset_index()

ambition_intensity['log_intensity'] = np.log10(ambition_intensity['CARBON_EMISSIONS_SCOPE_12_INTEN'])

ax1.scatter(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'], 
           alpha=0.2, s=20, c='#3182bd')

corr, p_value = stats.pearsonr(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'])

m, b = np.polyfit(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'], 1)
x_line = np.linspace(min(ambition_intensity['log_intensity']), max(ambition_intensity['log_intensity']), 100)
ax1.plot(x_line, m * x_line + b, color='red', linestyle='--', linewidth=2)

ax1.text(0.05, 0.95, f"r = {corr:.3f}, p = {p_value:.3f}", 
         transform=ax1.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

ax1.set_xlabel('Log₁₀ Emissions Intensity (Scope 1+2)', fontsize=12)
ax1.set_ylabel('Target Reduction Percentage (%)', fontsize=12)
ax1.set_title('4.1 Target Ambition Weakly Linked to Emissions Intensity', fontweight='bold', fontsize=14)
ax1.grid(alpha=0.3)

ax1.text(0.5, 0.05, "Higher emissions intensity doesn't lead\nto more ambitious reduction targets", 
         transform=ax1.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 4.2 Timeline Unrelated to Emissions Trend
ax2 = fig4.add_subplot(gs4[0, 1])

timeline_trend = target_level_analysis.copy()
timeline_trend = timeline_trend.dropna(subset=['CBN_TARGET_BASE_YEAR', 'CBN_TARGET_IMP_YEAR'])
timeline_trend['timeline'] = timeline_trend['CBN_TARGET_IMP_YEAR'] - timeline_trend['CBN_TARGET_BASE_YEAR']

timeline_trend = timeline_trend[
    (timeline_trend['timeline'] >= 0) &
    (timeline_trend['timeline'] <= 30)  # Cap at 30 years to remove outliers
]

timeline_emissions = pd.merge(
    timeline_trend[['ISSUERID', 'timeline']],
    company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']],
    on='ISSUERID',
    how='inner'
)

timeline_emissions = timeline_emissions.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
timeline_emissions = timeline_emissions.groupby('ISSUERID').agg({
    'timeline': 'mean',
    'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': 'first'
}).reset_index()

ax2.scatter(timeline_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], timeline_emissions['timeline'], 
           alpha=0.2, s=20, c='#3182bd')

corr, p_value = stats.pearsonr(timeline_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], timeline_emissions['timeline'])

m, b = np.polyfit(timeline_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], timeline_emissions['timeline'], 1)
x_line = np.linspace(min(timeline_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']), 
                    max(timeline_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']), 100)
ax2.plot(x_line, m * x_line + b, color='red', linestyle='--', linewidth=2)

ax2.text(0.05, 0.95, f"r = {corr:.3f}, p = {p_value:.3f}", 
         transform=ax2.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

ax2.set_xlabel('Emissions Trend (3Y, Scope 1+2) %', fontsize=12)
ax2.set_ylabel('Target Timeline (years)', fontsize=12)
ax2.set_title('4.2 Timeline Unrelated to Emissions Trend', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

ax2.text(0.5, 0.05, "Recent emissions performance has no relationship\nwith how far in the future targets are set", 
         transform=ax2.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 4.3 Target Ambition vs. Emissions Intensity (Individual Companies) - FIXED
ax3 = fig4.add_subplot(gs4[1, 0])

scatter = ax3.scatter(
    ambition_intensity['log_intensity'], 
    ambition_intensity['CBN_TARGET_REDUC_PCT'],
    alpha=0.3, 
    s=20, 
    c='#3182bd'
)

corr, p_value = stats.pearsonr(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'])

m, b = np.polyfit(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'], 1)
x_line = np.linspace(min(ambition_intensity['log_intensity']), max(ambition_intensity['log_intensity']), 100)
ax3.plot(x_line, m * x_line + b, color='red', linestyle='--', linewidth=2)

ax3.text(0.05, 0.95, f"r = {corr:.3f}, p = {p_value:.3f}", 
         transform=ax3.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

ax3.text(0.05, 0.85, f"n = {len(ambition_intensity)}", 
         transform=ax3.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

ax3.set_xlabel('Log₁₀ Emissions Intensity (Scope 1+2)', fontsize=12)
ax3.set_ylabel('Target Reduction Percentage (%)', fontsize=12)
ax3.set_title('4.3 No Clear Pattern Between Emissions Intensity\nand Target Ambition (Individual Companies View)', 
              fontweight='bold', fontsize=14)
ax3.grid(alpha=0.3)

ax3.text(0.5, 0.05, "Expected pattern would show higher targets for higher\nemissions, but data shows no systematic relationship", 
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 4.4 Correlation Between Emissions Intensity vs. 3-Year Trend
ax4 = fig4.add_subplot(gs4[1, 1])

intensity_trend = pd.merge(
    company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']],
    company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']],
    on='ISSUERID',
    how='inner'
)

intensity_trend = intensity_trend[
    (intensity_trend['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0) &
    (intensity_trend['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].notna())
]

intensity_trend['log_intensity'] = np.log10(intensity_trend['CARBON_EMISSIONS_SCOPE_12_INTEN'])

ax4.scatter(intensity_trend['log_intensity'], intensity_trend['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], 
           alpha=0.2, s=20, c='#3182bd')

corr, p_value = stats.pearsonr(intensity_trend['log_intensity'], intensity_trend['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])

m, b = np.polyfit(intensity_trend['log_intensity'], intensity_trend['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], 1)
x_line = np.linspace(min(intensity_trend['log_intensity']), max(intensity_trend['log_intensity']), 100)
ax4.plot(x_line, m * x_line + b, color='red', linestyle='--', linewidth=2)

ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)

ax4.text(0.05, 0.95, f"r = {corr:.3f}, p = {p_value:.3f}", 
         transform=ax4.transAxes, fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

ax4.set_xlabel('Log₁₀ Emissions Intensity', fontsize=12)
ax4.set_ylabel('3-Year Emission Trend (%)', fontsize=12)
ax4.set_title('4.4 Weak Link Between Emission Level and Trend', 
             fontweight='bold', fontsize=14)
ax4.grid(alpha=0.3)

ax4.text(0.5, 0.05, "Companies with higher emissions intensity show only\nslight tendency toward better emission reductions", 
         transform=ax4.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

fig4.suptitle('Group 4: Counterintuitive or Missing Relationships', 
              fontsize=18, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
save_figure(fig4, 'fig04_counterintuitive_relationships')
print("✓ Saved: Counterintuitive Relationships visualization")

# Print completion message
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated the following visualizations:")
print("1. fig01_emissions_and_targets")
print("2. fig02_target_types_effectiveness")
print("3. fig03_geographic_industry_context")
print("4. fig04_counterintuitive_relationships")
print("\nVisualizations saved to:")
print(f"- PNG: {os.path.join(paths.OUTPUT_DIR, 'png/')}")
print(f"- PDF: {os.path.join(paths.OUTPUT_DIR, 'pdf/')}")

# Print any warnings that were suppressed
if len([w for w in warnings.filters if w[0] == 'ignore']) > 0:
    print("\nNote: Some warnings were suppressed during execution. Use warnings.simplefilter('default')")
    print("at the start of the script to see all warnings.")