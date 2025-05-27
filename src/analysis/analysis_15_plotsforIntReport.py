"""
Generate visualizations for the internship report on emissions reduction targets and trends.

Input DataFrames:
1. company_emissions_df: Company emissions and intensity metrics
2. targets_df: Company emissions reduction targets and progress data

Key Analyses Performed:
1. Analyze relationship between target ambition and emission trends
2. Compare target achievement rates across different sectors
3. Visualize country-level patterns in target setting and achievement
4. Generate correlation plots for key metrics

Outputs:
- Multiple visualization files in the report_visuals directory
- Console logs of key statistics and analysis results
"""

# ===== IMPORTS =====
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.gridspec as gridspec

from src.utils.visualization_utils import (
    setup_plot_style,
    save_figure,
    add_correlation_annotation,
    plot_trendline,
    create_scatter_with_correlation,
    create_bar_chart,
)
from src.utils.analysis_utils import winsorize

# ===== CONSTANTS =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MERGED_PATH = os.path.join(DATA_DIR, "company_emissions_merged.csv")
TARGETS_PATH = os.path.join(DATA_DIR, "Reduktionsziele Results - 20241212 15_49_29.csv")

# Load data directly
print("Loading data...")
company_emissions_df = pd.read_csv(MERGED_PATH)
targets_df = pd.read_csv(TARGETS_PATH).dropna(axis=1, how='all')

print(f"Loaded company_emissions_df: shape={company_emissions_df.shape}, unique_ISSUERIDs={company_emissions_df['ISSUERID'].nunique():,}")
print(f"Loaded targets_df: shape={targets_df.shape}, unique_ISSUERIDs={targets_df['ISSUERID'].nunique():,}")

# Set up the visualization style
setup_plot_style()

# First, create the target level analysis dataframe from the raw data
target_level_analysis = targets_df.copy()

# Merge with emissions data to get emissions-related metrics
emissions_columns = ['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',
                    'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_3',
                    'ISSUER_CNTRY_DOMICILE_company', 'NACE_CLASS_DESCRIPTION']

# Merge target data with emissions data
target_level_analysis = pd.merge(
    target_level_analysis,
    company_emissions_df[emissions_columns].drop_duplicates('ISSUERID'),
    on='ISSUERID',
    how='left'
)

# Part 1: Scale of Emissions and Target-Setting Behavior
fig1 = plt.figure(figsize=(16, 8)) # Adjust figure size for side-by-side plots
gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.4) # 1 row, 2 columns, adjust horizontal space

# 1.1 Target Count vs. Log Absolute Emissions
ax1 = fig1.add_subplot(gs1[0, 0])

# Count the number of target records per company
target_counts = target_level_analysis.groupby('ISSUERID').size().reset_index(name='target_count')

# Get emissions data for each company
company_emissions = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12']].drop_duplicates()

# Merge target counts with emissions data
company_target_data = pd.merge(company_emissions, target_counts, on='ISSUERID', how='inner')

# Filter out companies with zero or missing emissions
company_target_data = company_target_data[
    (company_target_data['CARBON_EMISSIONS_SCOPE_12'] > 0) &
    (company_target_data['target_count'] > 0)
]

# Create scatter plot with correlation statistics and trendline
title = '5.2.1 Higher-Emitting Companies Set More Targets'
x_data = np.log10(company_target_data['CARBON_EMISSIONS_SCOPE_12'])
y_data = company_target_data['target_count']
xlabel = 'Log₁₀ Absolute Emissions (Scope 1+2)'
ylabel = 'Number of Carbon Targets'
annotation = "Companies with higher absolute emissions tend\nto establish more carbon reduction targets"

# Use log scale for y-axis
create_scatter_with_correlation(
    ax1, x_data, y_data, title, xlabel, ylabel,
    log_y=True, annotation_text=annotation
)

# 1.3 Target Ambition vs. Absolute Emissions
ax3 = fig1.add_subplot(gs1[0, 1]) # Place plot 1.3 in the second column

# Calculate average target ambition per company
company_ambition = target_level_analysis.groupby('ISSUERID')['CBN_TARGET_REDUC_PCT'].mean().reset_index()

# Merge with emissions data
company_ambition_emissions = pd.merge(
    company_ambition,
    company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12']],
    on='ISSUERID',
    how='inner'
)

# Filter out companies with zero or missing values
company_ambition_emissions = company_ambition_emissions[
    (company_ambition_emissions['CARBON_EMISSIONS_SCOPE_12'] > 0) &
    (company_ambition_emissions['CBN_TARGET_REDUC_PCT'].notna())
]

# Create log transformed x-values
company_ambition_emissions['log_emissions'] = np.log10(company_ambition_emissions['CARBON_EMISSIONS_SCOPE_12'])

# Create density heatmap instead of scatter
hb = ax3.hexbin(
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT'],
    gridsize=30, cmap='Oranges', mincnt=1, bins='log'
)

# Add correlation statistics
corr, p_value = add_correlation_annotation(
    ax3,
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT']
)

# Add trendline
plot_trendline(
    ax3,
    company_ambition_emissions['log_emissions'],
    company_ambition_emissions['CBN_TARGET_REDUC_PCT']
)

# Set labels and title
ax3.set_xlabel('Log₁₀ Absolute Emissions (Scope 1+2)', fontsize=12)
ax3.set_ylabel('Target Reduction Percentage (%)', fontsize=12)
ax3.set_title('5.2.2 Target Ambition vs. Absolute Emissions\nDensity Plot Shows Relationship Pattern', fontweight='bold', fontsize=14)
ax3.grid(alpha=0.3)

# Add colorbar for density
cb = plt.colorbar(hb, ax=ax3)
cb.set_label('Log Number of Companies')

# Add annotation based on correlation direction
if corr < 0:
    annotation = "Higher-emitting companies tend to set\nless ambitious reduction targets"
else:
    annotation = "Higher-emitting companies tend to set\nmore ambitious reduction targets"

ax3.text(0.5, 0.05, annotation,
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# Add an overall title for this group
fig1.suptitle('5.2. Scale of Emissions and Target-Setting Behavior',
              fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.subplots_adjust(top=0.88, wspace=0.4) # Adjust top and horizontal space

save_figure(fig1, 'figure1_internship')

# Visualization of Carbon Target Findings - Group 2: Types of Targets and Their Effectiveness
fig2 = plt.figure(figsize=(16, 8)) # Adjust figure size for side-by-side plots
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4) # 1 row, 2 columns, adjust horizontal space

# 2.1 Emissions by Target Type (Absolute vs. Intensity-based)
ax1 = fig2.add_subplot(gs2[0, 0])

# First, make sure we have the right emissions intensity data
# We'll need to merge target data with emissions intensity data
emissions_intensity_data = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()

# Create a clean target type dataset
target_types = target_level_analysis[['ISSUERID', 'TARGET_CARBON_TYPE']].dropna(subset=['TARGET_CARBON_TYPE'])

# Merge to get emissions intensity for each target type
target_type_emissions = pd.merge(
    target_types,
    emissions_intensity_data,
    on='ISSUERID',
    how='inner'
)

# Filter to positive values only
target_type_emissions = target_type_emissions[target_type_emissions['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]

# Handle outliers by using winsorization - cap extreme values at percentiles
def winsorize(data, lower_percentile=5, upper_percentile=95):
    low_val = np.percentile(data, lower_percentile)
    high_val = np.percentile(data, upper_percentile)
    return np.clip(data, low_val, high_val)

# Parameter to toggle winsorization
use_winsorization = False  # Set to False for the internship report

# Group by target carbon type and calculate robust statistics
emissions_by_type = []
for type_name, group in target_type_emissions.groupby('TARGET_CARBON_TYPE'):
    if len(group) >= 30:  # Only include types with enough samples
        # Apply winsorization if enabled
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

# Convert to DataFrame
emissions_by_type = pd.DataFrame(emissions_by_type)

# Sort by count to get most common types first
emissions_by_type = emissions_by_type.sort_values('count', ascending=False)

# Create color mapping based on type
target_type_colors = {
    'Absolute': '#1a9850',
    'Production intensity': '#A63603',
    'Sales intensity': '#74add1',
    'Others?': '#984ea3'
}

# Assign colors based on target type
colors = [target_type_colors.get(t, '#999999') for t in emissions_by_type['TARGET_CARBON_TYPE']]

# Create the bar chart using MEDIAN instead of mean to further reduce outlier impact
# Remove the black edgecolor by setting it to the same as the bar color
bars = ax1.bar(
    emissions_by_type['TARGET_CARBON_TYPE'],
    emissions_by_type['median'],  # Use median instead of mean
    yerr=emissions_by_type['std'] / np.sqrt(emissions_by_type['count']),  # Standard error
    color=colors,
    edgecolor=colors,  # Remove black border by matching the fill color
    width=0.6,
    capsize=5
)

# Add values on top of bars, with fewer decimal places
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
            f'{height:.1f}',  # Changed from .2f to .1f for fewer decimals
            ha='center', va='bottom', fontweight='bold')

# Set labels and title
ax1.set_ylabel('Median Emissions Intensity (Scope 1+2)', fontsize=12)
title_suffix = " (with Winsorization)" if use_winsorization else " (No Winsorization)"
ax1.set_title(f'6.1.1 Emissions Intensity by Target Type{title_suffix}', fontweight='bold', fontsize=14)
ax1.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

# Improved annotation with direction of difference
ax1.text(0.5, 0.05, "Companies using absolute targets have lower emissions\nintensity levels than those using production intensity targets",
         transform=ax1.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# 6.1.2 Target Progress and Emission Trends
ax3 = fig2.add_subplot(gs2[0, 1]) # Place plot 6.1.2 in the second column

# Get data with both progress percentage and emissions trend
progress_trend_data = target_level_analysis.dropna(subset=['TARGET_CARBON_PROGRESS_PCT', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])

# Filter out extreme progress values that are likely errors
progress_trend_data = progress_trend_data[
    (progress_trend_data['TARGET_CARBON_PROGRESS_PCT'] >= 0) &
    (progress_trend_data['TARGET_CARBON_PROGRESS_PCT'] <= 200)
]

# Handle outliers in the emissions trend data
if use_winsorization:
    progress_trend_data['robust_trend'] = progress_trend_data.groupby(
        pd.qcut(progress_trend_data['TARGET_CARBON_PROGRESS_PCT'], 8, duplicates='drop')
    )['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].transform(
        lambda x: winsorize(x, 5, 95)
    )
else:
    progress_trend_data['robust_trend'] = progress_trend_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']

# Create progress bins
progress_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]
bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%', '100-125%', '125-150%', '150-175%', '175-200%']

# Bin the progress data
progress_trend_data['progress_bin'] = pd.cut(
    progress_trend_data['TARGET_CARBON_PROGRESS_PCT'],
    bins=progress_bins,
    labels=bin_labels,
    include_lowest=True
)

# Calculate average trend by progress bin
trend_by_progress = progress_trend_data.groupby('progress_bin')['robust_trend'].agg(
    ['mean', 'count', 'std']).reset_index()

# Improved color scheme based on progress bins with a continuous gradient from light to dark green
# Use a sequential colormap that starts with light green and ends with dark green
greens = plt.cm.Greens(np.linspace(0.3, 0.9, len(trend_by_progress)))
colors = greens

# Create the bar chart with no black borders
pos = np.arange(len(trend_by_progress))  # Create positions for the bars
bars = ax3.bar(
    pos,
    trend_by_progress['mean'],
    yerr=trend_by_progress['std'] / np.sqrt(trend_by_progress['count']),  # Standard error
    color=colors,
    edgecolor=colors,  # Remove black borders
    width=0.7,
    capsize=5
)

# Add values on the bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height - 0.3 if height < 0 else height + 0.3,
            f'{height:.1f}%',
            ha='center', va='top' if height < 0 else 'bottom',
            fontweight='bold', color='white' if height < -5 else 'black')

# Set labels and title
ax3.set_xlabel('TARGET_CARBON_PROGRESS_PCT\n(% progress toward reduction target, >100% = target exceeded)', fontsize=12)
ax3.set_ylabel('3-Year Emissions Trend (%) (Scope 1+2)', fontsize=12)
ax3.set_title('6.1.2 Target Progress and Emissions Trends (No Winsorization)', fontweight='bold', fontsize=14)
ax3.grid(axis='y', alpha=0.3)

# Fix the FixedFormatter warning
ax3.set_xticks(pos)
ax3.set_xticklabels(trend_by_progress['progress_bin'], rotation=45, ha='right')

# Find the best performing progress category
best_idx = trend_by_progress['mean'].idxmin()
best_progress = trend_by_progress.iloc[best_idx]

# Add annotation explaining the finding with better explanation
ax3.text(0.5, 0.1,
         f"Targets with {best_progress['progress_bin']} progress show best trend\n({best_progress['mean']:.1f}%, n={best_progress['count']})\nHigher values indicate targets exceeded by a larger margin",
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

# Add an overall title for this group
fig2.suptitle('6.1. Types of Targets and Their Effectiveness',
              fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.subplots_adjust(top=0.88, wspace=0.4) # Adjust top and horizontal space

save_figure(fig2, 'figure2_internship')

# Visualization of Carbon Target Findings - Group 3: Geographic and Industry Context Matters
fig3 = plt.figure(figsize=(10, 8))
gs3 = gridspec.GridSpec(1, 1, figure=fig3, hspace=0.6, wspace=0.5)

# 3.1 Country-Level Target Count vs. Emission Trends
ax1 = fig3.add_subplot(gs3[0, 0])

# Get country data
country_emissions = company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()

# Calculate average emission trend by country
country_trends = country_emissions.groupby('ISSUER_CNTRY_DOMICILE_company')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean().reset_index()

# Count the number of target records per company per country
company_targets = target_level_analysis.groupby(['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company']).size().reset_index(name='target_count')

# Calculate average targets per company for each country
country_avg_targets = company_targets.groupby('ISSUER_CNTRY_DOMICILE_company').agg({
    'target_count': 'mean',
    'ISSUERID': 'nunique'
}).reset_index()
country_avg_targets.rename(columns={'ISSUERID': 'company_count'}, inplace=True)

# Merge trend and target data
country_data = pd.merge(country_trends, country_avg_targets, on='ISSUER_CNTRY_DOMICILE_company', how='inner')

# Filter to countries with enough companies (at least 10)
country_data = country_data[country_data['company_count'] >= 10]

# Create scatter plot
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
ax1.set_title(f'Country-Level Target Count vs. Emission Trends\nBased on {len(country_data)} Countries with ≥10 Companies', fontweight='bold', fontsize=14)
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

# Add an overall title for this group
fig3.suptitle('5.1.1. Geographic and Industry Context Matters',
              fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.subplots_adjust(top=0.87, hspace=0.6, wspace=0.4)

save_figure(fig3, 'figure3_internship')

print("Script completed. Figures saved in output_graphs/png/ and output_graphs/pdf/ directories as figure15_internship, figure15_2_internship, and figure15_3_internship")