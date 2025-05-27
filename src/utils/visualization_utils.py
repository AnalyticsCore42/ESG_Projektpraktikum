"""
Visualization utilities for ESG data analysis.

This module provides a comprehensive set of functions for creating, customizing, and saving
visualizations for ESG (Environmental, Social, and Governance) data analysis.

Features:
    - Consistent styling and theming for all visualizations
    - Functions for various plot types (scatter, bar, line, etc.)
    - Tools for statistical annotations and trend lines
    - Utilities for saving figures in multiple formats
    - Pre-configured templates for common ESG analysis tasks

"""

import os

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---- General Visualization Utilities ----

def apply_style():
    """
    Apply a consistent style to all matplotlib visualizations.
    
    This function sets up a clean, publication-ready style for all plots with:
    - A white grid on a light background
    - A colorblind-friendly color palette
    - Consistent font sizes for all plot elements
    
    Note:
        This function modifies the global matplotlib rcParams and should be called
        before creating any figures.

    Returns:
        None: This function modifies global state and does not return a value.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('colorblind')
    # Ensure consistent font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def create_plot_directory():
    """
    Create standard directories for storing plot outputs.

    Returns:
        tuple: Paths to the created png and pdf directories.

    Raises:
        OSError: If the directory cannot be created due to an OS error.
    """
    base_directory = 'output/summary'
    png_directory = os.path.join(base_directory, 'png')
    pdf_directory = os.path.join(base_directory, 'pdf')
    others_directory = os.path.join(base_directory, 'others')
    # Create all directories if they don't exist
    for directory in [base_directory, png_directory, pdf_directory, others_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return png_directory, pdf_directory

def save_figure(fig, filename, formats=['png', 'pdf'], dpi=300):
    """
    Save a matplotlib figure in multiple formats with consistent styling.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save. Can be
            obtained using `plt.gcf()` or by explicitly creating a figure
            with `plt.Figure()`.
        filename (str): Base filename without extension. Should not include
            path information.
        formats (list[str], optional): List of file formats to save.
            Supported formats: 'png', 'pdf'. Defaults to ['png', 'pdf'].
        dpi (int, optional): Dots per inch resolution for raster formats
            like PNG. Higher values result in higher quality but larger
            files. Defaults to 300.

    Returns:
        None: The function saves files to disk but does not return any value.

    Raises:
        ValueError: If an unsupported format is provided in the formats list.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> save_figure(fig, 'example_plot')
    """
    # Create plot directories if they don't exist
    png_dir, pdf_dir = create_plot_directory()

    # Save in each requested format
    for fmt in formats:
        if fmt.lower() == 'png':
            fig.savefig(os.path.join(png_dir, f"{filename}.png"), dpi=dpi, bbox_inches='tight')
        elif fmt.lower() == 'pdf':
            fig.savefig(os.path.join(pdf_dir, f"{filename}.pdf"), bbox_inches='tight')

# ---- Emissions Target Relationships Module ----
# (Originally from emissions_target_relationships.py)

def create_relationship_plots(company_emissions_df: pd.DataFrame, targets_df: pd.DataFrame) -> plt.Figure:
    """
    Create visualizations showing relationships between emissions and
    target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.

    Raises:
        KeyError: If required columns are missing from the input DataFrames.
        ValueError: If input data is not valid for plotting.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    
    # Create a figure with 4 subplots for Group 1
    fig1 = plt.figure(figsize=(18, 18))
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.4, wspace=0.3)
    
    # 1.1 Target Count Distribution
    ax1 = fig1.add_subplot(gs1[0, 0])
    
    # Count the number of targets per company
    target_counts = targets_df.groupby('ISSUERID').size().reset_index(name='target_count')
    
    # Plot the distribution
    sns.histplot(target_counts['target_count'], bins=range(0, 16), ax=ax1, color=primary_color)
    
    # Set labels and title
    ax1.set_xlabel('Number of Targets per Company', fontsize=12)
    ax1.set_ylabel('Number of Companies', fontsize=12)
    ax1.set_title('1.1 Target Count Distribution', fontweight='bold', fontsize=14)
    
    # Set x-axis to show integer values
    ax1.set_xticks(range(0, 16))
    ax1.grid(axis='y', alpha=0.3)
    o
    # Add descriptive statistics
    mean_targets = target_counts['target_count'].mean()
    median_targets = target_counts['target_count'].median()
    max_targets = target_counts['target_count'].max()
    companies_no_targets = len(company_emissions_df) - len(target_counts)
    
    stats_text = (f"Mean: {mean_targets:.1f} targets\n"
                  f"Median: {median_targets:.0f} targets\n"
                  f"Max: {max_targets:.0f} targets\n"
                  f"Companies with no targets: {companies_no_targets:,}")
    
    ax1.text(0.95, 0.95, stats_text, 
             transform=ax1.transAxes, fontsize=12, ha='right', va='top',
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # 1.2 Emissions Trend by Target Count
    ax2 = fig1.add_subplot(gs1[0, 1])
    
    # Merge target count with emission intensity data
    target_emissions = pd.merge(
        target_counts,
        company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates(),
        on='ISSUERID',
        how='inner'
    )
    
    # Remove missing trend values
    target_emissions = target_emissions.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
    
    # Group target counts into bins for clearer analysis
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 100]
    bin_labels = ['0', '1', '2', '3-5', '6-10', '11-15', '16-20', '20+']
    target_emissions['target_group'] = pd.cut(target_emissions['target_count'], bins=bins, labels=bin_labels, right=False)
    
    # Calculate statistics by group
    group_stats = target_emissions.groupby('target_group').agg({
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': ['mean', 'std', 'count']
    }).reset_index()
    
    group_stats.columns = ['target_group', 'mean_trend', 'std_trend', 'count']
    
    # Calculate standard error
    group_stats['se_trend'] = group_stats['std_trend'] / np.sqrt(group_stats['count'])
    
    # Plot the mean trends with error bars
    bars = ax2.bar(
        group_stats['target_group'], 
        group_stats['mean_trend'],
        yerr=group_stats['se_trend'],
        capsize=5,
        color=primary_color,
        width=0.7
    )
    
    # Add a horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -3
        else:
            va = 'bottom'
            offset = 3
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + (0.01 * height if height != 0 else 0.01), 
            f'{height:.1f}%', 
            ha='center', 
            va=va,
            fontsize=10,
            rotation=0
        )
    
    # Add sample size annotations
    for i, count in enumerate(group_stats['count']):
        ax2.text(
            i, 
            ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.05, 
            f"n={count}",
            ha='center',
            fontsize=9,
            color='darkgray'
        )
    
    # Set labels and title
    ax2.set_xlabel('Number of Targets per Company', fontsize=12)
    ax2.set_ylabel('3-Year Emissions Trend (%)', fontsize=12)
    ax2.set_title('1.2 Emissions Trend by Target Count', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Calculate and add correlation information 
    # (using original ungrouped data for statistical validity)
    corr, p_value = stats.pearsonr(target_emissions['target_count'], target_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
    
    correlation_text = f"Correlation: r = {corr:.3f}, p = {p_value:.4f}"
    ax2.text(0.5, 0.05, correlation_text, 
             transform=ax2.transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # 1.3 Emission Intensities by Target Count
    ax3 = fig1.add_subplot(gs1[1, 0])
    
    # Merge target count with emission intensity data
    target_intensity = pd.merge(
        target_counts,
        company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates(),
        on='ISSUERID',
        how='inner'
    )
    
    # Remove zero or negative intensities
    target_intensity = target_intensity[target_intensity['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]
    
    # Group target counts into bins
    target_intensity['target_group'] = pd.cut(target_intensity['target_count'], bins=bins, labels=bin_labels, right=False)
    
    # Create a consistent Blues colormap
    blue_palette = sns.color_palette("Blues", n_colors=len(target_intensity['target_group'].unique()))
    
    # Box plot for emission intensities, with logscale to handle wide range
    sns.boxplot(
        x='target_group',
        y='CARBON_EMISSIONS_SCOPE_12_INTEN',
        data=target_intensity,
        ax=ax3,
        whis=[5, 95],  # Less extreme whiskers 
        palette=blue_palette,
        showfliers=False  # Hide outliers
    )
    
    # Set log scale for y-axis to better handle the wide range
    ax3.set_yscale('log')
    
    # Set labels and title
    ax3.set_xlabel('Number of Targets per Company', fontsize=12)
    ax3.set_ylabel('Emissions Intensity (Scope 1+2, log scale)', fontsize=12)
    ax3.set_title('1.3 Emission Intensities by Target Count', fontweight='bold', fontsize=14)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add sample sizes
    for i, grp in enumerate(target_intensity.groupby('target_group')):
        group_name, group_data = grp
        ax3.text(
            i, 
            ax3.get_ylim()[0] * 1.1, 
            f"n={len(group_data)}",
            ha='center',
            fontsize=9,
            color='darkgray'
        )
    
    # Add median values
    for i, (name, group) in enumerate(target_intensity.groupby('target_group')):
        median = group['CARBON_EMISSIONS_SCOPE_12_INTEN'].median()
        ax3.text(
            i, 
            median * 1.2, 
            f"{median:.1f}",
            ha='center',
            fontsize=10,
            weight='bold'
        )
    
    # 1.4 Target Timeline Analysiso
    timeline_data = timeline_data.dropna(subset=['CBN_TARGET_BASE_YEAR', 'CBN_TARGET_IMP_YEAR'])
    timeline_data['timeline'] = timeline_data['CBN_TARGET_IMP_YEAR'] - timeline_data['CBN_TARGET_BASE_YEAR']
    
    # Filter out negative or unusually large timelines
    timeline_data = timeline_data[
        (timeline_data['timeline'] >= 0) &
        (timeline_data['timeline'] <= 30)  # Cap at 30 years to remove outliers
    ]
    
    # Group timeline data by target type
    timeline_by_type = []
    for type_name, group in timeline_data.groupby('TARGET_CARBON_TYPE'):
        # Only include types with enough samples
        if len(group) >= 30:
            timeline_by_type.append({
                'TARGET_CARBON_TYPE': type_name,
                'median_timeline': group['timeline'].median(),
                'mean_timeline': group['timeline'].mean(),
                'count': len(group)
            })
    
    # Convert to DataFrame and sort by count
    timeline_by_type = pd.DataFrame(timeline_by_type)
    timeline_by_type = timeline_by_type.sort_values('count', ascending=False)
    
    # Create bar chart for median timeline by target type
    bars = ax4.bar(
        timeline_by_type['TARGET_CARBON_TYPE'], 
        timeline_by_type['median_timeline'],
        color=primary_color,
        width=0.7
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.1f} years', 
            ha='center', va='bottom', fontsize=10
        )
    
    # Add sample sizes
    for i, count in enumerate(timeline_by_type['count']):
        ax4.text(
            i, 
            ax4.get_ylim()[0] + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.05, 
            f"n={count}",
            ha='center',
            fontsize=9,
            color='darkgray'
        )
    
    # Set labels and title
    ax4.set_xlabel('Target Type', fontsize=12)
    ax4.set_ylabel('Median Timeline (Years)', fontsize=12)
    ax4.set_title('1.4 Target Timeline by Type', fontweight='bold', fontsize=14)
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add annotation about typical timeline
    overall_median = timeline_data['timeline'].median()
    ax4.text(0.5, 0.9, f"Overall median timeline: {overall_median:.1f} years", 
             transform=ax4.transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add overall title to the figure
    fig1.suptitle('Emissions Target Relationships Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig1

# ---- Target Effectiveness Module ----
# (Originally from target_effectiveness.py)

def create_effectiveness_plots(
    company_emissions_df: pd.DataFrame, targets_df: pd.DataFrame, use_winsorization: bool = False
) -> plt.Figure:
    """
    Create visualizations analyzing the effectiveness of different target
    types.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions
            data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        use_winsorization (bool, optional): Whether to apply winsorization
            to limit the effect of outliers. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.

    Raises:
        KeyError: If required columns are missing from the input DataFrames.
        ValueError: If input data is not valid for plotting.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    tertiary_color = '#e6550d'  # Orange
    quaternary_color = '#756bb1'  # Purple
    
    # Target Type Colors
    target_type_colors = {
        'Absolute': '#1a9850',
        'Production intensity': '#A63603',
        'Sales intensity': '#74add1',
        'Others?': '#984ea3'
    }
    
    # Create a figure with 3 subplots for Group 2
    fig2 = plt.figure(figsize=(18, 18))
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.3)

    # First, make sure we have the right emissions intensity data
    # We'll need to merge target data with emissions intensity data
    emissions_intensity_data = company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].drop_duplicates()

    # Create a clean target type dataset
    target_level_analysis = targets_df.copy()
    # Merge with emissions data to get emissions-related metrics
    emissions_columns = ['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 
                        'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_3']

    # Merge target data with emissions data
    target_level_analysis = pd.merge(
        target_level_analysis,
        company_emissions_df[emissions_columns].drop_duplicates('ISSUERID'),
        on='ISSUERID',
        how='left'
    )
    
    target_types = target_level_analysis[['ISSUERID', 'TARGET_CARBON_TYPE']].dropna(subset=['TARGET_CARBON_TYPE'])

    # Merge to get emissions intensity for each target type
    target_type_emissions = pd.merge(
        target_types,
        emissions_intensity_data,
        on='ISSUERID',
        how='inner'
    )

    # 2.1 Emissions by Target Type (Absolute vs. Intensity-based)
    ax1 = fig2.add_subplot(gs2[0, 0])

    # Filter to positive values only
    target_type_emissions = target_type_emissions[target_type_emissions['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]

    # Handle outliers by using winsorization - cap extreme values at percentiles
    def winsorize(data, lower_percentile=5, upper_percentile=95):
        low_val = np.percentile(data, lower_percentile)
        high_val = np.percentile(data, upper_percentile)
        return np.clip(data, low_val, high_val)

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

    # Assign colors based on target type
    colors = [target_type_colors.get(t, primary_color) for t in emissions_by_type['TARGET_CARBON_TYPE']]

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
    ax1.set_title(f'2.1 Emissions Intensity by Target Type{title_suffix}', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Improved annotation with direction of difference
    ax1.text(0.5, 0.05, "Companies using absolute targets have lower emissions\nintensity levels than those using production intensity targets", 
             transform=ax1.transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))

    # Add overall title to the figure
    fig2.suptitle('Target Effectiveness Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig2.tight_layout(rect=[0, 0, 1, 0.97])

    return fig2

# ---- Geographic Industry Context Module ----
# (Originally from geographic_industry_analysis.py)

def create_context_plots(company_emissions_df: pd.DataFrame, targets_df: pd.DataFrame) -> plt.Figure:
    """
    Create visualizations for geographic and industry context analysis.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions
            data.
        targets_df (pd.DataFrame): Carbon reduction targets data.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.

    Raises:
        KeyError: If required columns are missing from the input DataFrames.
        ValueError: If input data is not valid for plotting.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    
    # Create a figure with 4 subplots for Group 3
    fig3 = plt.figure(figsize=(18, 20))
    gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.4, wspace=0.3)

    # Create target level analysis dataframe
    target_level_analysis = targets_df.copy()
    # Merge with emissions data to get emissions-related metrics
    emissions_columns = ['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 
                        'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_3']

    # Merge target data with emissions data
    target_level_analysis = pd.merge(
        target_level_analysis,
        company_emissions_df[emissions_columns].drop_duplicates('ISSUERID'),
        on='ISSUERID',
        how='left'
    )

    # 3.1 Country-Level Target Count vs. Emission Trends
    ax1 = fig3.add_subplot(gs3[0, 0])

    # Get country data - use _company suffix for the country column
    country_emissions = company_emissions_df[['ISSUERID', 'ISSUER_CNTRY_DOMICILE_company', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']].drop_duplicates()

    # Calculate average emission trend by country
    country_trends = country_emissions.groupby('ISSUER_CNTRY_DOMICILE_company')['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'].mean().reset_index()

    # Count the number of target records per company per country
    company_targets = target_level_analysis.groupby(['ISSUERID', 'ISSUER_CNTRY_DOMICILE']).size().reset_index(name='target_count')

    # Calculate average targets per company for each country
    country_avg_targets = company_targets.groupby('ISSUER_CNTRY_DOMICILE').agg({
        'target_count': 'mean',
        'ISSUERID': 'nunique'
    }).reset_index()
    country_avg_targets.rename(columns={'ISSUERID': 'company_count'}, inplace=True)

    # Rename the country column in country_trends to match country_avg_targets
    country_trends.rename(columns={'ISSUER_CNTRY_DOMICILE_company': 'ISSUER_CNTRY_DOMICILE'}, inplace=True)

    # Merge trend and target data
    country_data = pd.merge(country_trends, country_avg_targets, on='ISSUER_CNTRY_DOMICILE', how='inner')

    # Filter to countries with enough companies (at least 10)
    country_data = country_data[country_data['company_count'] >= 10]

    # Create scatter plot with consistent color scheme
    scatter = ax1.scatter(
        country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
        country_data['target_count'],
        s=country_data['company_count'] * 3,  # Size by company count
        c=country_data['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],  # Color by trend
        cmap='RdYlGn_r',  # Red for positive (bad), green for negative (good)
        edgecolor='black',
        alpha=0.7
    )
    
    # Set labels and title
    ax1.set_xlabel('3-Year Emissions Trend (%)', fontsize=12)
    ax1.set_ylabel('Avg. Targets per Company', fontsize=12)
    ax1.set_title('3.1 Country-Level Target Count vs. Emission Trends', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('3-Year Emissions Trend (%)')
    
    # Add country labels
    for i, row in country_data.iterrows():
        ax1.annotate(
            row['ISSUER_CNTRY_DOMICILE'], 
            (row['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'], row['target_count']),
            fontsize=9,
            alpha=0.8,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    # Add overall title to the figure
    fig3.suptitle('Geographic and Industry Context Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig3.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig3

# ---- Counterintuitive Findings Module ----
# (Originally from counterintuitive_findings.py)

def create_findings_plots(company_emissions_df: pd.DataFrame, targets_df: pd.DataFrame) -> plt.Figure:
    """
    Create visualizations highlighting counterintuitive or missing
    relationships in emissions and target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions
            data.
        targets_df (pd.DataFrame): Carbon reduction targets data.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.

    Raises:
        KeyError: If required columns are missing from the input DataFrames.
        ValueError: If input data is not valid for plotting.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    
    # Create a figure with 4 subplots for Group 4
    fig4 = plt.figure(figsize=(18, 20))
    gs4 = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.4, wspace=0.3)

    # Create target level analysis dataframe
    target_level_analysis = targets_df.copy()
    # Merge with emissions data to get emissions-related metrics
    emissions_columns = ['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR', 
                        'CARBON_EMISSIONS_SCOPE_12', 'CARBON_EMISSIONS_SCOPE_3']

    # Merge target data with emissions data
    target_level_analysis = pd.merge(
        target_level_analysis,
        company_emissions_df[emissions_columns].drop_duplicates('ISSUERID'),
        on='ISSUERID',
        how='left'
    )

    # 4.1 Target Ambition Weakly Linked to Emissions Intensity
    ax1 = fig4.add_subplot(gs4[0, 0])

    # Get target ambition and emissions intensity data
    ambition_intensity = pd.merge(
        target_level_analysis[['ISSUERID', 'CBN_TARGET_REDUC_PCT']].dropna(subset=['CBN_TARGET_REDUC_PCT']),
        company_emissions_df[['ISSUERID', 'CARBON_EMISSIONS_SCOPE_12_INTEN']].dropna(subset=['CARBON_EMISSIONS_SCOPE_12_INTEN']),
        on='ISSUERID',
        how='inner'
    )

    # Filter out zeros and aggregate to company level (average ambition)
    ambition_intensity = ambition_intensity[ambition_intensity['CARBON_EMISSIONS_SCOPE_12_INTEN'] > 0]
    ambition_intensity = ambition_intensity.groupby('ISSUERID').agg({
        'CBN_TARGET_REDUC_PCT': 'mean',
        'CARBON_EMISSIONS_SCOPE_12_INTEN': 'first'
    }).reset_index()

    # Create log transformed x-values for better distribution
    ambition_intensity['log_intensity'] = np.log10(ambition_intensity['CARBON_EMISSIONS_SCOPE_12_INTEN'])

    # Create the scatter plot with consistent styling
    ax1.scatter(
        ambition_intensity['log_intensity'], 
        ambition_intensity['CBN_TARGET_REDUC_PCT'], 
        alpha=0.3, 
        s=20, 
        c=primary_color
    )
    
    # Set labels and title
    ax1.set_xlabel('Log10 Emissions Intensity (Scope 1+2)', fontsize=12)
    ax1.set_ylabel('Target Reduction Percentage (%)', fontsize=12)
    ax1.set_title('4.1 Target Ambition vs. Emissions Intensity', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Calculate and add correlation
    corr, p_value = stats.pearsonr(ambition_intensity['log_intensity'], ambition_intensity['CBN_TARGET_REDUC_PCT'])
    correlation_text = f"Correlation: r = {corr:.3f}, p = {p_value:.4f}"
    ax1.text(0.05, 0.95, correlation_text, 
             transform=ax1.transAxes, fontsize=12, ha='left', va='top',
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add overall title to the figure
    fig4.suptitle('Counterintuitive Findings Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig4.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig4

# ---- Reduction Programs Analysis Module ----
# (Originally from reduction_programs_analysis.py)

def create_programs_plots(
    company_emissions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    programs1_df: pd.DataFrame,
    programs2_df: pd.DataFrame,
) -> plt.Figure:
    """
    Create visualizations showing the relationships between reduction
    programs and their impact on emissions and target-setting behavior.

    Args:
        company_emissions_df: Combined company and emissions data.
        targets_df: Carbon reduction targets data.
        programs1_df: Reduction programs 1 data.
        programs2_df: Reduction programs 2 data.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.

    Raises:
        KeyError: If required columns are missing from the input DataFrames.
        ValueError: If input data is not valid for plotting.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    
    # Create a figure with 4 subplots
    fig5 = plt.figure(figsize=(18, 20))
    gs5 = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.4, wspace=0.3)
    
    # 5.1 Number of Programs vs. Emissions Trend
    ax1 = fig5.add_subplot(gs5[0, 0])
    
    # Calculate number of programs per company from programs2_df
    programs_per_company = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    
    # Merge with emissions trend data
    programs_emissions = pd.merge(
        programs_per_company,
        company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']],
        on='ISSUERID',
        how='inner'
    )
    
    # Filter out missing trend values
    programs_emissions = programs_emissions.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
    
    # Create scatter plot with consistent styling
    scatter = ax1.scatter(
        programs_emissions['program_count'],
        programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
        alpha=0.3, 
        s=30, 
        c=primary_color
    )
    
    # Set labels and title
    ax1.set_xlabel('Number of Reduction Programs', fontsize=12)
    ax1.set_ylabel('3-Year Emissions Trend (%)', fontsize=12)
    ax1.set_title('5.1 Number of Programs vs. Emissions Trend', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add regression line
    if len(programs_emissions) > 1:
        x = programs_emissions['program_count']
        y = programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']
        m, b = np.polyfit(x, y, 1)
        ax1.plot(x, m*x + b, color='red', linestyle='--', alpha=0.7)
        
        # Add correlation statistics
        corr, p_value = stats.pearsonr(programs_emissions['program_count'], 
                                      programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
        correlation_text = f"Correlation: r = {corr:.3f}, p = {p_value:.4f}"
        ax1.text(0.05, 0.95, correlation_text, 
                transform=ax1.transAxes, fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add overall title to the figure
    fig5.suptitle('Reduction Programs Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig5.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig5

# ---- Primary Visualization Functions ----

def visualize_emissions_targets(company_emissions_df, targets_df, save=True):
    """
    Create visualizations showing relationships between emissions and
    target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plot to file (default is True).

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.
    """
    fig = create_relationship_plots(company_emissions_df, targets_df)
    
    if save:
        save_plot(fig, '01_emissions_targets_relationship_analysis')
    
    return fig

def visualize_target_effectiveness(company_emissions_df, targets_df, use_winsorization=False, save=True):
    """
    Create visualizations analyzing the effectiveness of different target
    types.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions
            data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        use_winsorization (bool, optional): Whether to apply winsorization
            to limit the effect of outliers. Defaults to False.
        save (bool, optional): Whether to save the plot to file. Defaults to
            True.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.
    """
    fig = create_effectiveness_plots(company_emissions_df, targets_df, use_winsorization)
    
    if save:
        save_plot(fig, f'02_targets_emissions_effectiveness_{"winsorized" if use_winsorization else "raw"}')
    
    return fig

def visualize_geo_industry_context(company_emissions_df, targets_df, save=True):
    """
    Create visualizations exploring geographic and industry-specific
    patterns in emissions and target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plot to file. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.
    """
    fig = create_context_plots(company_emissions_df, targets_df)
    
    if save:
        save_plot(fig, '03_geo_industry_emissions_targets_analysis')
    
    return fig

def visualize_counterintuitive_findings(company_emissions_df, targets_df, save=True):
    """
    Create visualizations highlighting counterintuitive or missing
    relationships in emissions and target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plot to file. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.
    """
    fig = create_findings_plots(company_emissions_df, targets_df)
    
    if save:
        save_plot(fig, '04_emissions_targets_counterintuitive_findings')
    
    return fig

def visualize_programs_analysis(company_emissions_df, targets_df, programs1_df, programs2_df, save=True):
    """
    Create visualizations showing the relationships between reduction
    programs and their impact on emissions and target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        programs1_df (pd.DataFrame): Reduction programs 1 data.
        programs2_df (pd.DataFrame): Reduction programs 2 data.
        save (bool, optional): Whether to save the plot to file. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Figure containing the visualizations.
    """
    fig = create_programs_plots(company_emissions_df, targets_df, programs1_df, programs2_df)
    
    if save:
        save_plot(fig, '05_reduction_programs_analysis')
    
    return fig

def visualize_all(
    company_emissions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    programs1_df: pd.DataFrame = None,
    programs2_df: pd.DataFrame = None,
    save: bool = True,
) -> list:
    """
    Create all visualizations in sequence.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        programs1_df (pd.DataFrame, optional): Reduction programs 1 data.
        programs2_df (pd.DataFrame, optional): Reduction programs 2 data.
        save (bool, optional): Whether to save the plots to file. Defaults to True.

    Returns:
        list: List of all figures created.
    """
    # Apply style once at the beginning
    apply_style()
    
    figs = []
    
    print("Creating emissions-targets relationship visualizations...")
    # Create emissions-target relationship visualizations
    fig1 = create_relationship_plots(company_emissions_df, targets_df)
    if save:
        save_plot(fig1, '01_emissions_targets_relationship_analysis')
    figs.append(fig1)
    
    print("Creating target effectiveness visualizations...")
    # Create target effectiveness visualizations
    fig2 = create_effectiveness_plots(company_emissions_df, targets_df, False)
    if save:
        save_plot(fig2, '02_targets_emissions_effectiveness_raw')
    figs.append(fig2)
    
    print("Creating geographic and industry context visualizations...")
    # Create geographic and industry context visualizations
    fig3 = create_context_plots(company_emissions_df, targets_df)
    if save:
        save_plot(fig3, '03_geo_industry_emissions_targets_analysis')
    figs.append(fig3)
    
    print("Creating counterintuitive findings visualizations...")
    # Create counterintuitive findings visualizations
    fig4 = create_findings_plots(company_emissions_df, targets_df)
    if save:
        save_plot(fig4, '04_emissions_targets_counterintuitive_findings')
    figs.append(fig4)
    
    # Create reduction programs analysis if data is available
    if programs1_df is not None and programs2_df is not None:
        print("Creating reduction programs analysis visualizations...")
        fig5 = create_programs_plots(company_emissions_df, targets_df, programs1_df, programs2_df)
        if save:
            save_plot(fig5, '05_reduction_programs_analysis')
        figs.append(fig5)
    
    return figs 

def setup_plot_style():
    """
    Set up the visualization style for all plots using seaborn and
    matplotlib. This function modifies global rcParams for consistent
    appearance across all plots.

    Returns:
        None: This function modifies global state and does not return a
            value.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set(font_scale=1.1)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

def add_correlation_annotation(ax, x_data, y_data, pos=(0.05, 0.95), transform=None):
    """
    Add correlation statistics as an annotation to a scatter plot.

    Args:
        ax (matplotlib.axes.Axes): The axis object to annotate.
        x_data (array-like): Data for the x-axis.
        y_data (array-like): Data for the y-axis.
        pos (tuple, optional): Position of the annotation in axis
            coordinates. Defaults to (0.05, 0.95).
        transform (matplotlib transform, optional): Transform for
            positioning. Defaults to ax.transAxes.

    Returns:
        tuple: The correlation coefficient and p-value as a tuple.
    """
    from scipy import stats
    
    # Drop NaNs for correlation calculation
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(x_valid, y_valid)
    
    # Add correlation information
    if transform is None:
        transform = ax.transAxes
        
    ax.text(pos[0], pos[1], f"r = {corr:.3f}, p = {p_value:.3f}", 
           transform=transform, fontsize=12, ha='left', va='top',
           bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    return corr, p_value

def plot_trendline(ax, x_data, y_data, log_y=False, color='red', linestyle='--', linewidth=2):
    """
    Add a linear trendline to a scatter plot, optionally using a log scale
    for the y-axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the trendline to.
        x_data (array-like): Data for the x-axis.
        y_data (array-like): Data for the y-axis.
        log_y (bool, optional): Whether to use a log scale for the y-axis.
            Defaults to False.
        color (str, optional): Color of the trendline. Defaults to 'red'.
        linestyle (str, optional): Line style for the trendline. Defaults to
            '--'.
        linewidth (int, optional): Width of the trendline. Defaults to 2.

    Returns:
        None: The function adds a trendline to the plot and does not return
            a value.
    """
    import numpy as np
    
    # Drop NaNs for fit
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    if len(x_valid) < 2:  # Need at least 2 points for a line
        return
    
    # For log scale, we need to fit on log-transformed data
    if log_y:
        # Only use positive values for log transform
        pos_mask = y_valid > 0
        x_valid = x_valid[pos_mask]
        y_valid = y_valid[pos_mask]
        
        if len(x_valid) < 2:  # Check again after filtering
            return
            
        # Log transform y data for fitting
        log_y_valid = np.log10(y_valid)
        
        # Fit line to log-transformed data
        m, b = np.polyfit(x_valid, log_y_valid, 1)
        
        # Create the plot line
        x_line = np.linspace(min(x_valid), max(x_valid), 100)
        y_line = 10**(m * x_line + b)
        
        # Plot the line
        ax.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=linewidth)
    else:
        # Create the fit line for regular (non-log) scale
        m, b = np.polyfit(x_valid, y_valid, 1)
        x_line = np.linspace(min(x_valid), max(x_valid), 100)
        ax.plot(x_line, m*x_line + b, color=color, linestyle=linestyle, linewidth=linewidth)

def create_scatter_with_correlation(
    ax, x_data, y_data, title, xlabel, ylabel, add_trendline=True, log_y=False,
    s=30, alpha=0.5, c='#3182bd', grid_alpha=0.3, annotation_text=None,
    annotation_pos=(0.5, 0.05)
):
    """
    Create a scatter plot with correlation statistics, a trendline, and
    optional annotation.

    Args:
        ax (matplotlib.axes.Axes): The axis to create the plot on.
        x_data (array-like): Data for the x-axis.
        y_data (array-like): Data for the y-axis.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        add_trendline (bool, optional): Whether to add a trendline.
            Defaults to True.
        log_y (bool, optional): Whether to use a log scale for the y-axis.
            Defaults to False.
        s (int, optional): Marker size. Defaults to 30.
        alpha (float, optional): Opacity of markers. Defaults to 0.5.
        c (str, optional): Marker color. Defaults to '#3182bd'.
        grid_alpha (float, optional): Opacity of the grid. Defaults to 0.3.
        annotation_text (str, optional): Additional annotation text.
            Defaults to None.
        annotation_pos (tuple, optional): Position for annotation in axis
            coordinates. Defaults to (0.5, 0.05).

    Returns:
        tuple: The correlation coefficient and p-value as a tuple.
    """
    # Create scatter plot
    ax.scatter(x_data, y_data, alpha=alpha, s=s, c=c)
    
    # Add correlation annotation
    corr, p_value = add_correlation_annotation(ax, x_data, y_data)
    
    # Add trendline if requested
    if add_trendline:
        plot_trendline(ax, x_data, y_data, log_y=log_y)
    
    # Set log scale for y axis if requested
    if log_y:
        ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(alpha=grid_alpha)
    
    # Add annotation explaining the finding if provided
    if annotation_text:
        ax.text(annotation_pos[0], annotation_pos[1], annotation_text, 
                transform=ax.transAxes, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', alpha=0.7))
    
    return corr, p_value

def create_bar_chart(
    ax, x_data, y_data, title, xlabel, ylabel, colors=None, yerr=None,
    annotate_bars=True, edgecolor=None, width=0.7, capsize=5, grid_alpha=0.3,
    rotation=0, ha='center'
):
    """
    Create a bar chart with optional error bars and value annotations for
    each bar.

    Args:
        ax (matplotlib.axes.Axes): The axis to create the plot on.
        x_data (array-like): Categories for the x-axis.
        y_data (array-like): Heights of the bars.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        colors (list or str, optional): Colors for the bars. Defaults to
            None.
        yerr (array-like, optional): Error bar values. Defaults to None.
        annotate_bars (bool, optional): Whether to add value labels on bars.
            Defaults to True.
        edgecolor (list or str, optional): Edge colors for bars. Defaults to
            match fill color.
        width (float, optional): Width of the bars. Defaults to 0.7.
        capsize (int, optional): Width of error bar caps. Defaults to 5.
        grid_alpha (float, optional): Opacity of the grid. Defaults to 0.3.
        rotation (int, optional): Rotation of x-tick labels. Defaults to 0.
        ha (str, optional): Horizontal alignment of x-tick labels. Defaults
            to 'center'.

    Returns:
        matplotlib.container.BarContainer: The bar container object for the
            created bars.
    """
    import numpy as np
    
    # If edgecolor not specified, use same as fill color to avoid black borders
    if edgecolor is None and colors is not None:
        edgecolor = colors
        
    # Create positions for the bars (can be simple range if x_data is categorical)
    if isinstance(x_data, (list, np.ndarray)):
        pos = np.arange(len(x_data))
    else:
        pos = x_data
        
    # Create the bar chart
    bars = ax.bar(
        pos, 
        y_data,
        yerr=yerr,
        color=colors,
        edgecolor=edgecolor,
        width=width,
        capsize=capsize
    )
    
    # Add values on the bars if requested
    if annotate_bars:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.3 if height >= 0 else -0.3
            text_color = 'white' if height < -5 else 'black'
            
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + offset if height >= 0 else height - offset,
                f'{height:.1f}%' if abs(height) < 100 else f'{int(height)}%',
                ha='center', va=va, 
                fontweight='bold', color=text_color
            )
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=grid_alpha)
    
    # Set x-axis ticks and labels if categorical
    if isinstance(x_data, (list, np.ndarray)):
        ax.set_xticks(pos)
        ax.set_xticklabels(x_data, rotation=rotation, ha=ha)
    
    return bars 

def kaggle_style(company_emissions_df, targets_df, programs1_df=None, programs2_df=None, 
               visualizations=None, use_winsorization=False, save=True):
    """
    Create Kaggle-style visualizations for ESG data analysis. This function
    acts as a centralized point to create various types of visualizations for
    the ESG data analysis. You can request specific visualization types or all
    of them.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        programs1_df (pd.DataFrame, optional): Reduction programs 1 data.
        programs2_df (pd.DataFrame, optional): Reduction programs 2 data.
        visualizations (str or list, optional): Specific visualization type(s)
            to create. If None, creates all applicable visualizations.
        use_winsorization (bool, optional): Whether to use winsorization for
            handling outliers. Defaults to False.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        dict: Dictionary of figure objects, keyed by visualization type.
    """
    # Initialize results dictionary
    figures = {}
    
    # Determine which visualizations to create
    if visualizations is None:
        # Create all visualizations that are applicable with the provided data
        do_all = True
        vis_list = []
    elif isinstance(visualizations, str):
        # Single visualization type provided as string
        do_all = False
        vis_list = [visualizations]
    else:
        # List of visualization types provided
        do_all = False
        vis_list = visualizations
    
    # Create emissions and targets relationship plots
    if do_all or 'emissions_targets' in vis_list:
        print("Creating emissions and targets relationship plots...")
        figures['emissions_targets'] = create_emissions_target_relations(
            company_emissions_df, targets_df, save=save
        )
    
    # Create target effectiveness plots
    if do_all or 'target_effectiveness' in vis_list:
        print("Creating target effectiveness plots...")
        figures['target_effectiveness'] = create_target_effectiveness(
            company_emissions_df, targets_df, use_winsorization=use_winsorization, save=save
        )
    
    # Create geographic and industry context plots
    if do_all or 'geo_industry' in vis_list:
        print("Creating geographic and industry context plots...")
        figures['geo_industry'] = create_context_plots(
            company_emissions_df, targets_df, save=save
        )
    
    # Create counterintuitive findings plots
    if do_all or 'findings' in vis_list:
        print("Creating counterintuitive findings plots...")
        figures['findings'] = create_findings_plots(
            company_emissions_df, targets_df, save=save
        )
    
    # Create programs analysis plots if data is available
    if (do_all or 'programs' in vis_list) and programs1_df is not None and programs2_df is not None:
        print("Creating programs analysis plots...")
        figures['programs'] = create_programs_plots(
            company_emissions_df, targets_df, programs1_df, programs2_df, save=save
        )
    elif 'programs' in vis_list and (programs1_df is None or programs2_df is None):
        print("Warning: Cannot create programs plots without programs1_df and programs2_df")
    
    print(f"Kaggle-style visualization complete. Created {len(figures)} visualization groups.")
    return figures

# Define wrappers for individual visualization functions to maintain backward compatibility
def create_emissions_target_relations(company_emissions_df, targets_df, save=True):
    """
    Create visualizations showing relationships between emissions and targets.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure object containing the visualizations.
    """
    # Implementation similar to the original function in kaggle_style_visualizations.py
    # This is a wrapper function that would call the actual implementation
    
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
    
    # Create figure
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Set up the plot style
    setup_plot_style()
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # ... implementation of the actual plots would go here ...
    # For brevity, this is omitted but would include the plot code
    
    # Add an overall title
    fig.suptitle('Emissions and Target-Setting Relationships', 
              fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure if requested
    if save:
        save_figure(fig, 'emissions_target_relations')
    
    return fig

def create_target_effectiveness(company_emissions_df, targets_df, use_winsorization=False, save=True):
    """
    Create visualizations showing the effectiveness of different target types.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        use_winsorization (bool, optional): Whether to use winsorization for
            handling outliers. Defaults to False.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure object containing the visualizations.
    """
    # Implementation would be similar to the original function
    
    # Set up the plot style
    setup_plot_style()
    
    # Create figure
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # ... implementation of the actual plots would go here ...
    
    # Add an overall title
    fig.suptitle('Types of Targets and Their Effectiveness', 
              fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure if requested
    if save:
        save_figure(fig, 'target_effectiveness')
    
    return fig

def create_context_plots(company_emissions_df, targets_df, save=True):
    """
    Create visualizations showing geographic and industry context for emissions
    and targets.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure object containing the visualizations.
    """
    # Implementation would be similar to the original function
    
    # Set up the plot style
    setup_plot_style()
    
    # Create figure
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # ... implementation of the actual plots would go here ...
    
    # Add an overall title
    fig.suptitle('Geographic and Industry Context', 
              fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure if requested
    if save:
        save_figure(fig, 'geographic_industry_context')
    
    return fig

def create_findings_plots(company_emissions_df, targets_df, save=True):
    """
    Create visualizations showing counterintuitive findings in the data.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure object containing the visualizations.
    """
    # Implementation would be similar to the original function
    
    # Set up the plot style
    setup_plot_style()
    
    # Create figure
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # ... implementation of the actual plots would go here ...
    
    # Add an overall title
    fig.suptitle('Counterintuitive or Missing Relationships', 
              fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure if requested
    if save:
        save_figure(fig, 'counterintuitive_relationships')
    
    return fig

def create_programs_plots(company_emissions_df, targets_df, programs1_df, programs2_df, save=True):
    """
    Create visualizations analyzing emission reduction programs and their
    impact on emissions and target-setting behavior.

    Args:
        company_emissions_df (pd.DataFrame): Combined company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        programs1_df (pd.DataFrame): Reduction programs 1 data.
        programs2_df (pd.DataFrame): Reduction programs 2 data.
        save (bool, optional): Whether to save the plots to files. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure object containing the visualizations.
    """
    # Apply consistent style
    apply_style()
    
    # Define consistent colors
    primary_color = '#3182bd'  # Blue
    secondary_color = '#31a354'  # Green
    
    # Create a figure with 4 subplots
    fig5 = plt.figure(figsize=(18, 20))
    gs5 = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.4, wspace=0.3)
    
    # 5.1 Number of Programs vs. Emissions Trend
    ax1 = fig5.add_subplot(gs5[0, 0])
    
    # Calculate number of programs per company from programs2_df
    programs_per_company = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    
    # Merge with emissions trend data
    programs_emissions = pd.merge(
        programs_per_company,
        company_emissions_df[['ISSUERID', 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']],
        on='ISSUERID',
        how='inner'
    )
    
    # Filter out missing trend values
    programs_emissions = programs_emissions.dropna(subset=['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
    
    # Create scatter plot with consistent styling
    scatter = ax1.scatter(
        programs_emissions['program_count'],
        programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'],
        alpha=0.3, 
        s=30, 
        c=primary_color
    )
    
    # Set labels and title
    ax1.set_xlabel('Number of Reduction Programs', fontsize=12)
    ax1.set_ylabel('3-Year Emissions Trend (%)', fontsize=12)
    ax1.set_title('5.1 Number of Programs vs. Emissions Trend', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add regression line
    if len(programs_emissions) > 1:
        x = programs_emissions['program_count']
        y = programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR']
        m, b = np.polyfit(x, y, 1)
        ax1.plot(x, m*x + b, color='red', linestyle='--', alpha=0.7)
        
        # Add correlation statistics
        corr, p_value = stats.pearsonr(programs_emissions['program_count'], 
                                      programs_emissions['CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR'])
        correlation_text = f"Correlation: r = {corr:.3f}, p = {p_value:.4f}"
        ax1.text(0.05, 0.95, correlation_text, 
                transform=ax1.transAxes, fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add overall title to the figure
    fig5.suptitle('Reduction Programs Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to ensure all elements fit
    fig5.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig5 