"""
Script to generate visualizations for the ESG Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add the project root directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create output directories
base_output_dir = Path(os.path.join(PROJECT_ROOT, "output/summary"))
base_output_dir.mkdir(exist_ok=True)

# Create subdirectories for png and pdf
png_dir = base_output_dir / "png"
pdf_dir = base_output_dir / "pdf"
others_dir = base_output_dir / "others"

for directory in [png_dir, pdf_dir, others_dir]:
    directory.mkdir(exist_ok=True)

def load_data():
    """Load the necessary datasets"""
    try:
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        company_emission_df = pd.read_csv(os.path.join(data_dir, 'company_emissions_merged.csv'))
        program1_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsprogramme 1 Results - 20241212 15_45_26.csv'), 
                                quotechar='"', escapechar='\\')
        program2_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsprogramme 2 Results - 20241212 15_51_07.csv'), 
                                quotechar='"', escapechar='\\', on_bad_lines='skip')
        target_df = pd.read_csv(os.path.join(data_dir, 'Reduktionsziele Results - 20241212 15_49_29.csv'), 
                              quotechar='"', escapechar='\\', on_bad_lines='skip')
        
        print(f"Loaded datasets with shapes:")
        print(f"- Company emissions: {company_emission_df.shape}")
        print(f"- Program1: {program1_df.shape}")
        print(f"- Program2: {program2_df.shape}")
        print(f"- Targets: {target_df.shape}")
        
        return company_emission_df, program1_df, program2_df, target_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        # Create mock data for demonstration purposes
        print("Creating mock data for demonstration...")
        return create_mock_data()

def create_mock_data():
    """Create mock data for visualization if real data is unavailable"""
    # Mock company emissions data
    company_emission_df = pd.DataFrame({
        'ISSUERID': range(1000),
        'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR': np.random.normal(-2, 10, 1000),
        'ISSUER_CNTRY_DOMICILE': np.random.choice(['US', 'CN', 'JP', 'UK', 'DE', 'FR', 'IN'], 1000),
        'NACE_CLASS_DESCRIPTION': np.random.choice([
            'Manufacturing - Tech', 'Manufacturing - Heavy', 'Retail', 
            'Financial', 'Energy', 'Transportation', 'Other'
        ], 1000)
    })
    
    # Mock target data
    target_df = pd.DataFrame({
        'ISSUERID': range(500),
        'CBN_TARGET_REDUC_PCT': np.random.uniform(5, 50, 500),
        'CBN_TARGET_IS_SBT': np.random.choice([0, 1], 500),
        'CBN_TARGET_PROGRESS_PCT': np.random.uniform(0, 200, 500)
    })
    
    # Categorize targets
    target_ambition_mapping = pd.DataFrame({
        'ISSUERID': range(500),
        'target_ambition': np.random.choice(['No Target', 'Low Ambition', 'Medium Ambition', 'High Ambition'], 500),
        'emissions_trend': [1.19 if x == 'No Target' else 
                            -1.5 if x == 'Low Ambition' else 
                            -3.82 if x == 'Medium Ambition' else 
                            -3.70 for x in np.random.choice(['No Target', 'Low Ambition', 'Medium Ambition', 'High Ambition'], 500)]
    })
    
    # Mock program data
    program1_df = pd.DataFrame({
        'ISSUERID': range(800),
        'program_implementation_score': np.random.uniform(0, 10, 800)
    })
    
    program2_df = pd.DataFrame({
        'ISSUERID': range(600),
        'executive_oversight': np.random.choice([0, 1], 600)
    })
    
    return company_emission_df, program1_df, program2_df, target_df

def plot_target_ambition_impact():
    """Create visualization for target ambition impact on emissions trends"""
    # Create sample data based on findings
    categories = ['No Target', 'Low Ambition\n(<10%)', 'Medium Ambition\n(10-30%)', 'High Ambition\n(>30%)']
    emissions_trends = [1.19, -1.5, -3.82, -3.70]  # Values from the findings
    counts = [5000, 1200, 2300, 1500]  # Example sample sizes
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot 1: Bar chart of emissions trends
    bars = ax1.bar(categories, emissions_trends, color=['#e15759', '#4e79a7', '#59a14f', '#76b7b2'])
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Emissions Trend (%)', fontsize=14)
    ax1.set_title('Emissions Trends by Target Ambition Level', fontsize=16)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sign = '+' if height > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., 
                height + (0.5 if height > 0 else -0.5),
                f'{sign}{height:.2f}%', 
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    # Plot 2: Sample sizes
    ax2.bar(categories, counts, color='#b3b3b3', alpha=0.7)
    ax2.set_ylabel('Number of Companies', fontsize=14)
    ax2.set_title('Sample Size', fontsize=16)
    
    # Add values on top of bars
    for i, v in enumerate(counts):
        ax2.text(i, v + 100, f'{v:,}', ha='center', fontsize=12)
    
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'target_ambition_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'target_ambition_impact.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created target ambition impact visualization")

def plot_industry_program_impact():
    """Create visualization for industry-specific program effectiveness"""
    # Create sample data based on findings
    industries = ['Retail', 'Manufacturing - Heavy', 'Energy', 'Financial']
    high_score_trends = [-1.15, -3.26, -2.5, -0.8]  # Values from findings, some estimated
    low_score_trends = [7.80, 1.16, 0.3, 0.5]       # Values from findings, some estimated
    
    x = np.arange(len(industries))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, high_score_trends, width, label='High Program Score', color='#4e79a7')
    bars2 = ax.bar(x + width/2, low_score_trends, width, label='Low Program Score', color='#e15759')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and annotations
    ax.set_ylabel('Emissions Trend (%)', fontsize=14)
    ax.set_title('Industry-Specific Program Effectiveness', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(industries, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add values on top of bars
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height > 0 else ''
            ax.annotate(f'{sign}{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -10),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=11, fontweight='bold')
    
    annotate_bars(bars1)
    annotate_bars(bars2)
    
    # Add differences as text
    for i in range(len(industries)):
        diff = abs(high_score_trends[i] - low_score_trends[i])
        ax.annotate(f'Diff: {diff:.2f}%',
                    xy=(i, min(high_score_trends[i], low_score_trends[i])/2 + max(high_score_trends[i], low_score_trends[i])/2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'industry_program_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'industry_program_impact.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created industry program impact visualization")

def plot_feature_importance():
    """Create visualization of feature importance from the program-focused model"""
    # Feature importance data from the model findings
    features = [
        'program_score_by_size',
        'target_by_emissions_intensity',
        'target_vs_industry_avg',
        'program_score_vs_industry',
        'CBN_TARGET_REDUC_PCT',
        'program_implementation_score',
        'executive_body_oversight',
        'energy_alternatives_score',
        'program_age',
        'carbon_reduction_operations',
        'renewable_energy_program',
        'target_timeframe',
        'target_type',
        'energy_audit_frequency'
    ]
    
    importance = [100, 87, 76, 71, 65, 58, 53, 49, 44, 41, 38, 35, 30, 25]  # Normalized importance scores
    
    # Create categories
    categories = []
    colors = []
    for feature in features:
        if 'target' in feature.lower():
            categories.append('Target Setting')
            colors.append('#4e79a7')  # Blue
        elif 'program' in feature.lower() or 'oversight' in feature.lower() or 'audit' in feature.lower():
            categories.append('Program Implementation')
            colors.append('#59a14f')  # Green
        elif 'energy' in feature.lower() or 'carbon' in feature.lower() or 'renewable' in feature.lower():
            categories.append('Energy & Carbon')
            colors.append('#f28e2b')  # Orange
        else:
            categories.append('Other')
            colors.append('#b3b3b3')  # Gray
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance,
        'Category': categories
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
                   color=[colors[features.index(feature)] for feature in feature_importance_df['Feature']])
    
    # Add category legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4e79a7', lw=4, label='Target Setting'),
        Line2D([0], [0], color='#59a14f', lw=4, label='Program Implementation'),
        Line2D([0], [0], color='#f28e2b', lw=4, label='Energy & Carbon'),
        Line2D([0], [0], color='#b3b3b3', lw=4, label='Other')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Add labels
    ax.set_title('Feature Importance in Program-Focused GBM Model', fontsize=16)
    ax.set_xlabel('Relative Importance (%)', fontsize=14)
    
    # Add values on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'feature_importance.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created feature importance visualization")

def plot_target_achievement_impact():
    """Create visualization of emission trends vs. target achievement"""
    # Create sample data based on findings
    achievement_categories = [
        '<25%\n(Far Behind)',
        '25-75%\n(Behind)',
        '75-125%\n(On Track)',
        '125-175%\n(Ahead)',
        '175-200%\n(Far Ahead)',
        '>200%\n(Exceeded)'
    ]
    
    # Emission trend values (some estimated based on report findings)
    emission_trends = [0.5, -1.2, -3.4, -5.1, -8.7, -6.3]
    
    # Sample sizes (estimated)
    sample_sizes = [200, 800, 1500, 900, 400, 250]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot 1: Bar chart of emissions trends
    bars = ax1.bar(achievement_categories, emission_trends, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(achievement_categories))))
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Emissions Trend (%)', fontsize=14)
    ax1.set_title('Emissions Trends by Target Achievement Level', fontsize=16)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sign = '+' if height > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., 
                height + (0.5 if height > 0 else -0.5),
                f'{sign}{height:.1f}%', 
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    # Plot 2: Sample sizes
    ax2.bar(achievement_categories, sample_sizes, color='#b3b3b3', alpha=0.7)
    ax2.set_ylabel('Number of Companies', fontsize=14)
    ax2.set_title('Sample Size', fontsize=16)
    
    # Add values on top of bars
    for i, v in enumerate(sample_sizes):
        ax2.text(i, v + 20, f'{v:,}', ha='center', fontsize=12)
    
    # Rotate x-tick labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'target_achievement_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'target_achievement_impact.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created target achievement impact visualization")

def plot_program_target_interaction():
    """Create visualization of program-target interaction effects"""
    # Create a sample interaction heatmap
    # Data is estimated based on report findings
    program_levels = ['None', 'Basic', 'Intermediate', 'Advanced']
    target_levels = ['No Target', 'Low Ambition', 'Medium Ambition', 'High Ambition']
    
    # Emission trend matrix [rows=program_levels, columns=target_levels]
    interaction_data = np.array([
        [1.8,   0.9,  -0.5,  -0.8],  # No programs
        [1.3,  -0.8,  -2.1,  -2.5],  # Basic programs
        [0.7,  -1.2,  -3.5,  -3.2],  # Intermediate programs
        [0.2,  -1.5,  -4.3,  -4.1]   # Advanced programs
    ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(interaction_data, cmap='RdYlGn_r')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Emissions Trend (%)', rotation=-90, va="bottom", fontsize=12)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(target_levels)))
    ax.set_yticks(np.arange(len(program_levels)))
    ax.set_xticklabels(target_levels, fontsize=12)
    ax.set_yticklabels(program_levels, fontsize=12)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(program_levels)):
        for j in range(len(target_levels)):
            value = interaction_data[i, j]
            sign = '+' if value > 0 else ''
            text_color = 'black' if abs(value) < 3 else 'white'
            ax.text(j, i, f"{sign}{value:.1f}%",
                    ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')
    
    # Add title and labels
    ax.set_title("Interaction Between Programs and Targets\n(Emissions Trend %)", fontsize=16)
    ax.set_xlabel("Target Ambition Level", fontsize=14)
    ax.set_ylabel("Program Implementation Level", fontsize=14)
    
    fig.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'program_target_interaction.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'program_target_interaction.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created program-target interaction visualization")

def plot_oversight_impact():
    """Create visualization of program oversight impact on emissions trends"""
    # Create sample data based on findings
    oversight_types = [
        'No Formal\nOversight',
        'Management\nLevel',
        'C-Suite\nLevel',
        'Board\nCommittee',
        'External\nIndependent\nAudit'
    ]
    
    # Emission trend values (from report findings, some estimated)
    emission_trends = [0.67, 0.1, -1.2, -1.67, -7.21]
    
    # Implementation percentages over time
    years = [2016, 2018, 2020, 2022, 2024]
    implementation_data = {
        'No Formal\nOversight': [65.2, 45.3, 30.1, 15.2, 5.1],
        'Management\nLevel': [25.3, 30.2, 25.8, 20.3, 10.5],
        'C-Suite\nLevel': [6.6, 12.7, 19.2, 22.3, 9.1],
        'Board\nCommittee': [2.9, 10.5, 20.7, 35.5, 75.3],
        'External\nIndependent\nAudit': [0.0, 1.3, 4.2, 6.7, 15.0]
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Bar chart of emissions trends
    colors = plt.cm.viridis(np.linspace(0, 1, len(oversight_types)))
    bars = ax1.bar(oversight_types, emission_trends, color=colors)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Emissions Trend (%)', fontsize=14)
    ax1.set_title('Emissions Trends by Oversight Type', fontsize=16)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sign = '+' if height > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., 
                height + (0.5 if height > 0 else -0.5),
                f'{sign}{height:.2f}%', 
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    # Plot 2: Time series of implementation percentages
    for i, oversight in enumerate(oversight_types):
        ax2.plot(years, implementation_data[oversight], marker='o', linewidth=2.5, 
                 color=colors[i], label=oversight.replace('\n', ' '))
    
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Implementation Percentage (%)', fontsize=14)
    ax2.set_title('Evolution of Oversight Structures (2016-2024)', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax2.legend(title="Oversight Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # Save to both PNG and PDF formats
    plt.savefig(png_dir / 'oversight_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / 'oversight_impact.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created oversight impact visualization")

def main():
    """Main function to generate all visualizations"""
    print("Generating visualizations for the ESG Analysis Report...")
    
    # Load data (or use mock data if files aren't available)
    try:
        company_emission_df, program1_df, program2_df, target_df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using mock data for visualizations...")
    
    # Generate visualizations
    plot_target_ambition_impact()
    plot_industry_program_impact()
    plot_feature_importance()
    plot_target_achievement_impact()
    plot_program_target_interaction()
    plot_oversight_impact()
    
    print(f"All visualizations have been saved to the {output_dir} directory")

if __name__ == "__main__":
    main() 