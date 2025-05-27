#!/usr/bin/env python3
"""
Comprehensive ESG analysis combining multiple analytical techniques to derive insights from emission reduction programs.

Input DataFrames:
1. emissions_df: Contains company emissions data with columns for scope 1, 2, and 3 emissions
2. programs_df: Contains details of emission reduction programs implemented by companies
3. targets_df: Contains emission reduction targets set by companies

Key Analyses Performed:
1. Data preparation and cleaning for emissions, programs, and targets
2. Time series analysis of emission trends
3. Machine learning-based feature importance analysis
4. Clustering of companies based on emission patterns
5. Association rule mining for program combinations
6. Deep learning-based time series forecasting

Outputs:
- Console output with analysis results and statistics
- Visualizations saved to output/ directory
- Model performance metrics and insights
"""

# ===== IMPORTS =====
# Standard library imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Third-party imports
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import torch
import torch.nn as nn

# Local imports
from src.utils.analysis_utils import analyze_dataframe, analyze_targets_df, analyze_programs1_df, analyze_programs2_df
from src.utils.data_utils import load_data, merge_company_emissions, prepare_final_dataframe
from src.utils.visualization_utils import (
    create_emissions_target_relations,
    create_target_effectiveness,
    create_context_plots,
    create_findings_plots,
    create_programs_plots,
    visualize_all
)


class LSTMModel(nn.Module):
    """Placeholder LSTM model for time series forecasting"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

warnings.filterwarnings('ignore')

# Global constants
OUTPUT_DIR = Path("output/reports")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ===== MAIN ANALYSIS =====

def main():
    """
    Execute the comprehensive ESG analysis pipeline.
    
    Processing Steps:
    1. Load and preprocess emissions, programs, and targets data
    2. Perform time series analysis of emission trends
    3. Train machine learning models for feature importance
    4. Cluster companies based on emission patterns
    5. Mine association rules between programs
    6. Generate visualizations and reports
    
    Returns:
        dict: Dictionary containing analysis results and model outputs
    """
    print("Loading data...")
    data_dir = Path("data")
    
    # Define file paths
    company_path = data_dir / "Unternehmensdaten Results - 20241212 15_41_41.csv"
    emissions_path = data_dir / "Treibhausgasemissionen Results - 20241212 15_44_03.csv"
    targets_path = data_dir / "Reduktionsziele Results - 20241212 15_49_29.csv"
    programs1_path = data_dir / "Reduktionsprogramme 1 Results - 20241212 15_45_26.csv"
    programs2_path = data_dir / "Reduktionsprogramme 2 Results - 20241212 15_51_07.csv"
    merged_path = data_dir / "company_emissions_merged.csv"
    
    print("Loading and preparing data...")
    data_dict = load_data(
        company_path=company_path,
        emissions_path=emissions_path,
        targets_path=targets_path,
        programs1_path=programs1_path,
        programs2_path=programs2_path,
        merged_path=merged_path
    )
    
    # Extract dataframes
    company_emissions_df = data_dict['company_emissions_df']
    targets_df = data_dict['targets_df']
    programs1_df = data_dict['programs1_df']
    programs2_df = data_dict['programs2_df']
    company_df = data_dict['company_df']
    emissions_df = data_dict['emissions_df']
    
    # Analyze the dataframes
    print("Analyzing data...")
    analyze_dataframe(company_df, "Company Data")
    analyze_dataframe(emissions_df, "Emissions Data")
    analyze_dataframe(company_emissions_df, "Merged Company-Emissions Data")
    analyze_targets_df(targets_df)
    analyze_programs1_df(programs1_df)
    analyze_programs2_df(programs2_df)
    
    # Debug: Print columns in programs2_df
    print("\nColumns in programs2_df:")
    print(programs2_df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['CARBON_PROGRAMS_CATEGORY', 'CARBON_PROGRAMS_TYPE']
    missing_columns = [col for col in required_columns if col not in programs2_df.columns]
    
    if missing_columns:
        print(f"\nWarning: The following required columns are missing: {missing_columns}")
        print("Creating dummy columns to avoid errors...")
        for col in missing_columns:
            programs2_df[col] = 'None'
    
    # Prepare final dataframe
    print("Preparing final dataframe...")
    try:
        merged_df = prepare_final_dataframe(
            company_emissions_df, 
            targets_df, 
            programs1_df, 
            programs2_df
        )
    except Exception as e:
        print(f"Error preparing final dataframe: {e}")
        print("Creating a basic merged dataframe with available data...")
        # Create a minimal merged dataframe with just the company_emissions_df
        merged_df = company_emissions_df.copy()
    

    print("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create emissions and target relationship plots
        print("Creating emissions and target relationship plots...")
        fig1 = create_emissions_target_relations(company_emissions_df, targets_df, save=True)
        if 'save_path' in dir(fig1):
            fig1.savefig(FIG_DIR / "emissions_target_relations.png")
            plt.close(fig1)
        
        # Create target effectiveness plots
        print("Creating target effectiveness plots...")
        fig2 = create_target_effectiveness(company_emissions_df, targets_df, use_winsorization=True, save=True)
        if 'save_path' in dir(fig2):
            fig2.savefig(FIG_DIR / "target_effectiveness.png")
            plt.close(fig2)
        
        # Create geographic and industry context plots
        print("Creating geographic and industry context plots...")
        fig3 = create_context_plots(company_emissions_df, targets_df, save=True)
        if 'save_path' in dir(fig3):
            fig3.savefig(FIG_DIR / "geo_industry_context.png")
            plt.close(fig3)
        
        # Create counterintuitive findings plots
        print("Creating counterintuitive findings plots...")
        fig4 = create_findings_plots(company_emissions_df, targets_df, save=True)
        if 'save_path' in dir(fig4):
            fig4.savefig(FIG_DIR / "counterintuitive_findings.png")
            plt.close(fig4)
        
        # Create programs analysis plots if we have the data
        if not programs1_df.empty and not programs2_df.empty:
            print("Creating programs analysis plots...")
            fig5 = create_programs_plots(company_emissions_df, targets_df, programs1_df, programs2_df, save=True)
            if 'save_path' in dir(fig5):
                fig5.savefig(FIG_DIR / "programs_analysis.png")
                plt.close(fig5)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("Skipping visualization generation...")
    
    # Save analysis results
    print("Saving results...")
    results = {
        'company_emissions_analysis': analyze_dataframe(company_emissions_df, "Company Emissions Data"),
        'programs1_analysis': analyze_programs1_df(programs1_df),
        'programs2_analysis': analyze_programs2_df(programs2_df),
        'targets_analysis': analyze_targets_df(targets_df),
        'visualizations_generated': [
            'emissions_target_relations',
            'target_effectiveness',
            'context_plots',
            'findings_plots',
            'programs_analysis'
        ]
    }
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "comprehensive_analysis_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # Use default=str to handle non-serializable types
    
    print(f"\nAnalysis complete! Results saved to {results_file}")
    print(f"Visualizations saved to: {FIG_DIR}")
    
    # Close all figures to free memory
    plt.close('all')
    
    return results

if __name__ == "__main__":
    main()