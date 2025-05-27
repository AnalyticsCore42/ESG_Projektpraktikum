#!/usr/bin/env python3
"""
Association Rule Mining Analysis for Emission Reduction Programs

Input DataFrames:
1. programs2_df: Contains program implementation data with columns for program types and years
2. company_emissions_df: Contains company emissions data with intensity and trend metrics

Key Analyses Performed:
1. Association rule mining to identify patterns in program implementation
2. Sequential pattern analysis of program implementation order
3. Network visualization of program transitions
4. Correlation between program sequences and emission reductions

Outputs:
- Console output with association rules and sequence patterns
- Visualizations saved to output/details/association_10/
- Analysis reports in output/reports/
"""

# ===== IMPORTS =====
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# ===== OUTPUT DIRECTORY SETUP =====
# Use new output folder structure for consistency
OUTPUT_DIR = Path("output/details/association_10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Option to control which formats to save
SAVE_PNG = True  # Set to True to save PNG
SAVE_PDF = True  # Set to True to save PDF

# Constants for column mappings
PROGRAMS_COLS = {
    'CATEGORY': 'CBN_PROG_LOW_CARB_RENEW',
    'IMPLEMENTATION_YEAR': 'CBN_IMP_PROG_YEAR',
    'ENERGY_TARGET': 'CBN_EVIDENCE_TARG_ENERGY_IMPROV',
    'EXEC_BODY': 'EXEC_BODY_ENV_ISSUES',
    'ENERGY_AUDITS': 'CBN_REG_ENERGY_AUDITS',
    'CORE_OPS': 'CBN_PROG_REDU_CARB_CORE_OP'
}

EMISSIONS_COLS = {
    'INTENSITY': 'CARBON_EMISSIONS_SCOPE_12_INTEN',
    'TREND': 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',
    'MARKETCAP': 'MarketCap_USD',
    'COUNTRY': 'ISSUER_CNTRY_DOMICILE_emissions',
    'INDUSTRY': 'NACE_CLASS_DESCRIPTION'
}

def load_datasets():
    """
    Load and prepare the required datasets for association rule mining.
    
    Returns:
        tuple: A tuple containing:
            - programs2_df (pd.DataFrame): Program implementation data
            - company_emissions_df (pd.DataFrame): Company emissions metrics
            
    Processing Steps:
    1. Load programs data from CSV
    2. Load company emissions data
    3. Print dataset statistics
    """
    # Load programs data
    programs2_df = pd.read_csv("data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv")
    print(f"PROGRAMS2 DATAFRAME: shape={programs2_df.shape}, unique companies={programs2_df['ISSUERID'].nunique():,}")
    
    # Load company emissions data
    company_emissions_df = pd.read_csv("data/company_emissions_merged.csv")
    print(f"COMPANY_EMISSIONS DATAFRAME: shape={company_emissions_df.shape}, unique companies={company_emissions_df['ISSUERID'].nunique():,}")
    
    return programs2_df, company_emissions_df

def prepare_association_data(programs2_df):
    """
    Prepare data for association rule mining by converting program implementations
    into a binary matrix format.
    
    Parameters:
        programs2_df (pd.DataFrame): Program implementation data
        
    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): Binary matrix of program implementations
            - unique_programs (list): List of unique program names
            
    Processing Steps:
    1. Create a dictionary mapping companies to their implemented programs
    2. Convert to transaction format
    3. Create binary matrix for association rule mining
    """
    # Create a dictionary to store program implementations for each company
    company_programs = defaultdict(set)
    
    for _, row in programs2_df.iterrows():
        company_id = row['ISSUERID']
        for col_name, col_value in PROGRAMS_COLS.items():
            if col_name != 'IMPLEMENTATION_YEAR' and pd.notna(row[col_value]):
                program_name = col_value.split('_')[-1]
                company_programs[company_id].add(program_name)
    
    # Convert to a list of transactions
    transactions = list(company_programs.values())
    
    # Create a binary matrix for association rule mining
    unique_programs = sorted(set().union(*transactions))
    binary_matrix = []
    
    for transaction in transactions:
        binary_row = [1 if program in transaction else 0 for program in unique_programs]
        binary_matrix.append(binary_row)
    
    # Create DataFrame
    df = pd.DataFrame(binary_matrix, columns=unique_programs, dtype=bool)
    
    return df, unique_programs

def analyze_association_rules(df):
    """
    Perform association rule mining analysis with varying support and confidence thresholds.
    
    Parameters:
        df (pd.DataFrame): Binary matrix of program implementations
        
    Returns:
        pd.DataFrame: Combined rules DataFrame with threshold information
        
    Processing Steps:
    1. Define threshold levels (high, medium, low)
    2. Generate frequent itemsets for each threshold
    3. Create association rules for each threshold
    4. Combine results and add threshold information
    """
    thresholds = [
        (0.3, 0.8),  # High confidence, high support
        (0.2, 0.7),  # Medium confidence, medium support
        (0.1, 0.6)   # Low confidence, low support
    ]
    
    all_rules = []
    for min_support, min_confidence in thresholds:
        # Find frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Add threshold info
        rules['threshold_group'] = f"support={min_support}, confidence={min_confidence}"
        all_rules.append(rules)
    
    # Combine all rules
    combined_rules = pd.concat(all_rules)
    combined_rules = combined_rules.sort_values('confidence', ascending=False)
    
    return combined_rules

def analyze_sequence_rules(programs2_df, company_emissions_df):
    """
    Analyze sequential patterns in program implementation and their impact on emissions.
    
    Parameters:
        programs2_df (pd.DataFrame): Program implementation data
        company_emissions_df (pd.DataFrame): Company emissions metrics
        
    Processing Steps:
    1. Group programs by company and year
    2. Identify sequential patterns
    3. Calculate transition statistics
    4. Correlate sequences with emission reductions
    """
    # Create a dictionary to store program sequences for each company
    company_sequences = defaultdict(list)
    
    for company_id in programs2_df['ISSUERID'].unique():
        company_data = programs2_df[programs2_df['ISSUERID'] == company_id]
        company_data = company_data.sort_values(PROGRAMS_COLS['IMPLEMENTATION_YEAR'])
        
        for _, row in company_data.iterrows():
            for col_name, col_value in PROGRAMS_COLS.items():
                if col_name != 'IMPLEMENTATION_YEAR' and pd.notna(row[col_value]):
                    program_name = col_value.split('_')[-1]
                    if not company_sequences[company_id] or company_sequences[company_id][-1] != program_name:
                        company_sequences[company_id].append(program_name)
    
    # Convert sequences to pairs for analysis
    sequence_pairs = []
    pair_emissions = defaultdict(list)
    
    for company_id, sequence in company_sequences.items():
        if len(sequence) > 1:
            # Get company's emission trend
            if company_id in company_emissions_df['ISSUERID'].values:
                emission_trend = company_emissions_df[
                    company_emissions_df['ISSUERID'] == company_id
                ][EMISSIONS_COLS['TREND']].iloc[0]
                
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i+1])
                    sequence_pairs.append(pair)
                    if not pd.isna(emission_trend):
                        pair_emissions[pair].append(emission_trend)
    
    # Count occurrences of each pair
    pair_counts = defaultdict(int)
    for pair in sequence_pairs:
        pair_counts[pair] += 1
    
    # Calculate transition probabilities and average emissions
    total_pairs = sum(pair_counts.values())
    transition_stats = {}
    
    for pair in pair_counts:
        prob = pair_counts[pair] / total_pairs
        avg_emission = np.mean(pair_emissions[pair]) if pair in pair_emissions else np.nan
        transition_stats[pair] = {
            'probability': prob,
            'count': pair_counts[pair],
            'avg_emission_trend': avg_emission
        }
    
    return transition_stats

def visualize_transitions(transition_stats, output_path):
    """
    Create a network visualization showing the relationships between program transitions.
    
    Parameters:
        transition_stats (dict): Transition statistics between programs
        output_path (str): Path to save the visualization
        
    Processing Steps:
    1. Create a network graph from transition statistics
    2. Calculate node positions using spring layout
    3. Draw nodes, edges, and labels
    4. Save visualization in both PNG and PDF formats
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with weights based on probability
    for (source, target), stats in transition_stats.items():
        G.add_edge(source, target, 
                  weight=stats['probability'],
                  count=stats['count'],
                  emission=stats['avg_emission_trend'])
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw edges with varying thickness based on probability
    edges = G.edges()
    weights = [G[u][v]['weight'] * 10 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                          arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge labels (probability percentage)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1%}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Program Implementation Transition Network")
    plt.axis('off')
    if SAVE_PNG:
        plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to execute the association rule mining analysis.
    
    Processing Steps:
    1. Load required datasets
    2. Prepare data for association rule mining
    3. Perform association rule analysis
    4. Analyze sequential patterns
    5. Generate and save visualizations
    6. Output key findings
    """
    print("Starting Association Rule Mining Analysis...")
    
    # Load datasets
    programs2_df, company_emissions_df = load_datasets()
    
    # Prepare data for association rule mining
    binary_df, unique_programs = prepare_association_data(programs2_df)
    
    # Run association rule mining
    association_results = analyze_association_rules(binary_df)
    
    # Analyze sequential patterns
    transition_stats = analyze_sequence_rules(programs2_df, company_emissions_df)
    
    # Visualize transitions
    transition_viz_path = OUTPUT_DIR / "program_transition_network.png"
    visualize_transitions(transition_stats, transition_viz_path)
    
    # Format results for report
    results = []
    results.append("# Association Rule Mining Analysis")
    results.append("\n## Summary")
    results.append("This analysis identifies patterns in how companies implement emission reduction programs.")
    results.append("We looked at both which programs tend to be implemented together and the sequence of implementation.")
    
    # Add association rules results
    results.append("\n## Association Rules")
    results.append("The following rules show which programs tend to be implemented together:")
    
    # Display top rules by confidence
    top_rules = association_results.head(10)
    for i, row in top_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        results.append(f"- When a company implements {', '.join(antecedents)}, they have a {row['confidence']:.1%} chance of also implementing {', '.join(consequents)}")
        results.append(f"  (Support: {row['support']:.2f}, Lift: {row['lift']:.2f})")
    
    # Add sequence analysis results
    results.append("\n## Program Implementation Sequences")
    results.append("The most common sequences of program implementation are:")
    
    # Sort transitions by probability
    sorted_transitions = sorted(transition_stats.items(), key=lambda x: x[1]['probability'], reverse=True)
    for (source, target), stats in sorted_transitions[:10]:
        results.append(f"- {source} → {target}: {stats['probability']:.1%} probability ({stats['count']} companies)")
        if not pd.isna(stats['avg_emission_trend']):
            trend = "decrease" if stats['avg_emission_trend'] < 0 else "increase"
            results.append(f"  Companies with this pattern show an average emissions {trend} of {abs(stats['avg_emission_trend']):.2f}%")
    
    results.append("\nA network visualization of program transitions has been saved to: "
                  f"{transition_viz_path}")
    
    # Save results to file
    report_path = OUTPUT_DIR / "association_rule_analysis.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(results))
    
    print(f"\nAnalysis complete. Results saved to {report_path}")

if __name__ == "__main__":
    main() 