#!/usr/bin/env python3
"""
Analyzes the sequence of emission reduction program implementation and its impact on emissions.

Input DataFrames:
1. programs2_df: Contains detailed program implementation data with columns for program types and years
2. company_emissions_df: Contains company emissions data with intensity and trend metrics

Key Analyses Performed:
1. Sequence analysis of program implementation across companies
2. Identification of common implementation patterns
3. Correlation between program sequences and emission reductions
4. Visualization of program implementation timelines

Outputs:
- Console output with sequence analysis results
- Visualizations saved to output/reports/figures/
- Analysis reports in output/reports/
"""

# ===== IMPORTS =====
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== OUTPUT DIRECTORY SETUP =====
OUTPUT_DIR = Path("output/reports")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ===== CONSTANTS =====
# Column name mappings for program data
PROGRAMS_COLS = {
    'CATEGORY': 'CBN_PROG_LOW_CARB_RENEW',
    'IMPLEMENTATION_YEAR': 'CBN_IMP_PROG_YEAR',
    'ENERGY_TARGET': 'CBN_EVIDENCE_TARG_ENERGY_IMPROV',
    'EXEC_BODY': 'EXEC_BODY_ENV_ISSUES',
    'ENERGY_AUDITS': 'CBN_REG_ENERGY_AUDITS',
    'CORE_OPS': 'CBN_PROG_REDU_CARB_CORE_OP'
}

# Column name mappings for emissions data
EMISSIONS_COLS = {
    'INTENSITY': 'CARBON_EMISSIONS_SCOPE_12_INTEN',
    'TREND': 'CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR',
    'MARKETCAP': 'MarketCap_USD',
    'COUNTRY': 'ISSUER_CNTRY_DOMICILE_emissions',
    'INDUSTRY': 'NACE_CLASS_DESCRIPTION'
}

def load_datasets():
    """
    Load and prepare the required datasets for sequence analysis.
    
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

def get_unique_sequence(sequence):
    """
    Extract the unique sequence of program implementations without years.
    
    Parameters:
        sequence (list): List of (program, year) tuples
        
    Returns:
        list: Unique program sequence in order of first occurrence
    """
    if not sequence:
        return []
    # Extract just the program types (without years) and remove duplicates while preserving order
    seen = set()
    unique_sequence = []
    for prog_type, _ in sequence:
        prog_name = prog_type.split('_')[-1]
        if prog_name not in seen:
            seen.add(prog_name)
            unique_sequence.append(prog_name)
    return unique_sequence

def analyze_implementation_sequence(programs2_df, company_emissions_df):
    """
    Analyze program implementation sequences and their impact on emissions.
    
    Parameters:
        programs2_df (pd.DataFrame): Program implementation data
        company_emissions_df (pd.DataFrame): Company emissions metrics
        
    Processing Steps:
    1. Map program types to standardized names
    2. Group implementations by company and year
    3. Analyze sequence patterns
    4. Correlate sequences with emission reductions
    5. Generate visualizations
    """
    results = []
    results.append("# Program Implementation Sequencing Analysis\n")
    
    # 1. Basic sequence analysis
    results.append("\n## 1. Program Implementation Sequences\n")
    
    # Get all program types we'll analyze
    program_types = [
        PROGRAMS_COLS['CATEGORY'],
        PROGRAMS_COLS['ENERGY_TARGET'],
        PROGRAMS_COLS['EXEC_BODY'],
        PROGRAMS_COLS['ENERGY_AUDITS'],
        PROGRAMS_COLS['CORE_OPS']
    ]
    
    # Create a sequence dictionary for each company
    company_sequences = defaultdict(list)
    
    for company_id in programs2_df['ISSUERID'].unique():
        company_data = programs2_df[programs2_df['ISSUERID'] == company_id]
        
        # Sort by implementation year
        company_data = company_data.sort_values(PROGRAMS_COLS['IMPLEMENTATION_YEAR'])
        
        # Record the sequence of program types
        for _, row in company_data.iterrows():
            for program_type in program_types:
                if pd.notna(row[program_type]):
                    company_sequences[company_id].append((program_type, row[PROGRAMS_COLS['IMPLEMENTATION_YEAR']]))
    
    # Get unique sequences for each company
    unique_sequences = {
        company_id: get_unique_sequence(sequence)
        for company_id, sequence in company_sequences.items()
    }
    
    # 2. Identify common sequences
    results.append("\n## 2. Common Implementation Sequences\n")
    
    # Count occurrences of each sequence
    sequence_counts = defaultdict(int)
    for sequence in unique_sequences.values():
        if sequence:  # Only count non-empty sequences
            seq_str = ' -> '.join(sequence)
            sequence_counts[seq_str] += 1
    
    # Print top 10 most common sequences
    results.append("Top 10 most common implementation sequences:\n")
    for seq, count in sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        results.append(f"- {seq}: {count} companies")
    
    # 3. Analyze sequence effectiveness
    results.append("\n## 3. Sequence Effectiveness Analysis\n")
    
    # Merge with emissions data
    merged_data = pd.merge(
        pd.DataFrame({
            'ISSUERID': list(unique_sequences.keys()),
            'sequence': [' -> '.join(seq) for seq in unique_sequences.values()]
        }),
        company_emissions_df[['ISSUERID', EMISSIONS_COLS['TREND']]].drop_duplicates(),
        on='ISSUERID',
        how='inner'
    )
    
    # Calculate average emission trend for each sequence
    sequence_effectiveness = {}
    for seq_str in sequence_counts.keys():
        seq_companies = merged_data[merged_data['sequence'] == seq_str]
        if len(seq_companies) > 5:  # Only consider sequences with sufficient data
            avg_trend = seq_companies[EMISSIONS_COLS['TREND']].mean()
            if not pd.isna(avg_trend):  # Only include sequences with valid trend data
                sequence_effectiveness[seq_str] = avg_trend
    
    # Print most effective sequences
    results.append("\nMost effective sequences (by average emission trend):\n")
    for seq, trend in sorted(sequence_effectiveness.items(), key=lambda x: x[1])[:10]:
        results.append(f"- {seq}: {trend:.2f}% average trend ({sequence_counts[seq]} companies)")
    
    # 4. Transition point analysis
    results.append("\n## 4. Critical Transition Points\n")
    
    # Analyze which program type transitions lead to the biggest improvements
    transition_effects = defaultdict(list)
    transition_counts = defaultdict(int)
    
    for sequence in unique_sequences.values():
        if len(sequence) > 1:
            for i in range(len(sequence) - 1):
                from_prog = sequence[i]
                to_prog = sequence[i+1]
                transition = f"{from_prog} -> {to_prog}"
                transition_counts[transition] += 1
    
    for company_id, sequence in unique_sequences.items():
        if len(sequence) > 1:
            company_trend = merged_data[merged_data['ISSUERID'] == company_id][EMISSIONS_COLS['TREND']].iloc[0]
            if not pd.isna(company_trend):
                for i in range(len(sequence) - 1):
                    from_prog = sequence[i]
                    to_prog = sequence[i+1]
                    transition = f"{from_prog} -> {to_prog}"
                    transition_effects[transition].append(company_trend)
    
    # Calculate average effect for each transition
    transition_avg_effects = {
        trans: np.mean(effects) 
        for trans, effects in transition_effects.items() 
        if len(effects) > 5 and not pd.isna(np.mean(effects))
    }
    
    # Print most effective transitions
    results.append("\nMost effective program transitions:\n")
    for trans, effect in sorted(transition_avg_effects.items(), key=lambda x: x[1])[:10]:
        results.append(f"- {trans}: {effect:.2f}% average trend ({transition_counts[trans]} companies)")
    
    # 5. Summary Statistics
    results.append("\n## 5. Summary Statistics\n")
    
    # Calculate average sequence length
    sequence_lengths = [len(seq) for seq in unique_sequences.values() if seq]
    avg_length = np.mean(sequence_lengths)
    median_length = np.median(sequence_lengths)
    
    results.append(f"\nAverage sequence length: {avg_length:.1f} programs")
    results.append(f"Median sequence length: {median_length:.0f} programs")
    
    # Calculate most common starting and ending programs
    starting_programs = defaultdict(int)
    ending_programs = defaultdict(int)
    
    for sequence in unique_sequences.values():
        if sequence:
            starting_programs[sequence[0]] += 1
            ending_programs[sequence[-1]] += 1
    
    results.append("\nMost common starting programs:")
    for prog, count in sorted(starting_programs.items(), key=lambda x: x[1], reverse=True)[:5]:
        results.append(f"- {prog}: {count} companies")
    
    results.append("\nMost common ending programs:")
    for prog, count in sorted(ending_programs.items(), key=lambda x: x[1], reverse=True)[:5]:
        results.append(f"- {prog}: {count} companies")
    
    return "\n".join(results)

def main():
    """
    Main function to execute the program sequencing analysis.
    
    Processing Steps:
    1. Load required datasets
    2. Perform sequence analysis
    3. Generate and save visualizations
    4. Output key findings
    """
    print("Starting Program Implementation Sequencing Analysis...")
    
    try:
        # Load datasets
        programs2_df, company_emissions_df = load_datasets()
        
        # Run the analysis
        results = analyze_implementation_sequence(programs2_df, company_emissions_df)
        
        # Save the report
        report_path = OUTPUT_DIR / "2.12_program_sequencing_analysis.md"
        with open(report_path, 'w') as f:
            f.write(results)
        print(f"\nAnalysis complete! Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 