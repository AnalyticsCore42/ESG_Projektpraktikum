"""
Data utilities for ESG (Environmental, Social, and Governance) analysis.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- Data Loading Functions ----

def load_data(
    company_path: str,
    emissions_path: str,
    targets_path: str,
    programs1_path: str,
    programs2_path: str,
    merged_path: str
) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare ESG datasets from CSV files.

    Args:
        company_path (str): Path to company data CSV.
        emissions_path (str): Path to emissions data CSV.
        targets_path (str): Path to targets data CSV.
        programs1_path (str): Path to programs1 data CSV.
        programs2_path (str): Path to programs2 data CSV.
        merged_path (str): Path for merged company-emissions data.

    Returns:
        dict: Dictionary of loaded DataFrames.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If required columns are missing.
    """
    try:
        company_df = pd.read_csv(company_path).dropna(axis=1, how='all')
        emissions_df = pd.read_csv(emissions_path).dropna(axis=1, how='all')
        targets_df = pd.read_csv(targets_path).dropna(axis=1, how='all')
        programs1_df = pd.read_csv(programs1_path).dropna(axis=1, how='all')
        programs2_df = pd.read_csv(programs2_path).dropna(axis=1, how='all')

        if os.path.exists(merged_path):
            logger.info("Loading pre-merged company_emissions_df from CSV...")
            company_emissions_df = pd.read_csv(merged_path)
        else:
            logger.info("Creating and saving merged company_emissions_df...")
            company_emissions_df = merge_company_emissions(company_df, emissions_df, merged_path)

        logger.info(
            f"Loaded company_df: shape={company_df.shape}, unique_ISSUERIDs={company_df['ISSUERID'].nunique():,}"
        )
        logger.info(
            f"Loaded emissions_df: shape={emissions_df.shape}, unique_ISSUERIDs={emissions_df['ISSUERID'].nunique():,}"
        )
        logger.info(
            f"Merged company_emissions_df: shape={company_emissions_df.shape}, unique_ISSUERIDs={company_emissions_df['ISSUERID'].nunique():,}"
        )
        logger.info(
            f"Loaded targets_df: shape={targets_df.shape}, unique_ISSUERIDs={targets_df['ISSUERID'].nunique():,}"
        )
        logger.info(
            f"Loaded programs1_df: shape={programs1_df.shape}, unique_ISSUERIDs={programs1_df['ISSUERID'].nunique():,}"
        )
        logger.info(
            f"Loaded programs2_df: shape={programs2_df.shape}, unique_ISSUERIDs={programs2_df['ISSUERID'].nunique():,}"
        )

        return {
            'company_df': company_df,
            'emissions_df': emissions_df,
            'targets_df': targets_df,
            'programs1_df': programs1_df,
            'programs2_df': programs2_df,
            'company_emissions_df': company_emissions_df
        }

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Please ensure the data files are present at the specified paths.")
        raise


def merge_company_emissions(
    company_df: pd.DataFrame,
    emissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge company and emissions dataframes, resolving column conflicts.

    Args:
        company_df (pd.DataFrame): Company data.
        emissions_df (pd.DataFrame): Emissions data.
        save_path (str, optional): Path to save the merged dataframe.

    Returns:
        pd.DataFrame: Merged company and emissions data.
    """
    company_emissions_df = pd.merge(
        company_df,
        emissions_df,
        on="ISSUERID",
        how="left",
        suffixes=("_company", "_emissions")
    )

    x_columns = [col for col in company_emissions_df.columns if col.endswith('_company')]
    y_columns = [col for col in company_emissions_df.columns if col.endswith('_emissions')]
    x_to_original = {col: col[:-8] for col in x_columns}
    y_to_original = {col: col[:-10] for col in y_columns}

    for original_name in set(x_to_original.values()) & set(y_to_original.values()):
        x_col = f"{original_name}_company"
        y_col = f"{original_name}_emissions"
        if x_col not in company_emissions_df.columns or y_col not in company_emissions_df.columns:
            continue
        x_null_count = company_emissions_df[x_col].isna().sum()
        y_null_count = company_emissions_df[y_col].isna().sum()
        if y_null_count < x_null_count:
            company_emissions_df[original_name] = company_emissions_df[y_col]
        else:
            company_emissions_df[original_name] = company_emissions_df[x_col]
        company_emissions_df = company_emissions_df.drop([x_col, y_col], axis=1)

    company_emissions_df.columns = [col[:-8] if col.endswith('_company') else col for col in company_emissions_df.columns]
    company_emissions_df.columns = [col[:-10] if col.endswith('_emissions') else col for col in company_emissions_df.columns]

    if save_path:
        company_emissions_df.to_csv(save_path, index=False)
        logger.info(f"Merged dataframe saved to {save_path}")
    return company_emissions_df


def create_short_industry_name(industry_name: str, max_len: int = 30) -> str:
    """
    Create a shortened version of an industry name for display.

    Args:
        industry_name (str): Original industry name.
        max_len (int, optional): Maximum length of the name.

    Returns:
        str: Shortened industry name.
    """
    if pd.isna(industry_name):
        return "Unknown"
    if len(industry_name) <= max_len:
        return industry_name
    parts = industry_name.split()
    abbreviations = {
        'Manufacture': 'Mfg.',
        'and': '&',
        'of': '',
        'Other': 'Oth.',
        'Activities': 'Act.',
        'Services': 'Svc.',
        'except': 'exc.',
        'Insurance': 'Ins.',
        'Reinsurance': 'Reins.',
        'Pension': 'Pens.',
        'Funding': 'Fund.',
        'Management': 'Mgmt.',
        'Consultancy': 'Consult.',
        'Related': 'Rel.',
        'Development': 'Dev.',
        'Experimental': 'Exp.',
        'Biotechnology': 'Biotech.',
        'Pharmaceutical': 'Pharma.',
        'Preparations': 'Prep.',
        'Intermediate': 'Interm.',
        'Intermediation': 'Interm.',
        'Monetary': 'Mon.',
        'Activities': 'Act.',
        'Auxiliary': 'Aux.',
        'Financial': 'Fin.',
        'Service': 'Svc.',
        'Information': 'Info.',
    }
    short_parts = [abbreviations.get(part, part) for part in parts]
    short_name = ' '.join(p for p in short_parts if p)
    if len(short_name) <= max_len:
        return short_name
    return short_name[:max_len-3] + '...'


def prepare_final_dataframe(
    company_emissions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    programs1_df: pd.DataFrame,
    programs2_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare a final dataframe with features from all data sources.

    Args:
        company_emissions_df (pd.DataFrame): Company and emissions data.
        targets_df (pd.DataFrame): Carbon reduction targets data.
        programs1_df (pd.DataFrame): Reduction programs 1 data.
        programs2_df (pd.DataFrame): Reduction programs 2 data.

    Returns:
        pd.DataFrame: Final dataframe with engineered features.
    """
    # Start with company emissions data as the base
    final_df = company_emissions_df.copy()

    # --- Aggregate Targets Data ---
    # Count targets per company
    target_counts = targets_df.groupby('ISSUERID').size().reset_index(name='target_count')
    final_df = pd.merge(final_df, target_counts, on='ISSUERID', how='left')
    final_df['target_count'] = final_df['target_count'].fillna(0).astype(int)

    # Average reduction percentage per company
    avg_reduc_pct = targets_df.groupby('ISSUERID')['CBN_TARGET_REDUC_PCT'].mean().reset_index(name='avg_target_reduc_pct')
    final_df = pd.merge(final_df, avg_reduc_pct, on='ISSUERID', how='left')

    # Most common target category per company
    common_target_category = targets_df.groupby('ISSUERID')['CBN_TARGET_CATEGORY'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'None'
    ).reset_index(name='common_target_category')
    final_df = pd.merge(final_df, common_target_category, on='ISSUERID', how='left')

    # --- Aggregate Programs1 Data ---
    # Flag for manufacturing mitigation program
    programs1_df['has_mfg_program'] = programs1_df['CBN_GHG_MITIG_MFG'].apply(
        lambda x: 1 if pd.notna(x) and x not in ['No'] else 0
    )
    mfg_prog_flag = programs1_df.groupby('ISSUERID')['has_mfg_program'].max().reset_index()
    final_df = pd.merge(final_df, mfg_prog_flag, on='ISSUERID', how='left')
    final_df['has_mfg_program'] = final_df['has_mfg_program'].fillna(0).astype(int)

    # --- Aggregate Programs2 Data ---
    # Count programs per company
    program_counts = programs2_df.groupby('ISSUERID').size().reset_index(name='program_count')
    final_df = pd.merge(final_df, program_counts, on='ISSUERID', how='left')
    final_df['program_count'] = final_df['program_count'].fillna(0).astype(int)

    # Most common program category per company
    common_program_category = programs2_df.groupby('ISSUERID')['CARBON_PROGRAMS_CATEGORY'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'None'
    ).reset_index(name='common_program_category')
    final_df = pd.merge(final_df, common_program_category, on='ISSUERID', how='left')

    # Most common program type per company
    common_program_type = programs2_df.groupby('ISSUERID')['CARBON_PROGRAMS_TYPE'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'None'
    ).reset_index(name='common_program_type')
    final_df = pd.merge(final_df, common_program_type, on='ISSUERID', how='left')

    # --- Additional Feature Engineering ---
    # Emission Intensity Rank within Industry
    final_df['emission_intensity_rank_in_industry'] = final_df.groupby('NACE_CLASS_DESCRIPTION')[
        'CARBON_EMISSIONS_SCOPE_12_INTEN'
    ].rank(pct=True)

    # Target Count Rank within Industry
    final_df['target_count_rank_in_industry'] = final_df.groupby('NACE_CLASS_DESCRIPTION')[
        'target_count'
    ].rank(pct=True)

    # Program Count Rank within Industry
    final_df['program_count_rank_in_industry'] = final_df.groupby('NACE_CLASS_DESCRIPTION')[
        'program_count'
    ].rank(pct=True)

    # Ratio of Programs to Targets
    final_df['program_target_ratio'] = (
        final_df['program_count'] / final_df['target_count']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Short Industry Name for display
    final_df['Short_Industry'] = final_df['NACE_CLASS_DESCRIPTION'].apply(create_short_industry_name)

    # Convert aggregated categorical features to 'category' dtype
    for col in ['common_target_category', 'common_program_category', 'common_program_type']:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('category')

    return final_df 