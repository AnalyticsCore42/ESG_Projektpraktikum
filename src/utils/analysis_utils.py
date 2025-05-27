"""
Analysis utilities for ESG (Environmental, Social, and Governance) data processing.

This module provides functions for analyzing and processing various ESG-related dataframes,
including company emissions, targets, and program data. It includes utilities for
dataframe analysis, statistical calculations, and data cleaning.

Key Features:
- Comprehensive dataframe analysis functions
- Target and program data processing utilities
- Statistical analysis and outlier handling
- Data quality assessment tools

Dependencies:
- pandas
- numpy
- re (regular expressions)
- collections.Counter
"""

import logging
import re
from collections import Counter
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 2. BASIC COMPANY-EMISSIONS DATAFRAME ANALYSIS
# Comprehensive analysis of company_emissions_df
def analyze_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Perform a comprehensive analysis of a pandas DataFrame.
    
    This function provides a detailed analysis of the input DataFrame, including:
    - Basic shape and column information
    - Categorization of columns by type (ID, numerical, categorical, text)
    - Summary statistics for numerical columns
    - Distribution of values in categorical columns
    - Identification of missing values
    
    Args:
        df (pd.DataFrame): The DataFrame to be analyzed.
        name (str): A descriptive name for the DataFrame (used in output).
        
    Returns:
        dict: A dictionary containing the categorized column types.
        
    Example
    -------
    >>> import pandas as pd
    >>> data = {'id': [1, 2, 3], 'value': [10.5, 20.3, 30.1]}
    >>> df = pd.DataFrame(data)
    >>> column_info = analyze_dataframe(df, 'Sample Data')
    """
    logger.info(f"\n{name}: shape={df.shape}")
    
    # List of all columns with their data types
    logger.info("\nColumns and Types:")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        logger.info(f"{i}. {col} ({dtype})")
    
    # Create dictionary to categorize columns
    column_types = {
        'id_columns': [col for col in df.columns if 'ISSUER' in col],
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': [],
        'text': []
    }
    
    # Identify potential categorical and text columns among object type
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count <= 30 or (unique_count / len(df) < 0.05):
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)
    
    # 1. Analyze Identifier columns
    logger.info("\n\nIDENTIFIER COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['id_columns']:
        # Skip ISSUER_CNTRY_DOMICILE
        if 'DOMICILE' in col:
            continue
            
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        logger.info(f"{col} - Missing Values: {missing_count} ({missing_count/len(df)*100:.2f}%), Unique Values: {unique_count}")
    
    # 2. Analyze Geographical Data
    geo_cols = [col for col in df.columns if 'CNTRY' in col or 'COUNTRY' in col]
    if geo_cols:
        logger.info("\nGEOGRAPHICAL DATA ANALYSIS")
        logger.info("-" * 80)
        for col in geo_cols:
            logger.info(f"Column: {col}")
            value_counts = df[col].value_counts()
            top_n = min(10, len(value_counts))
            logger.info(f"\nTop {top_n} countries:")
            for country, count in value_counts.head(top_n).items():
                logger.info(f"  - {country}: {count} ({count/len(df)*100:.2f}%)")
            logger.info(f"Total unique countries: {df[col].nunique()}")
            logger.info("")
    
    # 3. Analyze Industry Classification (NACE)
    nace_cols = [col for col in df.columns if 'NACE' in col]
    if nace_cols:
        logger.info("\nINDUSTRY CLASSIFICATION ANALYSIS")
        logger.info("-" * 80)
        for col in nace_cols:
            if df[col].dtype == 'object':
                logger.info(f"Column: {col}")
                value_counts = df[col].value_counts()
                top_n = min(10, len(value_counts))
                logger.info(f"\nTop {top_n} industry classifications:")
                for industry, count in value_counts.head(top_n).items():
                    logger.info(f"  - {industry}: {count} ({count/len(df)*100:.2f}%)")
                logger.info(f"Total unique classifications: {df[col].nunique()}")
                logger.info("")
            else:
                logger.info(f"Column: {col} (Numeric NACE code)")
                logger.info(f"Unique codes: {df[col].nunique()}")
                logger.info("")
    
    # 4. Analyze Numerical Columns
    logger.info("\nNUMERICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['numerical']:
        if col in column_types['id_columns'] or col in nace_cols:
            continue  # Skip already analyzed columns
            
        logger.info(f"Column: {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
        
        # Handle empty columns
        if df[col].count() == 0:
            logger.info("No data in this column")
            continue
            
        # Basic statistics
        non_null = df[col].dropna()
        stats = df[col].describe()
        
        # Only print zeros and negatives if they exist
        zero_count = (non_null == 0).sum()
        negative_count = (non_null < 0).sum()
        zeros_info = f"Zero values: {zero_count} ({zero_count/len(non_null)*100:.2f}%), " if zero_count > 0 else ""
        negatives_info = f"Negative values: {negative_count} ({negative_count/len(non_null)*100:.2f}%), " if negative_count > 0 else ""
        
        # Check for potential outliers using IQR method
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((non_null < (q1 - 1.5 * iqr)) | (non_null > (q3 + 1.5 * iqr))).sum()
        
        # Create single line with min/max/avg/median
        logger.info(f"Range: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Median={stats['50%']:.2f}")
        
        # Put all percentiles on one line
        logger.info(f"Percentiles: 10th={np.percentile(non_null, 10):.2f}, 25th={stats['25%']:.2f}, 75th={stats['75%']:.2f}, 90th={np.percentile(non_null, 90):.2f}")
        
        # Combine other stats on one line
        logger.info(f"Distribution: {zeros_info}{negatives_info}Outliers={outlier_count} ({outlier_count/len(non_null)*100:.2f}%), Std Dev={stats['std']:.2f}")
        
        # Only include skewness/kurtosis for non-year columns
        if not ('YEAR' in col or (non_null.min() >= 1900 and non_null.max() <= 2100 and non_null.dtype == 'float64')):
            skewness = non_null.skew()
            kurtosis = non_null.kurtosis()
            skew_desc = 'Symmetric' if abs(skewness) < 0.5 else 'Moderately skewed' if abs(skewness) < 1 else 'Highly skewed'
            kurtosis_desc = 'Heavy tailed' if kurtosis > 0 else 'Light tailed'
            logger.info(f"Shape: Skewness={skewness:.2f} ({skew_desc}), Kurtosis={kurtosis:.2f} ({kurtosis_desc})")
        
        # Year distribution if it looks like a year column (simplified)
        if 'YEAR' in col or (non_null.min() >= 1900 and non_null.max() <= 2100 and non_null.dtype == 'float64'):
            logger.info("Year distribution:", end=" ")
            year_counts = non_null.value_counts().sort_index()
            year_info = []
            for year, count in year_counts.items():
                if count > len(df) * 0.01:  # Only show years with at least 1% of data
                    year_info.append(f"{int(year)}: {count} ({count/len(non_null)*100:.2f}%)")
            logger.info(", ".join(year_info))
        
        logger.info("")
    
    # 5. Analyze Categorical Columns
    logger.info("\nCATEGORICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['categorical']:
        if col in column_types['id_columns'] or col in geo_cols or col in nace_cols:
            continue  # Skip already analyzed columns
            
        value_counts = df[col].value_counts()
        logger.info(f"Column: {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        
        if len(value_counts) <= 7:
            logger.info("Categories by frequency:")
            for value, count in value_counts.items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        else:
            logger.info(f"Top 7 categories (out of {len(value_counts)}):")
            for value, count in value_counts.head(7).items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        
        logger.info("")

    # 6. Analyze Text/High Cardinality Columns
    logger.info("\nTEXT/HIGH CARDINALITY COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['text']:
        if col in column_types['id_columns'] or col in geo_cols or col in nace_cols:
            continue # Skip already analyzed
        logger.info(f"Column: {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        logger.info("(Analysis skipped for text/high cardinality)")
        logger.info("")

    # 7. Correlations
    if len(column_types['numerical']) > 1:
        logger.info("\nSTRONGLY CORRELATED NUMERICAL COLUMNS")
        logger.info("-" * 80)
        # Calculate correlation matrix for numerical columns
        corr_matrix = df[column_types['numerical']].corr()
        
        # Find pairs with strong correlation (>0.6)
        strong_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Only check each pair once
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.6: # Define threshold
                        strong_corrs.append((col1, col2, corr_value))
        
        if strong_corrs:
            # Sort by absolute correlation value
            strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in strong_corrs:
                logger.info(f"  - {col1} and {col2}: {corr:.4f}")
        else:
            logger.info("No strong correlations found")

    return column_types

# 3. BASIC ANALYSIS OF TARGETS DATAFRAME
def analyze_targets_df(df: pd.DataFrame) -> None:
    """
    Analyze a targets DataFrame for structure, content, and key statistics.

    Args:
        df: The targets DataFrame to analyze.
    """
    # Get column mapping for easy reference
    col_mapping = {i+1: col for i, col in enumerate(df.columns)}
    
    unique_companies = df['ISSUERID'].nunique()
    avg_targets = len(df) / unique_companies
    
    logger.info(f"\nTARGETS DATAFRAME: shape={df.shape}, unique companies={unique_companies:,}, avg targets per company={avg_targets:.2f}")
    
    # List all columns
    logger.info("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"{i}. {col}")
    
    # Create dictionary to categorize columns
    column_types = {
        'id_columns': [col for col in df.columns if 'ISSUER' in col],
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': [],
        'text': []
    }
    
    # Identify potential categorical and text columns among object type
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count <= 30 or (unique_count / len(df) < 0.05):
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)
    
    # 1. Analyze Identifier columns
    logger.info("\n\nIDENTIFIER COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['id_columns']:
        # Skip ISSUER_CNTRY_DOMICILE
        if 'DOMICILE' in col:
            continue
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        logger.info(f"{col_num}. {col} - Missing Values: {missing_count} ({missing_count/len(df)*100:.2f}%), Unique Values: {unique_count}")
    
    # 2. Analyze Geographical Data
    geo_cols = [col for col in df.columns if 'CNTRY' in col or 'COUNTRY' in col]
    if geo_cols:
        logger.info("\nGEOGRAPHICAL DATA ANALYSIS")
        logger.info("-" * 80)
        for col in geo_cols:
            col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
            logger.info(f"{col_num}. {col}")
            value_counts = df[col].value_counts()
            top_n = min(10, len(value_counts))
            logger.info(f"\nTop {top_n} countries:")
            for country, count in value_counts.head(top_n).items():
                logger.info(f"  - {country}: {count} ({count/len(df)*100:.2f}%)")
            logger.info(f"Total unique countries: {df[col].nunique()}")
            logger.info("")
    
    # 3. Analyze Numerical Columns
    logger.info("\nNUMERICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['numerical']:
        if col in column_types['id_columns']:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        
        # Check if it's a year column
        is_year_column = 'YEAR' in col or (
            df[col].dropna().min() >= 1900 and 
            df[col].dropna().max() <= 2100 and 
            df[col].dtype == 'float64'
        )
        
        if is_year_column:
            # Simplified analysis for year columns
            non_null = df[col].dropna()
            year_counts = non_null.value_counts().sort_index()
            
            logger.info(f"{col_num}. {col}")
            logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
            logger.info(f"Year Range: {int(non_null.min())} to {int(non_null.max())}, Average Year: {non_null.mean():.1f}")
            
            # Show distribution of main years
            logger.info("Year distribution:", end=" ")
            year_info = []
            for year, count in year_counts.items():
                if count > len(df) * 0.01:  # Only show years with at least 1% of data
                    year_info.append(f"{int(year)}: {count} ({count/len(non_null)*100:.2f}%)")
            logger.info(", ".join(year_info))
            
        else:
            # Standard analysis for other numerical columns
            logger.info(f"{col_num}. {col}")
            logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
            
            # Handle empty columns
            if df[col].count() == 0:
                logger.info("No data in this column")
                continue
                
            # Basic statistics
            non_null = df[col].dropna()
            stats = df[col].describe()
            
            # Only print zeros and negatives if they exist
            zero_count = (non_null == 0).sum()
            negative_count = (non_null < 0).sum()
            zeros_info = f"Zero values: {zero_count} ({zero_count/len(non_null)*100:.2f}%), " if zero_count > 0 else ""
            negatives_info = f"Negative values: {negative_count} ({negative_count/len(non_null)*100:.2f}%), " if negative_count > 0 else ""
            
            # Check for potential outliers using IQR method
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((non_null < (q1 - 1.5 * iqr)) | (non_null > (q3 + 1.5 * iqr))).sum()
            
            # Create single line with min/max/avg/median
            logger.info(f"Range: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Median={stats['50%']:.2f}")
            
            # Put all percentiles on one line
            logger.info(f"Percentiles: 10th={np.percentile(non_null, 10):.2f}, 25th={stats['25%']:.2f}, 75th={stats['75%']:.2f}, 90th={np.percentile(non_null, 90):.2f}")
            
            # Combine other stats on one line
            logger.info(f"Distribution: {zeros_info}{negatives_info}Outliers={outlier_count} ({outlier_count/len(non_null)*100:.2f}%), Std Dev={stats['std']:.2f}")
            
            skewness = non_null.skew()
            kurtosis = non_null.kurtosis()
            skew_desc = 'Symmetric' if abs(skewness) < 0.5 else 'Moderately skewed' if abs(skewness) < 1 else 'Highly skewed'
            kurtosis_desc = 'Heavy tailed' if kurtosis > 0 else 'Light tailed'
            logger.info(f"Shape: Skewness={skewness:.2f} ({skew_desc}), Kurtosis={kurtosis:.2f} ({kurtosis_desc})")
        
        logger.info("")
    
    # 4. Analyze Categorical Columns
    logger.info("\nCATEGORICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['categorical']:
        if col in column_types['id_columns'] or col in geo_cols:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        value_counts = df[col].value_counts()
        
        logger.info(f"{col_num}. {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        
        if len(value_counts) <= 7:
            logger.info("Categories by frequency:")
            for value, count in value_counts.items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        else:
            logger.info(f"Top 7 categories (out of {len(value_counts)}):")
            for value, count in value_counts.head(7).items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        
        logger.info("")
    
    # 5. Analyze Text/High Cardinality Columns
    logger.info("\nTEXT/HIGH CARDINALITY COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['text']:
        if col in column_types['id_columns'] or col in geo_cols:
            continue # Skip already analyzed
        
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        logger.info(f"{col_num}. {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        logger.info("Large number of unique text descriptions, sample top 5 most common:")
        top_n = min(5, len(df[col].value_counts()))
        for value, count in df[col].value_counts().head(top_n).items():
            # Truncate long descriptions
            display_value = (value[:50] + '...') if len(str(value)) > 50 else value
            logger.info(f"  - {display_value}: {count} ({count/len(df)*100:.2f}%)")
        logger.info("")

    # 6. Correlations
    if len(column_types['numerical']) > 1:
        logger.info("\nSTRONGLY CORRELATED NUMERICAL COLUMNS")
        logger.info("-" * 80)
        # Calculate correlation matrix for numerical columns
        corr_matrix = df[column_types['numerical']].corr()
        
        # Find pairs with strong correlation (>0.3)
        strong_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Only check each pair once
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3: # Define threshold
                        strong_corrs.append((col1, col2, corr_value))
        
        if strong_corrs:
            # Sort by absolute correlation value
            strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in strong_corrs:
                col1_num = list(col_mapping.keys())[list(col_mapping.values()).index(col1)]
                col2_num = list(col_mapping.keys())[list(col_mapping.values()).index(col2)]
                logger.info(f"  - {col1_num}. {col1} and {col2_num}. {col2}: {corr:.4f}")
        else:
            logger.info("No strong correlations found")

# 4. BASIC ANALYSIS OF PROGRAM 1 df
def analyze_programs1_df(df: pd.DataFrame) -> None:
    """
    Analyze a Programs 1 DataFrame for structure, content, and key statistics.

    Args:
        df: The Programs 1 DataFrame to analyze.
    """
    # Get column mapping for easy reference
    col_mapping = {i+1: col for i, col in enumerate(df.columns)}
    
    logger.info(f"\nPROGRAMS1 DATAFRAME: shape={df.shape}")
    
    # List all columns
    logger.info("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"{i}. {col}")
    
    # Create dictionary to categorize columns
    column_types = {
        'id_columns': [col for col in df.columns if 'ISSUER' in col],
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': [],
        'text': []
    }
    
    # Identify potential categorical and text columns among object type
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count <= 30 or (unique_count / len(df) < 0.05):
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)
    
    # 1. Analyze Identifier columns
    logger.info("\n\nIDENTIFIER COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['id_columns']:
        # Skip ISSUER_CNTRY_DOMICILE
        if 'DOMICILE' in col:
            continue
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        logger.info(f"{col_num}. {col} - Missing Values: {missing_count} ({missing_count/len(df)*100:.2f}%), Unique Values: {unique_count}")
    
    # 2. Analyze Geographical Data
    geo_cols = [col for col in df.columns if 'CNTRY' in col or 'COUNTRY' in col]
    if geo_cols:
        logger.info("\nGEOGRAPHICAL DATA ANALYSIS")
        logger.info("-" * 80)
        for col in geo_cols:
            col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
            logger.info(f"{col_num}. {col}")
            value_counts = df[col].value_counts()
            top_n = min(10, len(value_counts))
            logger.info(f"\nTop {top_n} countries:")
            for country, count in value_counts.head(top_n).items():
                logger.info(f"  - {country}: {count} ({count/len(df)*100:.2f}%)")
            logger.info(f"Total unique countries: {df[col].nunique()}")
            logger.info("")
    
    # 3. Analyze Numerical Columns
    logger.info("\nNUMERICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    if not column_types['numerical']:
        logger.info("(None in this specific dataframe)")
    else:
        for col in column_types['numerical']:
            if col in column_types['id_columns']:
                continue  # Skip already analyzed columns
            # ... (rest of numerical analysis - assuming none for programs1)
      
    # 4. Analyze Categorical Columns
    logger.info("\nCATEGORICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['categorical']:
        if col in column_types['id_columns'] or col in geo_cols:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        value_counts = df[col].value_counts()
        
        logger.info(f"{col_num}. {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        
        if len(value_counts) <= 7:
            logger.info("Categories by frequency:")
            for value, count in value_counts.items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        else:
            logger.info(f"Top 7 categories (out of {len(value_counts)}):")
            for value, count in value_counts.head(7).items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        
        logger.info("")
    
    # 5. Analyze Text/High Cardinality Columns
    logger.info("\nTEXT/HIGH CARDINALITY COLUMNS ANALYSIS")
    logger.info("-" * 80)
    if not column_types['text']:
        logger.info("(None in this specific dataframe)")
    else:
        for col in column_types['text']:
            if col in column_types['id_columns'] or col in geo_cols:
                continue # Skip already analyzed
            
            col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
            logger.info(f"{col_num}. {col}")
            logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
            logger.info("Large number of unique text descriptions, analysis omitted.")
            logger.info("")

# 5. BASIC ANALYSIS OF PROGRAM 2 df
def analyze_programs2_df(df: pd.DataFrame) -> None:
    """
    Analyze a Programs 2 DataFrame for structure, content, and key statistics.

    Args:
        df: The Programs 2 DataFrame to analyze.
    """
    # Get column mapping for easy reference
    col_mapping = {i+1: col for i, col in enumerate(df.columns)}
    
    unique_companies = df['ISSUERID'].nunique()
    avg_programs = len(df) / unique_companies
    
    logger.info(f"\nPROGRAMS2 DATAFRAME: shape={df.shape}, unique companies={unique_companies:,}, avg programs per company={avg_programs:.2f}")
    
    # List all columns
    logger.info("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"{i}. {col}")
    
    # Create dictionary to categorize columns
    column_types = {
        'id_columns': [col for col in df.columns if 'ISSUER' in col],
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': [],
        'text': []
    }
    
    # Identify potential categorical and text columns among object type
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count <= 30 or (unique_count / len(df) < 0.05):
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)
    
    # 1. Analyze Identifier columns
    logger.info("\n\nIDENTIFIER COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['id_columns']:
        # Skip ISSUER_CNTRY_DOMICILE
        if 'DOMICILE' in col:
            continue
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        logger.info(f"{col_num}. {col} - Missing Values: {missing_count} ({missing_count/len(df)*100:.2f}%), Unique Values: {unique_count}")
    
    # 2. Analyze Geographical Data with separate top 10 lists
    geo_cols = [col for col in df.columns if 'CNTRY' in col or 'COUNTRY' in col]
    if geo_cols:
        logger.info("\nGEOGRAPHICAL DATA ANALYSIS")
        logger.info("-" * 80)
        for col in geo_cols:
            col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
            logger.info(f"{col_num}. {col}")
            
            # Program count by country
            program_counts = df[col].value_counts()
            
            # Unique company count by country
            company_df = df[['ISSUERID', col]].drop_duplicates()
            company_counts = company_df[col].value_counts()
            
            # Calculate programs per company for each country
            programs_per_company = {}
            for country in program_counts.index:
                if country in company_counts:
                    programs_per_company[country] = program_counts[country] / company_counts[country]
            
            # Top 10 by program count
            logger.info("\nTop 10 countries by total program count:")
            logger.info("┌─────────┬────────────────────┬──────────────┬───────────────────┐")
            logger.info("│ Country │ Program Count      │ Percentage   │ Programs/Company  │")
            logger.info("├─────────┼────────────────────┼──────────────┼───────────────────┤")
            
            top_program_countries = program_counts.head(10).index
            for country in top_program_countries:
                prog_count = program_counts[country]
                prog_pct = prog_count / len(df) * 100
                prog_per_comp = programs_per_company.get(country, 0)
                
                logger.info(f"│ {country:<7} │ {prog_count:>18,} │ {prog_pct:>10.2f}% │ {prog_per_comp:>17.2f} │")
            
            logger.info("└─────────┴────────────────────┴──────────────┴───────────────────┘")
            
            # Top 10 by company count
            logger.info("\nTop 10 countries by unique company count:")
            logger.info("┌─────────┬────────────────────┬──────────────┬───────────────────┐")
            logger.info("│ Country │ Company Count      │ Percentage   │ Programs/Company  │")
            logger.info("├─────────┼────────────────────┼──────────────┼───────────────────┤")
            
            top_company_countries = company_counts.head(10).index
            for country in top_company_countries:
                comp_count = company_counts[country]
                comp_pct = comp_count / unique_companies * 100
                prog_per_comp = programs_per_company.get(country, 0)
                
                logger.info(f"│ {country:<7} │ {comp_count:>18,} │ {comp_pct:>10.2f}% │ {prog_per_comp:>17.2f} │")
            
            logger.info("└─────────┴────────────────────┴──────────────┴───────────────────┘")
            
            logger.info(f"\nTotal unique countries: {df[col].nunique()}")
            logger.info("")
    
    # 3. Analyze Numerical Columns
    logger.info("\nNUMERICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['numerical']:
        if col in column_types['id_columns']:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        
        # Check if it's a year column
        is_year_column = 'YEAR' in col or (
            df[col].dropna().min() >= 1900 and 
            df[col].dropna().max() <= 2100 and 
            df[col].dtype == 'float64'
        )
        
        if is_year_column:
            # Simplified analysis for year columns
            non_null = df[col].dropna()
            year_counts = non_null.value_counts().sort_index()
            
            logger.info(f"{col_num}. {col}")
            logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
            logger.info(f"Year Range: {int(non_null.min())} to {int(non_null.max())}, Average Year: {non_null.mean():.1f}")
            
            # Show distribution of main years
            logger.info("Year distribution:", end=" ")
            year_info = []
            for year, count in year_counts.items():
                if count > len(df) * 0.01:  # Only show years with at least 1% of data
                    year_info.append(f"{int(year)}: {count} ({count/len(non_null)*100:.2f}%)")
            logger.info(", ".join(year_info))
            
        else:
            # Standard analysis for other numerical columns
            logger.info(f"{col_num}. {col}")
            logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
            
            # Handle empty columns
            if df[col].count() == 0:
                logger.info("No data in this column")
                continue
                
            # Basic statistics
            non_null = df[col].dropna()
            stats = df[col].describe()
            
            # Only print zeros and negatives if they exist
            zero_count = (non_null == 0).sum()
            negative_count = (non_null < 0).sum()
            zeros_info = f"Zero values: {zero_count} ({zero_count/len(non_null)*100:.2f}%), " if zero_count > 0 else ""
            negatives_info = f"Negative values: {negative_count} ({negative_count/len(non_null)*100:.2f}%), " if negative_count > 0 else ""
            
            # Check for potential outliers using IQR method
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((non_null < (q1 - 1.5 * iqr)) | (non_null > (q3 + 1.5 * iqr))).sum()
            
            # Create single line with min/max/avg/median
            logger.info(f"Range: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Median={stats['50%']:.2f}")
            
            # Put all percentiles on one line
            logger.info(f"Percentiles: 10th={np.percentile(non_null, 10):.2f}, 25th={stats['25%']:.2f}, 75th={stats['75%']:.2f}, 90th={np.percentile(non_null, 90):.2f}")
            
            # Combine other stats on one line
            logger.info(f"Distribution: {zeros_info}{negatives_info}Outliers={outlier_count} ({outlier_count/len(non_null)*100:.2f}%), Std Dev={stats['std']:.2f}")
            
            skewness = non_null.skew()
            kurtosis = non_null.kurtosis()
            skew_desc = 'Symmetric' if abs(skewness) < 0.5 else 'Moderately skewed' if abs(skewness) < 1 else 'Highly skewed'
            kurtosis_desc = 'Heavy tailed' if kurtosis > 0 else 'Light tailed'
            logger.info(f"Shape: Skewness={skewness:.2f} ({skew_desc}), Kurtosis={kurtosis:.2f} ({kurtosis_desc})")
        
        logger.info("")
    
    # 4. Analyze Categorical Columns
    logger.info("\nCATEGORICAL COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['categorical']:
        if col in column_types['id_columns'] or col in geo_cols:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        value_counts = df[col].value_counts()
        
        logger.info(f"{col_num}. {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        
        if len(value_counts) <= 7:
            logger.info("Categories by frequency:")
            for value, count in value_counts.items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        else:
            logger.info(f"Top 7 categories (out of {len(value_counts)}):")
            for value, count in value_counts.head(7).items():
                logger.info(f"  - {value}: {count} ({count/len(df)*100:.2f}%)")
        
        logger.info("")
    
    # 5. Improved Text/High Cardinality Columns Analysis
    logger.info("\nTEXT/HIGH CARDINALITY COLUMNS ANALYSIS")
    logger.info("-" * 80)
    for col in column_types['text']:
        if col in column_types['id_columns'] or col in geo_cols:
            continue  # Skip already analyzed columns
            
        col_num = list(col_mapping.keys())[list(col_mapping.values()).index(col)]
        logger.info(f"{col_num}. {col}")
        logger.info(f"Data Type: {df[col].dtype}, Missing Values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%), Unique Values: {df[col].nunique()}")
        logger.info("Free text or high cardinality column")
        
        # Analyze text length if object type
        if df[col].dtype == 'object':
            lengths = df[col].dropna().str.len()
            logger.info(f"Text Length - Min: {lengths.min()}, Max: {lengths.max()}, Avg: {lengths.mean():.1f}")
            
            # Basic content checks
            has_url = df[col].dropna().astype(str).str.contains('http|www').any()
            has_email = df[col].dropna().astype(str).str.contains('@').any()
            logger.info(f"Contains URLs: {'Yes' if has_url else 'No'}")
            logger.info(f"Contains email addresses: {'Yes' if has_email else 'No'}")
            
            # Improved source analysis for SOURCE columns
            if 'SOURCE' in col:
                # Process all data instead of a sample
                all_text = df[col].dropna().astype(str)
                
                # Define source keywords to look for (improved categories)
                source_keywords = {
                    'Annual Report': ['annual report', 'annual integrated report', 'annual financial report', 'annual accounts'],
                    'Sustainability Report': ['sustainability report', 'csr report', 'esg report', 'corporate responsibility'],
                    'CDP Response': ['cdp', 'carbon disclosure'],
                    'Corporate Website': ['website', '.com', '.org', 'corporate web'],
                    'Integrated Report': ['integrated report', 'integrated annual'],
                    'Financial Filing': ['10-k', '10k', 'financial statement', 'sec', 'sedar', 'regulatory'],
                    'Environmental Policy': ['environmental policy', 'climate policy', 'energy policy'],
                    'Press Release': ['press release', 'news release', 'press statement']
                }
                
                # Count documents containing each source type
                source_counts = {source: 0 for source in source_keywords}
                for text in all_text:
                    text_lower = text.lower()
                    for source, keywords in source_keywords.items():
                        if any(keyword in text_lower for keyword in keywords):
                            source_counts[source] += 1
                
                # Calculate real percentages based on non-null values
                total_non_null = len(all_text)
                
                logger.info("\nSource distribution (percentage of non-null values):")
                # Sort by frequency and display
                sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
                for source, count in sorted_sources:
                    if count > 0:  # Only show sources with occurrences
                        percentage = count / total_non_null * 100
                        logger.info(f"  - {source}: {count:,} ({percentage:.2f}%)")
                
                # Also calculate most frequent first words to identify document types
                first_words = all_text.str.split(n=1).str[0].value_counts()
                
                logger.info("\nMost common initial terms (likely document references):")
                for word, count in first_words.head(5).items():
                    percentage = count / total_non_null * 100
                    logger.info(f"  - {word}: {count:,} ({percentage:.2f}%)")
            
            # For non-SOURCE text columns, attempt to identify common patterns
            elif not 'SOURCE' in col:
                # Get word frequency analysis from full dataset
                all_text = ' '.join(df[col].dropna().astype(str))
                
                # Simple tokenization
                words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
                word_counts = Counter(words)
                
                # Remove common stopwords
                stopwords = ['the', 'and', 'for', 'that', 'this', 'are', 'with', 'from', 'has', 'have', 
                            'our', 'its', 'their', 'they', 'been', 'were', 'which', 'will', 'each', 'also']
                for word in stopwords:
                    if word in word_counts:
                        del word_counts[word]
                
                # Display top words with counts and percentages
                logger.info("\nMost common terms:")
                total_words = sum(word_counts.values())
                for word, count in word_counts.most_common(10):
                    percentage = count / total_words * 100
                    logger.info(f"  - {word}: {count:,} ({percentage:.2f}%)")
        
        logger.info("")
    
    # 6. Correlations
    if len(column_types['numerical']) > 1:
        logger.info("\nSTRONGLY CORRELATED NUMERICAL COLUMNS")
        logger.info("-" * 80)
        # Calculate correlation matrix for numerical columns
        corr_matrix = df[column_types['numerical']].corr()
        
        # Find pairs with strong correlation (>0.3)
        strong_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Only check each pair once
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:  # Threshold for programs2 data
                        strong_corrs.append((col1, col2, corr_value))
        
        if strong_corrs:
            # Sort by absolute correlation value
            strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in strong_corrs:
                col1_num = list(col_mapping.keys())[list(col_mapping.values()).index(col1)]
                col2_num = list(col_mapping.keys())[list(col_mapping.values()).index(col2)]
                logger.info(f"  - {col1_num}. {col1} and {col2_num}. {col2}: {corr:.4f}")
        else:
            logger.info("No strong correlations found")

def winsorize(data: np.ndarray, lower_percentile: float = 5, upper_percentile: float = 95) -> np.ndarray:
    """
    Cap extreme values at percentiles to handle outliers.

    Args:
        data: Data array to be winsorized.
        lower_percentile: Lower percentile value for capping.
        upper_percentile: Upper percentile value for capping.

    Returns:
        np.ndarray: Winsorized data with values clipped to the specified
            percentiles.
    """
    low_val = np.percentile(data, lower_percentile)
    high_val = np.percentile(data, upper_percentile)
    return np.clip(data, low_val, high_val) 