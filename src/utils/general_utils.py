"""
General utility functions for ESG data analysis.
This module combines categorization and mitigation strategy utilities.
"""

import pandas as pd
import numpy as np

# ---- Categorization Functions ----

def categorize_size(market_cap: float) -> str:
    """
    Categorize companies by market capitalization.
    
    Args:
        market_cap (float): Market capitalization in USD
    
    Returns:
        str: Size category name
    """
    if pd.isna(market_cap):
        return "Unknown"
    elif market_cap >= 200e9:
        return "Mega-cap"
    elif market_cap >= 10e9:
        return "Large-cap"
    elif market_cap >= 2e9:
        return "Mid-cap"
    elif market_cap >= 300e6:
        return "Small-cap"
    else:
        return "Micro-cap"

def categorize_region(country: str) -> str:
    """
    Categorize countries into regional groups.
    
    Args:
        country (str): ISO country code
    
    Returns:
        str: Regional category name
    """
    if pd.isna(country):
        return "Unknown"
    
    # Define regions with country codes
    north_america = ['US', 'CA', 'MX']
    
    europe = ['GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'CH', 'SE', 'NO', 'DK', 'FI', 
              'IE', 'BE', 'AT', 'PT', 'GR', 'PL', 'CZ', 'HU', 'RO', 'SK', 'HR', 
              'SI', 'BG', 'EE', 'LV', 'LT', 'CY', 'MT', 'LU', 'IS']
    
    asia_pacific = ['JP', 'CN', 'HK', 'KR', 'TW', 'SG', 'AU', 'NZ', 'IN', 'ID', 
                   'MY', 'TH', 'PH', 'VN', 'MM', 'KH', 'LA', 'BN', 'PG', 'FJ', 
                   'MN', 'NP', 'BD', 'LK', 'PK']
    
    south_america = ['BR', 'AR', 'CL', 'CO', 'PE', 'VE', 'EC', 'UY', 'PY', 'BO', 
                     'GY', 'SR', 'GF']
    
    africa = ['ZA', 'EG', 'NG', 'KE', 'MA', 'GH', 'TZ', 'ET', 'CI', 'UG', 'SN', 
              'CM', 'ZM', 'AO', 'TN', 'NA', 'MW', 'BW', 'MU', 'RW', 'LY', 'DZ', 
              'MZ', 'ZW', 'CD', 'SD', 'SS', 'MG']
    
    middle_east = ['AE', 'SA', 'IL', 'QA', 'KW', 'BH', 'OM', 'JO', 'LB', 'IQ', 
                  'IR', 'TR']
    
    # Determine region based on country code
    if country in north_america:
        return "North America"
    elif country in europe:
        return "Europe"
    elif country in asia_pacific:
        return "Asia-Pacific"
    elif country in south_america:
        return "South America"
    elif country in africa:
        return "Africa"
    elif country in middle_east:
        return "Middle East"
    else:
        return "Other Regions"

def categorize_industry(nace_desc: str) -> str:
    """
    Categorize industries based on NACE descriptions.
    
    Parameters:
    nace_desc (str): NACE class description
    
    Returns:
    str: Industry group name
    """
    if pd.isna(nace_desc):
        return "Unknown"
    nace_lower = nace_desc.lower()
    if any(term in nace_lower for term in ['manufacture', 'manufacturing', 'production']):
        return "Manufacturing"
    elif any(term in nace_lower for term in ['mining', 'extraction', 'crude', 'quarrying']):
        return "Extractive"
    elif any(term in nace_lower for term in ['bank', 'insurance', 'financial', 'fund', 'trust']):
        return "Financial Services"
    elif any(term in nace_lower for term in ['retail', 'wholesale', 'shop', 'store']):
        return "Retail & Wholesale"
    elif any(term in nace_lower for term in ['transport', 'logistics', 'shipping', 'airline']):
        return "Transportation"
    elif any(term in nace_lower for term in ['energy', 'electricity', 'gas', 'power']):
        return "Energy & Utilities"
    elif any(term in nace_lower for term in ['technology', 'software', 'computer', 'data']):
        return "Technology"
    else:
        return "Other Industries"

# ---- Mitigation Strategy Functions ----
# (Originally from mitigation_utils.py)

def map_distribution_category(category: str) -> str:
    """
    Map distribution mitigation categories to simplified versions.

    Args:
        category (str): Original category name.

    Returns:
        str: Simplified category name.
    """
    mapping = {
        'All or most stores and distribution centers': 'Comprehensive',
        'Some stores/distribution centers (anecdotal cases)': 'Partial',
        'General statement': 'General',
        'No': 'None'
    }
    return mapping.get(category, category)

def map_raw_materials_category(category: str) -> str:
    """
    Map raw materials mitigation categories to simplified versions.

    Args:
        category (str): Original category name.

    Returns:
        str: Simplified category name.
    """
    mapping = {
        'All or core products': 'Comprehensive',
        'Some products (anecdotal cases)': 'Partial',
        'General statement': 'General',
        'No': 'None'
    }
    return mapping.get(category, category)

def map_manufacturing_category(category: str) -> str:
    """
    Map manufacturing mitigation categories to simplified versions.

    Args:
        category (str): Original category name.

    Returns:
        str: Simplified category name.
    """
    mapping = {
        'All or core production facilities': 'Comprehensive',
        'Some facilities (anecdotal cases)': 'Partial',
        'General statement': 'General',
        'No': 'None'
    }
    return mapping.get(category, category)

def map_transport_category(category: str) -> str:
    """
    Map transport mitigation categories to simplified versions.

    Args:
        category (str): Original category name.

    Returns:
        str: Simplified category name.
    """
    mapping = {
        'Improvements in fleet, routes, AND load/packaging optimization': 'Comprehensive',
        'Improvements in fleet, routes, OR load/packaging optimization': 'Partial',
        'General statement': 'General',
        'No': 'None'
    }
    return mapping.get(category, category)

def map_capture_category(category: str) -> str:
    """
    Map carbon capture mitigation categories to simplified versions.

    Args:
        category (str): Original category name.

    Returns:
        str: Simplified category name.
    """
    mapping = {
        'Aggressive efforts': 'Comprehensive',
        'Some efforts': 'Substantial',
        'Limited efforts / information': 'Limited',
        'No evidence': 'None'
    }
    return mapping.get(category, category)

def get_mitigation_score_maps() -> dict:
    """
    Get scoring maps for each mitigation strategy.
    
    Returns:
    dict: Dictionary of scoring maps for each mitigation strategy
    """
    return {
        'CBN_GHG_MITIG_DISTRIBUTION': {
            'All or most stores and distribution centers': 6,
            'Some stores/distribution centers (anecdotal cases)': 4,
            'General statement': 2,
            'No': 0,
            np.nan: 0
        },
        'CBN_GHG_MITIG_RAW_MAT': {
            'All or core products': 6,
            'Some products (anecdotal cases)': 4,
            'General statement': 2,
            'No': 0,
            np.nan: 0
        },
        'CBN_GHG_MITIG_MFG': {
            'All or core production facilities': 6,
            'Some facilities (anecdotal cases)': 4,
            'General statement': 2,
            'No': 0,
            np.nan: 0
        },
        'CBN_GHG_MITIG_TRANSPORT': {
            'Improvements in fleet, routes, AND load/packaging optimization': 6,
            'Improvements in fleet, routes, OR load/packaging optimization': 4,
            'General statement': 2,
            'No': 0,
            np.nan: 0
        },
        'CBN_GHG_MITIG_CAPTURE': {
            'Aggressive efforts': 6,
            'Some efforts': 5,
            'Limited efforts / information': 3,
            'No evidence': 0,
            np.nan: 0
        }
    }

def get_friendly_name_mapping() -> dict:
    """
    Get friendly name mapping for display purposes.
    
    Returns:
    dict: Mapping from original category names to display-friendly names
    """
    return {
        'All or most stores and distribution centers': 'All/Most stores/distribution centers',
        'Some stores/distribution centers (anecdotal cases)': 'Some stores/distribution centers',
        'All or core production facilities': 'All/Core production facilities',
        'Some facilities (anecdotal cases)': 'Some facilities',
        'Improvements in fleet, routes, AND load/packaging optimization': 'Comprehensive',
        'Improvements in fleet, routes, OR load/packaging optimization': 'Partial',
    } 