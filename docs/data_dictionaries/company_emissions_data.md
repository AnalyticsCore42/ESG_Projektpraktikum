# Company Emissions Data Dictionary

**File:** `data/company_emissions_merged.csv` (merged emissions data)  
**Description:** This dataset contains company information and greenhouse gas emissions data.  
**Shape:** 19,276 rows × 38 columns  
**Unique companies:** 19,276  

## Quick Links
- [Key Identifiers](#key-identifiers)
- [Emissions Data](#emissions-data)
- [Financial Metrics](#financial-metrics)
- [Geographic Information](#geographic-information)
- [Industry Classification](#industry-classification)

## Key Identifiers
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| ISSUERID | string | Unique identifier for each company | 0% | - |
| ISSUER_NAME | string | Company name | 0% | - |
| ISSUER_TICKER | string | Company ticker symbol | 17.51% | - |
| ISSUER_ISIN | string | International Securities Identification Number | 6.74% | - |

## Emissions Data
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CARBON_EMISSIONS_SCOPE_1_KEY | string | Information source for Scope 1 emissions | - | - |
| CARBON_EMISSIONS_SCOPE_1_INTEN | float | Scope 1 emissions intensity | 36.86% | Min=0.00, Max=55407.61, Mean=216.30, Median=6.61 |
| CARBON_EMISSIONS_EVIC_SCOPE_1_INTEN | float | Scope 1 emissions intensity by EVIC | 36.90% | Min=0.00, Max=34164.07, Mean=120.14, Median=2.07 |
| CARBON_EMISSIONS_SCOPE_2 | float | Scope 2 emissions | 35.75% | Min=0.00, Max=121129435.46, Mean=214018.21 |
| CARBON_EMISSIONS_SCOPE_2_KEY | string | Information source for Scope 2 emissions | - | - |
| CARBON_EMISSIONS_SCOPE_2_INTEN | float | Scope 2 emissions intensity | 36.86% | Min=0.00, Max=27869.57, Mean=54.42, Median=13.38 |
| CARBON_EMISSIONS_EVIC_SCOPE_2_INTEN | float | Scope 2 emissions intensity by EVIC | 36.90% | Min=0.00, Max=2875.47, Mean=26.30, Median=4.78 |
| CARBON_EMISSIONS_SCOPE_12 | float | Combined Scope 1+2 emissions | 35.75% | Min=0.00, Max=565449323.00, Mean=1499920.50 |
| CARBON_EMISSIONS_SCOPE_12_KEY | string | Information source for Scope 1+2 emissions | - | - |
| CARBON_EMISSIONS_SCOPE_12_INTEN | float | Scope 1+2 emissions intensity | 35.23% | Min=0.00, Max=83277.17, Mean=267.74, Median=28.98 |
| CARBON_EMISSIONS_EVIC_SCOPE_12_INTEN | float | Scope 1+2 emissions intensity by EVIC | 36.90% | Min=0.00, Max=34863.64, Mean=146.36, Median=10.06 |
| CARBON_EMISSIONS_SCOPE_3 | float | Scope 3 emissions | 75.36% | Min=0.00, Max=1147000000.00, Mean=8903623.85 |
| CARBON_EMISSIONS_INTENSITY_YEAR | float | Reporting year for emissions intensity data | 35.23% | 2021-2023 |
| CARBON_SCOPE_12_INTEN_3Y_AVG | float | 3-year average of Scope 1+2 intensity | - | - |
| CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO | float | Ratio comparing to industry peers | - | - |
| CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR | float | 3-year compound annual growth rate of Scope 1+2 intensity | - | - |

## Financial Metrics
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| MarketCap_USD | float | Market capitalization in USD | 28.09% | Min=0.08, Max=3.75T, Mean=8.83B, Median=1.19B |
| SALES_USD_YEAR | float | Reporting year for sales data | 30.54% | 2021-2024 |
| SALES_USD_RECENT | float | Recent sales figure in USD | 30.07% | Min=-2.43, Max=648125.00, Mean=5934.55, Median=1059.93 |
| EVIC_EUR | float | Enterprise Value including Cash in EUR | 34.68% | Min=0.74, Max=6395000.00, Mean=16067.92, Median=2635.26 |
| EVIC_USD_RECENT | float | Recent Enterprise Value including Cash in USD | 34.73% | Min=0.81, Max=7064236.75, Mean=17673.67, Median=2901.88 |
| EVIC_USD_YEAR | float | Reporting year for EVIC data | 34.73% | 2015-2024 |
| CBN_EVIC_PUB_DATE | float | Publication date for EVIC data | 36.52% | Format: YYYYMMDD |

## Geographic Information
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| ISSUER_CNTRY_DOMICILE | string | Country of domicile | - | Top countries: US (21.19%), CN (11.63%), JP (8.92%) |

## Industry Classification
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| NACE_CLASS_CODE | string | NACE industry classification code | - | - |
| NACE_CLASS_DESCRIPTION | string | NACE industry classification description | - | Top sectors: Other monetary intermediation (6.31%), Electronic components (3.50%) |

## Notes
- Missing values percentages indicate data gaps
- EVIC = Enterprise Value Including Cash, an alternative valuation metric
- Intensity measures generally refer to emissions relative to revenue or another metric
- CAGR = Compound Annual Growth Rate

## Related Datasets
- [Reduction Targets](targets_data.md) - Contains information about emissions reduction targets
- [Reduction Programs 1](programs1_data.md) - Contains information about emission reduction strategies
- [Reduction Programs 2](programs2_data.md) - Contains detailed information about specific reduction programs 