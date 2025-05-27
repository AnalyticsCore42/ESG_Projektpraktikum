# Reduction Programs 1 Data Dictionary

**File:** `data/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv`  
**Description:** This dataset contains information about emission reduction strategies implemented by companies across different operational areas.  
**Shape:** 2,873 rows × 10 columns  
**Unique companies:** 2,873  

## Quick Links
- [Key Identifiers](#key-identifiers)
- [Mitigation Strategies](#mitigation-strategies)

## Key Identifiers
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| ISSUERID | string | Unique identifier for each company | 0% | - |
| ISSUER_NAME | string | Company name | 0% | - |
| ISSUER_TICKER | string | Company ticker symbol | 0.49% | - |
| ISSUER_ISIN | string | International Securities Identification Number | 0.45% | - |
| ISSUER_CNTRY_DOMICILE | string | Country of domicile | - | Top countries: US (23.42%), JP (10.34%), CN (8.32%) |

## Mitigation Strategies
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CBN_GHG_MITIG_DISTRIBUTION | string | Distribution center emission reduction strategies | 71.67% | Some stores/distribution centers (anecdotal cases): 14.17%, No: 7.66%, General statement: 3.79%, All or most stores and distribution centers: 2.71% |
| CBN_GHG_MITIG_RAW_MAT | string | Raw materials emission reduction strategies | 64.98% | No: 18.10%, Some products (anecdotal cases): 8.77%, General statement: 7.21%, All or core products: 0.94% |
| CBN_GHG_MITIG_MFG | string | Manufacturing emission reduction strategies | 64.98% | Some facilities (anecdotal cases): 13.54%, No: 13.40%, General statement: 4.70%, All or core production facilities: 3.38% |
| CBN_GHG_MITIG_TRANSPORT | string | Transportation emission reduction strategies | 64.98% | Improvements in fleet, routes, OR load/packaging optimization: 12.88%, No: 8.95%, Improvements in fleet, routes, AND load/packaging optimization: 7.48%, General statement: 5.71% |
| CBN_GHG_MITIG_CAPTURE | string | Carbon capture strategies | 34.77% | No evidence: 39.47%, Limited efforts / information: 16.29%, Some efforts: 8.28%, Aggressive efforts: 1.18% |

## Notes
- This dataset has high percentage of missing values, especially in the mitigation strategy columns
- Each row represents a single company (no duplicates)
- Strategy categories generally range from "No" (no strategy) to comprehensive implementation
- Visualizations comparing mitigation strategies to emission trends are available in `output/figures/png/programs1_detailed_visualization.png`

## Strategy Quality Scale
For the mitigation strategy columns, the values generally follow this quality scale:
1. "No" or "No evidence" - No implementation
2. "General statement" - Vague or limited information
3. "Some" (stores/products/facilities) - Partial implementation
4. "All" or "Aggressive efforts" - Comprehensive implementation

## Related Datasets
- [Company Emissions](company_emissions_data.md) - Contains emissions data for analyzing strategy effectiveness
- [Reduction Targets](targets_data.md) - Contains information about emissions reduction targets
- [Reduction Programs 2](programs2_data.md) - Contains detailed information about specific reduction programs 