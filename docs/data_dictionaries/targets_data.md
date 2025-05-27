# Reduction Targets Data Dictionary

**File:** `data/Reduktionsziele Results - 20241212 15_49_29.csv`  
**Description:** This dataset contains information about companies' carbon emission reduction targets.  
**Shape:** 58,501 rows × 25 columns  
**Unique companies:** 7,226  

## Quick Links
- [Key Identifiers](#key-identifiers)
- [Target Categories](#target-categories)
- [Target Parameters](#target-parameters)
- [Target Progress](#target-progress)
- [Carbon Offsets](#carbon-offsets)

## Key Identifiers
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| ISSUERID | string | Unique identifier for each company | 0% | - |
| ISSUER_NAME | string | Company name | 0% | - |
| ISSUER_TICKER | string | Company ticker symbol | 0.69% | - |
| ISSUER_ISIN | string | International Securities Identification Number | 1.51% | - |
| ISSUER_CNTRY_DOMICILE | string | Country of domicile | - | Top countries: JP (17.46%), US (16.59%), GB (8.41%) |

## Target Categories
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CBN_TARGET_CATEGORY | string | Category of emission reduction target | 6.83% | Carbon emissions - Absolute (52.63%), Carbon emissions - Intensity (20.71%), Energy consumption - Absolute (12.58%), Energy consumption - Intensity (5.16%), Other (2.09%) |
| CBN_TARGET_AGGR | string | Aggressiveness of target | 3.19% | Very Aggressive Target (Total reduction > 30%): 41.95%, Weak Target (Total reduction <=5%): 13.12%, No reduction percentage reported: 12.21%, Aggressive Target (20% < Total reduction <=30%): 11.54% |
| CBN_TARGET_SCOPE | string | Scope of target within organization | 7.53% | Targets covers all relevant segments: 59.89%, Targets covers selected segments: 22.83%, Scope not determinable: 9.75% |
| TARGET_CARBON_TYPE | string | Type of carbon target | 10.65% | Absolute: 63.45%, Production intensity: 14.42%, Sales intensity: 5.95%, Others?: 5.53% |
| TARGET_CARBON_TYPE_OTHER | string | Additional details for other target types | 93.90% | 1,845 unique values including: tCO2e/unit FTE employee, tCO2e/square meter, etc. |
| TARGET_CARBON_UNITS | string | Units used for target measurement | 10.72% | tCO2e: 50.21%, Other: 23.91%, MWh: 3.93%, tCO2e/unit of revenue: 3.34% |
| TARGET_CARBON_UNITS_OTHER | string | Details for other units | - | - |

## Target Parameters
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CBN_TARGET_IMP_YEAR | float | Year of target implementation | 14.02% | Range: 1990-2030, Common years: 2021 (22.16%), 2022 (17.02%), 2020 (13.58%) |
| CBN_TARGET_REDUC_PCT | float | Target reduction percentage | 17.30% | Min=0.00, Max=100.00, Mean=42.46, Median=30.00 |
| CBN_TARGET_BASE_YEAR | float | Base year for target measurement | 20.61% | Range: 1900-2030, Common years: 2019 (19.37%), 2020 (12.22%), 2018 (9.57%) |
| CBN_TARGET_BASE_YEAR_VAL | float | Base year value | 50.21% | - |

## Target Progress
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CBN_TARGET_STATUS | string | Current status of target | 3.19% | Ongoing Target: 64.77%, Historical Target: 32.04% |
| CBN_TARGET_STATUS_DETAIL | string | Detailed status information | 93.52% | On pace to achieve target: 3.96%, Achieved the set target: 0.88%, etc. |
| TARGET_CARBON_CURRENT_REPORTING_YEAR | float | Current reporting year for target | 15.78% | Range: 2005-2040, Common years: 2021 (28.13%), 2022 (26.50%), 2020 (14.77%) |
| TARGET_CARBON_PROGRESS_VALUE | float | Current progress value | 65.08% | Min=-2.1B, Max=49.5B, Mean=10.5M, Median=4842.00 |
| TARGET_CARBON_PROGRESS_PCT | float | Progress percentage | 66.56% | Min=-1.6M, Max=18.6M, Mean=1900.72, Median=54.81 |

## Carbon Offsets
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| TARGET_CARBON_OFFSET | string | Indicator for carbon offset usage | 98.79% | T: 1.21% |
| TARGET_CARBON_OFFSET_DESC | string | Description of carbon offset approach | 99.06% | 494 unique text descriptions |
| TARGET_CARBON_OFFSET_PCT | float | Percentage of target using offsets | 99.85% | Min=0.00, Max=100.00, Mean=39.14, Median=20.00 |
| TARGET_CARBON_OFFSET_VOLUME | float | Volume of carbon offsets | 99.91% | Min=0.00, Max=16M, Mean=767,653.78, Median=14,510.00 |

## Notes
- Missing values percentages indicate data gaps
- Many companies have multiple targets, resulting in multiple rows per company
- Target aggressiveness (CBN_TARGET_AGGR) is a calculated field based on reduction percentage

## Related Datasets
- [Company Emissions](company_emissions_data.md) - Contains emissions data for analyzing target progress
- [Reduction Programs 1](programs1_data.md) - Contains information about emission reduction strategies
- [Reduction Programs 2](programs2_data.md) - Contains detailed information about specific reduction programs 