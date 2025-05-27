# Reduction Programs 2 Data Dictionary

**File:** `data/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv`  
**Description:** This dataset contains detailed information about specific emission reduction programs implemented by companies.  
**Shape:** 52,217 rows × 14 columns  
**Unique companies:** 7,388  
**Average programs per company:** 7.07  

## Quick Links
- [Key Identifiers](#key-identifiers)
- [Program Information](#program-information)
- [Program Details](#program-details)
- [Reference Information](#reference-information)

## Key Identifiers
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| ISSUERID | string | Unique identifier for each company | 0% | - |
| ISSUER_NAME | string | Company name | 0% | - |
| ISSUER_TICKER | string | Company ticker symbol | 0.92% | - |
| ISSUER_ISIN | string | International Securities Identification Number | 2.05% | - |
| ISSUER_CNTRY_DOMICILE | string | Country of domicile | - | Top countries: US (21.89%), JP (10.40%), GB (7.30%) |

## Program Information
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CARBON_PROGRAMS_CATEGORY | string | Category of emission reduction program | 0% | Energy Saving Programs: 36.02%, Energy Alternatives: 25.68%, Responsible Level in Company: 20.13%, Waste Reduction: 11.08%, Energy Audits: 5.65%, Water Conservation: 1.44% |
| CARBON_PROGRAMS_TYPE | string | Specific type of program | 0% | Monitoring/Management System: 11.86%, Use of Renewable Energy: 11.53%, Other Energy Conservation Measures: 8.62%, Process Optimization: 7.65%, etc. |
| CARBON_PROGRAMS_IMP_YEAR | float | Year of program implementation | 3.99% | Range: 1990-2024, Common years: 2021 (20.86%), 2020 (16.27%), 2022 (14.71%) |
| CARBON_PROGRAMS_OVERSIGHT | string | Responsible entity for program oversight | 79.87% | Dedicated Board committee: 8.03%, Sustainability committee: 4.59%, CEO: 3.40%, etc. |

## Program Details
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CARBON_PROGRAMS_DESCRIPTION | string | Detailed description of the program | 2.63% | Text with average length of 126.2 characters |

## Reference Information
| Column | Type | Description | Missing | Example Values |
|--------|------|-------------|---------|----------------|
| CARBON_PROGRAMS_SOURCE | string | Source of information about the program | 2.46% | Sustainability Report: 50.72%, Annual Report: 35.80%, Corporate Website: 13.49%, etc. |
| CARBON_PROGRAMS_SOURCE_DATE | float | Date of information source | 2.46% | Format: YYYYMMDD |
| CARBON_PROGRAMS_COMMENT | string | Additional comments about the program | 93.02% | Various notes and clarifications |
| CARBON_PROGRAMS_LINK | string | URL to program information | 34.23% | Web links to source documents |

## Notes
- This dataset has many rows per company, with each row representing a specific program
- Most companies have implemented multiple emission reduction programs
- US companies have the highest average number of programs per company (12.66)
- Programs from different categories can be analyzed for effectiveness using the company_emissions_data
- Programs can be analyzed sequentially using implementation years

## Program Categories
The main program categories and their frequencies:
1. **Energy Saving Programs** (36.02%) - Efficiency improvements and conservation measures
2. **Energy Alternatives** (25.68%) - Renewable energy and alternative fuel adoption
3. **Responsible Level in Company** (20.13%) - Governance structures for sustainability
4. **Waste Reduction** (11.08%) - Waste management and reduction initiatives
5. **Energy Audits** (5.65%) - Systematic energy usage assessments
6. **Water Conservation** (1.44%) - Water usage reduction initiatives

## Related Datasets
- [Company Emissions](company_emissions_data.md) - Contains emissions data for analyzing program effectiveness
- [Reduction Targets](targets_data.md) - Contains information about emissions reduction targets
- [Reduction Programs 1](programs1_data.md) - Contains information about emission reduction strategies 