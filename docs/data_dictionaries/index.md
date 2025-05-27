# ESG Data Dictionary Index

This document provides centralized access to data dictionaries for all datasets used in the ESG Analysis Project.

## Core Datasets

| Dataset | Description | File | Rows | Companies | Documentation |
|---------|-------------|------|------|-----------|---------------|
| **Company Emissions** | Greenhouse gas emissions and company financials | `company_emissions_merged.csv` | 19,276 | 19,276 | [Data Dictionary](company_emissions_data.md) |
| **Reduction Targets** | Carbon emissions and energy reduction targets | `Reduktionsziele Results - 20241212 15_49_29.csv` | 58,501 | 7,226 | [Data Dictionary](targets_data.md) |
| **Reduction Programs 1** | Emission reduction strategies by operational area | `Reduktionsprogramme 1 Results - 20241212 15_45_26.csv` | 2,873 | 2,873 | [Data Dictionary](programs1_data.md) |
| **Reduction Programs 2** | Detailed emission reduction program information | `Reduktionsprogramme 2 Results - 20241212 15_51_07.csv` | 52,217 | 7,388 | [Data Dictionary](programs2_data.md) |

## Relationships Between Datasets

All datasets can be joined using the `ISSUERID` field, which serves as the primary key for companies.

```
Company Emissions ───┐
                     │
Reduction Targets ───┼── ISSUERID (join key)
                     │
Reduction Programs 1─┤
                     │
Reduction Programs 2─┘
```

## Key Analysis Questions That Can Be Addressed

1. **Effectiveness of different reduction strategies**
   - Which mitigation strategies (Programs 1) correlate with the greatest emissions reductions?
   - How do specific program implementations (Programs 2) impact emission trends?

2. **Progress towards targets**
   - How effective are companies at meeting their reduction targets?
   - What factors correlate with successful target achievement?

3. **Industry and regional patterns**
   - How do reduction efforts vary across industries and regions?
   - Which industries show the most progress in emissions reduction?

4. **Strategic approaches**
   - What sequence of reduction programs yields the best results?
   - How comprehensive are companies' mitigation strategies?

## Common Analysis Patterns

When conducting new analyses, consider the following common analysis patterns:

1. **Join emissions data with strategies/programs** to evaluate effectiveness
2. **Group by industry or region** to identify patterns 
3. **Create binary indicators** for strategy implementation
4. **Calculate difference between target and actual** for performance evaluation
5. **Use implementation years** to analyze sequential program adoption


## Troubleshooting Common Data Issues

1. **Missing values** - All datasets contain significant missing values, especially in the reduction strategy columns
2. **Multiple rows per company** - Targets and Programs 2 datasets have multiple rows per company
3. **Different company coverage** - Not all companies appear in all datasets
4. **Temporal consistency** - Ensure year columns are aligned when comparing time-series data 