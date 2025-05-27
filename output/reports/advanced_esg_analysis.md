# Advanced ESG Analysis Report

## Summary of Results

### Market Cap Analysis

Shape: (10159, 7)

First 5 rows:

|    | ISSUERID           |   program_count |   MarketCap_USD |   CARBON_EMISSIONS_SCOPE_12 |   CARBON_EMISSIONS_SCOPE_12_INTEN | Size_Category   |   programs_per_billion |
|---:|:-------------------|----------------:|----------------:|----------------------------:|----------------------------------:|:----------------|-----------------------:|
|  0 | IID000000002123682 |               5 |     4.11267e+09 |             71364           |                             29.1  | Medium-Large    |              1.21575   |
|  1 | IID000000002123684 |               8 |     1.29491e+09 |             44141.2         |                             11.84 | Medium-Small    |              6.17805   |
|  2 | IID000000002123685 |               6 |     6.58941e+10 |                 4.41908e+06 |                            301.5  | Large           |              0.0910552 |
|  3 | IID000000002123686 |               8 |     1.82712e+10 |                 1.60644e+06 |                            282.62 | Large           |              0.437848  |
|  4 | IID000000002123687 |               8 |     3.33226e+08 |             41293           |                             52.79 | Small           |             24.0077    |

### Sales Emissions Analysis

Shape: (12481, 8)

First 5 rows:

|    | ISSUERID           |   SALES_USD_RECENT |   CARBON_EMISSIONS_SCOPE_12 |   CARBON_EMISSIONS_SCOPE_12_INTEN | NACE_CLASS_DESCRIPTION                                            | Industry_Group     | Sales_Category   |   sales_per_emission |
|---:|:-------------------|-------------------:|----------------------------:|----------------------------------:|:------------------------------------------------------------------|:-------------------|:-----------------|---------------------:|
|  0 | IID000000002123682 |            3066.7  |             71364           |                             29.1  | Motion picture projection activities                              | Other Industries   | Medium-High      |           0.0429726  |
|  1 | IID000000002123684 |            3717.03 |             44141.2         |                             11.84 | Retail sale of cosmetic and toilet articles in specialised stores | Retail & Wholesale | Medium-High      |           0.0842078  |
|  2 | IID000000002123685 |           14657    |                 4.41908e+06 |                            301.5  | Freight rail transport                                            | Transportation     | High             |           0.00331675 |
|  3 | IID000000002123686 |            5684    |                 1.60644e+06 |                            282.62 | Extraction of crude petroleum                                     | Extractive         | High             |           0.00353826 |
|  4 | IID000000002123687 |             782.28 |             41293           |                             52.79 | Manufacture of machinery for mining, quarrying and construction   | Manufacturing      | Medium-Low       |           0.0189446  |

### Peer Comparison Analysis

Shape: (6863, 10)

First 5 rows:

|    | ISSUERID           | ISSUER_NAME_company            | ISSUER_CNTRY_DOMICILE_emissions   | NACE_CLASS_DESCRIPTION                                                          |   CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO |   CARBON_EMISSIONS_SCOPE_12_INTEN | Region        | Industry_Group   | Performance_Category       | is_leader   |
|---:|:-------------------|:-------------------------------|:----------------------------------|:--------------------------------------------------------------------------------|-----------------------------------------------:|----------------------------------:|:--------------|:-----------------|:---------------------------|:------------|
|  2 | IID000000002123685 | CSX Corporation                | US                                | Freight rail transport                                                          |                                        96.6333 |                            301.5  | North America | Transportation   | Significant Underperformer | False       |
|  3 | IID000000002123686 | Coterra Energy Inc.            | US                                | Extraction of crude petroleum                                                   |                                       377.5    |                            282.62 | North America | Extractive       | Significant Underperformer | False       |
|  4 | IID000000002123687 | OIL STATES INTERNATIONAL, INC. | US                                | Manufacture of machinery for mining, quarrying and construction                 |                                        51.15   |                             52.79 | North America | Manufacturing    | Significant Underperformer | False       |
|  5 | IID000000002123689 | CTS CORPORATION                | US                                | Manufacture of instruments and appliances for measuring, testing and navigation |                                        46.1333 |                            310.65 | North America | Manufacturing    | Significant Underperformer | False       |
|  6 | IID000000002123690 | Magna International Inc.       | CA                                | Manufacture of other parts and accessories for motor vehicles                   |                                        58.7667 |                             46.31 | North America | Manufacturing    | Significant Underperformer | False       |

