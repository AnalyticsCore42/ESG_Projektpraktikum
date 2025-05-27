COMPANY-EMISSIONS DATAFRAME: shape=(19276, 30)

Columns:
1. ISSUERID
2. MarketCap_USD
3. NACE_CLASS_CODE
4. NACE_CLASS_DESCRIPTION
5. SALES_USD_YEAR
6. SALES_USD_RECENT
7. CARBON_EMISSIONS_SCOPE_1_KEY
8. CARBON_EMISSIONS_EVIC_SCOPE_1_INTEN
9. CARBON_EMISSIONS_SCOPE_1_INTEN
10. CARBON_EMISSIONS_SCOPE_12
11. CARBON_EMISSIONS_EVIC_SCOPE_12_INTEN
12. CARBON_EMISSIONS_SCOPE_12_INTEN
13. CARBON_EMISSIONS_INTENSITY_YEAR
14. CARBON_EMISSIONS_SCOPE_12_KEY
15. CARBON_EMISSIONS_SCOPE_2
16. CARBON_EMISSIONS_SCOPE_2_KEY
17. CARBON_EMISSIONS_EVIC_SCOPE_2_INTEN
18. CARBON_EMISSIONS_SCOPE_2_INTEN
19. CARBON_EMISSIONS_SCOPE_3
20. EVIC_EUR
21. CBN_EVIC_PUB_DATE
22. EVIC_USD_RECENT
23. EVIC_USD_YEAR
24. CARBON_SCOPE_12_INTEN_3Y_AVG
25. CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO
26. CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR
27. ISSUER_TICKER
28. ISSUER_ISIN
29. ISSUER_CNTRY_DOMICILE
30. ISSUER_NAME


IDENTIFIER COLUMNS ANALYSIS
--------------------------------------------------------------------------------
ISSUERID - Missing Values: 0 (0.00%), Unique Values: 19276
ISSUER_TICKER - Missing Values: 3376 (17.51%), Unique Values: 14615
ISSUER_ISIN - Missing Values: 1300 (6.74%), Unique Values: 17976
ISSUER_NAME - Missing Values: 0 (0.00%), Unique Values: 19267

GEOGRAPHICAL DATA ANALYSIS
--------------------------------------------------------------------------------
Column: ISSUER_CNTRY_DOMICILE

Top 10 countries:
  - US: 4085 (21.19%)
  - CN: 2241 (11.63%)
  - JP: 1719 (8.92%)
  - IN: 911 (4.73%)
  - KR: 855 (4.44%)
  - GB: 838 (4.35%)
  - TW: 748 (3.88%)
  - CA: 692 (3.59%)
  - DE: 608 (3.15%)
  - AU: 575 (2.98%)
Total unique countries: 199


INDUSTRY CLASSIFICATION ANALYSIS
--------------------------------------------------------------------------------
Column: NACE_CLASS_CODE (Numeric NACE code)
Unique codes: 536

Column: NACE_CLASS_DESCRIPTION

Top 10 industry classifications:
  - Other monetary intermediation: 1216 (6.31%)
  - Manufacture of electronic components: 674 (3.50%)
  - Manufacture of pharmaceutical preparations: 665 (3.45%)
  - Renting and operating of own or leased real estate: 633 (3.28%)
  - Other software publishing: 529 (2.74%)
  - Research and experimental development on biotechnology: 460 (2.39%)
  - Production of electricity: 400 (2.08%)
  - General public administration activities: 339 (1.76%)
  - Mining of other non-ferrous metal ores: 289 (1.50%)
  - Computer programming activities: 256 (1.33%)
Total unique classifications: 536


NUMERICAL COLUMNS ANALYSIS
--------------------------------------------------------------------------------
Column: MarketCap_USD
Data Type: float64, Missing Values: 5415 (28.09%)
Range: Min=0.08, Max=3745247464710.00, Mean=8825687940.82, Median=1189420539.00
Percentiles: 10th=47897989.17, 25th=361066842.07, 75th=4126488068.25, 90th=13633598768.87
Distribution: Outliers=1821 (13.14%), Std Dev=69238197145.84
Shape: Skewness=36.08 (Highly skewed), Kurtosis=1603.68 (Heavy tailed)

Column: SALES_USD_YEAR
Data Type: float64, Missing Values: 5886 (30.54%)
Range: Min=2012.00, Max=2024.00, Mean=2022.92, Median=2023.00
Percentiles: 10th=2022.00, 25th=2023.00, 75th=2023.00, 90th=2023.00
Distribution: Outliers=2397 (17.90%), Std Dev=0.67
Year distribution: 2021: 331 (2.47%), 2022: 946 (7.06%), 2023: 10993 (82.10%), 2024: 1005 (7.51%)

Column: SALES_USD_RECENT
Data Type: float64, Missing Values: 5797 (30.07%)
Range: Min=-2.43, Max=648125.00, Mean=5934.55, Median=1059.93
Percentiles: 10th=97.31, 25th=324.06, 75th=3630.20, 90th=11991.38
Distribution: Zero values: 213 (1.58%), Negative values: 2 (0.01%), Outliers=1738 (12.89%), Std Dev=22237.64
Shape: Skewness=12.68 (Highly skewed), Kurtosis=230.57 (Heavy tailed)

Column: CARBON_EMISSIONS_EVIC_SCOPE_1_INTEN
Data Type: float64, Missing Values: 7113 (36.90%)
Range: Min=0.00, Max=34164.07, Mean=120.14, Median=2.07
Percentiles: 10th=0.03, 25th=0.22, 75th=16.94, 90th=168.36
Distribution: Zero values: 465 (3.82%), Outliers=2155 (17.72%), Std Dev=711.90
Shape: Skewness=21.40 (Highly skewed), Kurtosis=738.38 (Heavy tailed)

Column: CARBON_EMISSIONS_SCOPE_1_INTEN
Data Type: float64, Missing Values: 7106 (36.86%)
Range: Min=0.00, Max=55407.61, Mean=216.30, Median=6.61
Percentiles: 10th=0.23, 25th=0.88, 75th=30.18, 90th=287.22
Distribution: Zero values: 346 (2.84%), Outliers=2225 (18.28%), Std Dev=1202.83
Shape: Skewness=18.72 (Highly skewed), Kurtosis=606.42 (Heavy tailed)

Column: CARBON_EMISSIONS_SCOPE_12
Data Type: float64, Missing Values: 6891 (35.75%)
Range: Min=0.00, Max=565449323.00, Mean=1499920.50, Median=29763.00
Percentiles: 10th=917.40, 25th=4804.00, 75th=203983.00, 90th=1297883.80
Distribution: Zero values: 9 (0.07%), Outliers=2028 (16.37%), Std Dev=11511807.84
Shape: Skewness=23.35 (Highly skewed), Kurtosis=793.01 (Heavy tailed)

Column: CARBON_EMISSIONS_EVIC_SCOPE_12_INTEN
Data Type: float64, Missing Values: 7113 (36.90%)
Range: Min=0.00, Max=34863.64, Mean=146.36, Median=10.06
Percentiles: 10th=0.36, 25th=1.67, 75th=51.28, 90th=256.82
Distribution: Zero values: 64 (0.53%), Outliers=1854 (15.24%), Std Dev=738.89
Shape: Skewness=20.46 (Highly skewed), Kurtosis=690.68 (Heavy tailed)

Column: CARBON_EMISSIONS_SCOPE_12_INTEN
Data Type: float64, Missing Values: 6790 (35.23%)
Range: Min=0.00, Max=83277.17, Mean=267.74, Median=28.98
Percentiles: 10th=1.88, 25th=6.57, 75th=102.40, 90th=427.76
Distribution: Zero values: 22 (0.18%), Outliers=1911 (15.31%), Std Dev=1357.62
Shape: Skewness=26.26 (Highly skewed), Kurtosis=1272.55 (Heavy tailed)

Column: CARBON_EMISSIONS_INTENSITY_YEAR
Data Type: float64, Missing Values: 6790 (35.23%)
Range: Min=2017.00, Max=2024.00, Mean=2022.22, Median=2022.00
Percentiles: 10th=2022.00, 25th=2022.00, 75th=2023.00, 90th=2023.00
Distribution: Outliers=65 (0.52%), Std Dev=0.60
Year distribution: 2021: 875 (7.01%), 2022: 7826 (62.68%), 2023: 3679 (29.47%)

Column: CARBON_EMISSIONS_SCOPE_2
Data Type: float64, Missing Values: 6892 (35.75%)
Range: Min=0.00, Max=121129435.46, Mean=214018.21, Median=14879.43
Percentiles: 10th=460.00, 25th=2729.62, 75th=77713.00, 90th=336486.30
Distribution: Zero values: 51 (0.41%), Outliers=1870 (15.10%), Std Dev=1559239.21
Shape: Skewness=44.92 (Highly skewed), Kurtosis=3072.98 (Heavy tailed)

Column: CARBON_EMISSIONS_EVIC_SCOPE_2_INTEN
Data Type: float64, Missing Values: 7113 (36.90%)
Range: Min=0.00, Max=2875.47, Mean=26.30, Median=4.78
Percentiles: 10th=0.20, 25th=0.96, 75th=20.52, 90th=62.38
Distribution: Zero values: 170 (1.40%), Outliers=1517 (12.47%), Std Dev=77.81
Shape: Skewness=11.05 (Highly skewed), Kurtosis=229.88 (Heavy tailed)

Column: CARBON_EMISSIONS_SCOPE_2_INTEN
Data Type: float64, Missing Values: 7106 (36.86%)
Range: Min=0.00, Max=27869.57, Mean=54.42, Median=13.38
Percentiles: 10th=1.08, 25th=4.03, 75th=43.60, 90th=118.99
Distribution: Zero values: 97 (0.80%), Outliers=1452 (11.93%), Std Dev=369.58
Shape: Skewness=57.02 (Highly skewed), Kurtosis=3844.74 (Heavy tailed)

Column: CARBON_EMISSIONS_SCOPE_3
Data Type: float64, Missing Values: 14527 (75.36%)
Range: Min=0.00, Max=1147000000.00, Mean=8903623.85, Median=206296.73
Percentiles: 10th=757.83, 25th=10466.00, 75th=2023878.00, 90th=12225624.80
Distribution: Zero values: 5 (0.11%), Outliers=789 (16.61%), Std Dev=46431682.28
Shape: Skewness=12.13 (Highly skewed), Kurtosis=200.95 (Heavy tailed)

Column: EVIC_EUR
Data Type: float64, Missing Values: 6685 (34.68%)
Range: Min=0.74, Max=6395000.00, Mean=16067.92, Median=2635.26
Percentiles: 10th=448.40, 25th=929.68, 75th=8267.24, 90th=26418.12
Distribution: Outliers=1612 (12.80%), Std Dev=105131.79
Shape: Skewness=32.60 (Highly skewed), Kurtosis=1496.67 (Heavy tailed)

Column: CBN_EVIC_PUB_DATE
Data Type: float64, Missing Values: 7039 (36.52%)
Range: Min=20201226.00, Max=20241211.00, Mean=20240320.12, Median=20240606.00
Percentiles: 10th=20240305.00, 25th=20240403.00, 75th=20241002.00, 90th=20241106.00
Distribution: Outliers=372 (3.04%), Std Dev=2191.57
Shape: Skewness=-7.77 (Highly skewed), Kurtosis=81.62 (Heavy tailed)

Column: EVIC_USD_RECENT
Data Type: float64, Missing Values: 6694 (34.73%)
Range: Min=0.81, Max=7064236.75, Mean=17673.67, Median=2901.88
Percentiles: 10th=488.73, 25th=1022.17, 75th=9094.16, 90th=28867.97
Distribution: Outliers=1612 (12.81%), Std Dev=115616.08
Shape: Skewness=32.73 (Highly skewed), Kurtosis=1512.31 (Heavy tailed)

Column: EVIC_USD_YEAR
Data Type: float64, Missing Values: 6694 (34.73%)
Range: Min=2015.00, Max=2024.00, Mean=2022.94, Median=2023.00
Percentiles: 10th=2023.00, 25th=2023.00, 75th=2023.00, 90th=2023.00
Distribution: Outliers=1782 (14.16%), Std Dev=0.58
Year distribution: 2022: 673 (5.35%), 2023: 10800 (85.84%), 2024: 788 (6.26%)

Column: CARBON_SCOPE_12_INTEN_3Y_AVG
Data Type: float64, Missing Values: 11553 (59.93%)
Range: Min=0.00, Max=83270.97, Mean=328.41, Median=29.56
Percentiles: 10th=1.66, 25th=6.61, 75th=139.17, 90th=561.16
Distribution: Zero values: 11 (0.14%), Outliers=1160 (15.02%), Std Dev=1581.91
Shape: Skewness=24.17 (Highly skewed), Kurtosis=1048.80 (Heavy tailed)

Column: CARBON_SCOPE_12_INTEN_3Y_GICS_IND_PEER_RATIO
Data Type: float64, Missing Values: 12413 (64.40%)
Range: Min=1.87, Max=4117.77, Mean=265.26, Median=34.13
Percentiles: 10th=6.53, 25th=15.72, 75th=107.30, 90th=739.03
Distribution: Outliers=1394 (20.31%), Std Dev=707.16
Shape: Skewness=4.08 (Highly skewed), Kurtosis=16.79 (Heavy tailed)

Column: CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR
Data Type: float64, Missing Values: 8117 (42.11%)
Range: Min=-100.00, Max=647.10, Mean=-1.67, Median=-2.09
Percentiles: 10th=-16.01, 25th=-8.73, 75th=1.94, 90th=10.83
Distribution: Zero values: 386 (3.46%), Negative values: 6648 (59.58%), Outliers=1076 (9.64%), Std Dev=21.03
Shape: Skewness=9.33 (Highly skewed), Kurtosis=196.77 (Heavy tailed)


CATEGORICAL COLUMNS ANALYSIS
--------------------------------------------------------------------------------
Column: CARBON_EMISSIONS_SCOPE_1_KEY
Data Type: object, Missing Values: 6892 (35.75%), Unique Values: 10
Top 5 categories (out of 10):
  - Reported: 7479 (38.80%)
  - E.Segmt-Moderate: 1494 (7.75%)
  - E.Segmt-Moderately Low: 1199 (6.22%)
  - E.Segmt-Low: 889 (4.61%)
  - E.CSI: 604 (3.13%)
Potential relationships with: CARBON_EMISSIONS_SCOPE_12_KEY, CARBON_EMISSIONS_SCOPE_2_KEY

Column: CARBON_EMISSIONS_SCOPE_12_KEY
Data Type: object, Missing Values: 6898 (35.79%), Unique Values: 10
Top 5 categories (out of 10):
  - Reported: 7554 (39.19%)
  - E.Segmt-Moderate: 1387 (7.20%)
  - E.Segmt-Moderately Low: 1345 (6.98%)
  - E.Segmt-Low: 965 (5.01%)
  - E.CSI: 712 (3.69%)
Potential relationships with: CARBON_EMISSIONS_SCOPE_1_KEY, CARBON_EMISSIONS_SCOPE_2_KEY

Column: CARBON_EMISSIONS_SCOPE_2_KEY
Data Type: object, Missing Values: 6892 (35.75%), Unique Values: 8
Top 5 categories (out of 8):
  - Reported: 7383 (38.30%)
  - E.Segmt-Moderate: 1910 (9.91%)
  - E.Segmt-Moderately Low: 1043 (5.41%)
  - E.Segmt-Moderately High: 929 (4.82%)
  - E.CSI: 705 (3.66%)
Potential relationships with: CARBON_EMISSIONS_SCOPE_1_KEY, CARBON_EMISSIONS_SCOPE_12_KEY


TEXT/HIGH CARDINALITY COLUMNS ANALYSIS
--------------------------------------------------------------------------------
(Analysis skipped for ISSUER_TICKER, ISSUER_ISIN, ISSUER_NAME, NACE_CLASS_DESCRIPTION)

STRONGLY CORRELATED NUMERICAL COLUMNS
--------------------------------------------------------------------------------
  - MarketCap_USD and EVIC_EUR: 0.7249
  - MarketCap_USD and EVIC_USD_RECENT: 0.7260
  - CARBON_EMISSIONS_EVIC_SCOPE_1_INTEN and CARBON_EMISSIONS_EVIC_SCOPE_12_INTEN: 0.9949
  - CARBON_EMISSIONS_SCOPE_1_INTEN and CARBON_EMISSIONS_SCOPE_12_INTEN: 0.9654
  - CARBON_EMISSIONS_SCOPE_12 and CARBON_EMISSIONS_SCOPE_2: 0.9005
  - EVIC_EUR and EVIC_USD_RECENT: 0.9986
  - CARBON_EMISSIONS_EVIC_SCOPE_2_INTEN and CARBON_EMISSIONS_SCOPE_2_INTEN: 0.9024 