PROGRAMS1 DATAFRAME: shape=(2873, 10)

Columns:
1. ISSUERID
2. ISSUER_NAME
3. ISSUER_TICKER
4. ISSUER_ISIN
5. ISSUER_CNTRY_DOMICILE
6. CBN_GHG_MITIG_DISTRIBUTION
7. CBN_GHG_MITIG_RAW_MAT
8. CBN_GHG_MITIG_MFG
9. CBN_GHG_MITIG_TRANSPORT
10. CBN_GHG_MITIG_CAPTURE


IDENTIFIER COLUMNS ANALYSIS
--------------------------------------------------------------------------------
1. ISSUERID - Missing Values: 0 (0.00%), Unique Values: 2873
2. ISSUER_NAME - Missing Values: 0 (0.00%), Unique Values: 2873
3. ISSUER_TICKER - Missing Values: 14 (0.49%), Unique Values: 2740
4. ISSUER_ISIN - Missing Values: 13 (0.45%), Unique Values: 2860

GEOGRAPHICAL DATA ANALYSIS
--------------------------------------------------------------------------------
5. ISSUER_CNTRY_DOMICILE

Top 10 countries:
  - US: 673 (23.42%)
  - JP: 297 (10.34%)
  - CN: 239 (8.32%)
  - IN: 176 (6.13%)
  - CA: 162 (5.64%)
  - GB: 126 (4.39%)
  - KR: 126 (4.39%)
  - AU: 113 (3.93%)
  - TW: 81 (2.82%)
  - BR: 57 (1.98%)
Total unique countries: 68


NUMERICAL COLUMNS ANALYSIS
--------------------------------------------------------------------------------
(None in this specific dataframe)

CATEGORICAL COLUMNS ANALYSIS
--------------------------------------------------------------------------------
6. CBN_GHG_MITIG_DISTRIBUTION
Data Type: object, Missing Values: 2059 (71.67%), Unique Values: 4
Categories by frequency:
  - Some stores/distribution centers (anecdotal cases): 407 (14.17%)
  - No: 220 (7.66%)
  - General statement: 109 (3.79%)
  - All or most stores and distribution centers: 78 (2.71%)

7. CBN_GHG_MITIG_RAW_MAT
Data Type: object, Missing Values: 1867 (64.98%), Unique Values: 4
Categories by frequency:
  - No: 520 (18.10%)
  - Some products (anecdotal cases): 252 (8.77%)
  - General statement: 207 (7.21%)
  - All or core products: 27 (0.94%)

8. CBN_GHG_MITIG_MFG
Data Type: object, Missing Values: 1867 (64.98%), Unique Values: 4
Categories by frequency:
  - Some facilities (anecdotal cases): 389 (13.54%)
  - No: 385 (13.40%)
  - General statement: 135 (4.70%)
  - All or core production facilities: 97 (3.38%)

9. CBN_GHG_MITIG_TRANSPORT
Data Type: object, Missing Values: 1867 (64.98%), Unique Values: 4
Categories by frequency:
  - Improvements in fleet, routes, OR load/packaging optimization: 370 (12.88%)
  - No: 257 (8.95%)
  - Improvements in fleet, routes, AND load/packaging optimization: 215 (7.48%)
  - General statement: 164 (5.71%)

10. CBN_GHG_MITIG_CAPTURE
Data Type: object, Missing Values: 999 (34.77%), Unique Values: 4
Categories by frequency:
  - No evidence: 1134 (39.47%)
  - Limited efforts / information: 468 (16.29%)
  - Some efforts: 238 (8.28%)
  - Aggressive efforts: 34 (1.18%)


TEXT/HIGH CARDINALITY COLUMNS ANALYSIS
--------------------------------------------------------------------------------
(None in this specific dataframe) 