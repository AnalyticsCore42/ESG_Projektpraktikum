
====== COMBINED PROGRAMS ANALYSIS ======
================================================================================

--- DATASETS OVERLAP ANALYSIS ---
================================================================================
Total unique companies across both datasets: 11,165
Companies in both datasets: 2,867 (25.68%)
Companies only in programs1_df: 6 (0.05%)
Companies only in programs2_df: 8,292 (74.27%)

Top 10 countries in the overlap:
  - US: 673 companies (23.47%)
  - JP: 297 companies (10.36%)
  - CN: 239 companies (8.34%)
  - IN: 176 companies (6.14%)
  - CA: 162 companies (5.65%)
  - GB: 126 companies (4.39%)
  - KR: 126 companies (4.39%)
  - AU: 111 companies (3.87%)
  - TW: 81 companies (2.83%)
  - BR: 57 companies (1.99%)

--- MITIGATION STRATEGY vs PROGRAM IMPLEMENTATION ANALYSIS ---
================================================================================
Companies with both mitigation strategy data and program implementation data: 2867

1. MITIGATION SCORE vs PROGRAM COUNT
--------------------------------------------------------------------------------
Correlation between combined mitigation score and program count: 0.189

Correlations between individual mitigation strategies and program count:
  - Distribution: 0.099
  - Raw_Mat: 0.093
  - Mfg: 0.075
  - Transport: 0.104
  - Capture: 0.202

2. MITIGATION STRATEGIES vs PROGRAM CATEGORIES
--------------------------------------------------------------------------------
Top 10 strongest correlations between mitigation strategies and program categories:
  - Distribution vs Minimum practices expected based on domestic industry norms: nan
  - Raw_Mat vs Minimum practices expected based on domestic industry norms: nan
  - Mfg vs Less than 50% of energy requirements from low-carbon emissions alternatives (estimated): 0.235
  - Raw_Mat vs Less than 50% of energy requirements from low-carbon emissions alternatives (estimated): 0.232
  - Mfg vs Minimum practices expected based on domestic industry norms: nan
  - Transport vs Less than 50% of energy requirements from low-carbon emissions alternatives (estimated): 0.205
  - Mfg vs No evidence: -0.173
  - Raw_Mat vs No evidence: -0.170
  - Transport vs Minimum practices expected based on domestic industry norms: nan
  - Capture vs Less than 50% of energy requirements from low-carbon emissions alternatives (estimated): 0.151

3. IMPLEMENTATION TIMELINE vs MITIGATION SCORE
--------------------------------------------------------------------------------
Correlation between avg implementation year and combined mitigation score: -0.142

Correlations between avg implementation year and individual mitigation strategies:
  - Distribution: -0.091
  - Raw_Mat: -0.062
  - Mfg: -0.055
  - Transport: -0.073
  - Capture: -0.148

--- EMISSIONS RELATIONSHIPS ANALYSIS ---
================================================================================
Companies with complete data (strategies, programs, and emissions): 2867

1. EMISSIONS vs STRATEGIES AND PROGRAMS
--------------------------------------------------------------------------------

Correlations with emissions intensity trend (3Y CAGR):
  - Combined_Mitigation_Score: -0.031
  - program_count: -0.112
  - CBN_GHG_MITIG_DISTRIBUTION_Score: -0.018
  - CBN_GHG_MITIG_RAW_MAT_Score: -0.005
  - CBN_GHG_MITIG_MFG_Score: 0.010
  - CBN_GHG_MITIG_TRANSPORT_Score: -0.007
  - CBN_GHG_MITIG_CAPTURE_Score: -0.074

2. COMBINED EFFECT OF MITIGATION STRATEGIES AND PROGRAMS
--------------------------------------------------------------------------------
Average emission trends by mitigation strategy level and program implementation level:

  Mitigation Level: Low
    - Program Level Low: 1.56% trend (341 companies)
    - Program Level Medium-Low: -0.66% trend (416 companies)
    - Program Level Medium-High: -3.92% trend (901 companies)
    - Program Level High: -4.55% trend (151 companies)

  Mitigation Level: Medium-Low
    - Program Level Low: -1.10% trend (53 companies)
    - Program Level Medium-Low: -1.43% trend (168 companies)
    - Program Level Medium-High: -4.06% trend (409 companies)
    - Program Level High: -2.87% trend (78 companies)

  Mitigation Level: Medium-High
    - Program Level Low: -0.30% trend (16 companies)
    - Program Level Medium-Low: -1.88% trend (59 companies)
    - Program Level Medium-High: -3.04% trend (158 companies)
    - Program Level High: 1.02% trend (27 companies)

  Mitigation Level: High
    - Program Level Low: -15.74% trend (2 companies)
    - Program Level Medium-Low: 4.19% trend (9 companies)
    - Program Level Medium-High: -7.07% trend (61 companies)
    - Program Level High: -4.10% trend (17 companies)

3. BEST PERFORMING COMBINATIONS
--------------------------------------------------------------------------------

Best performing combinations (highest emissions reductions):
  - High mitigation + Low programs: -15.74% trend (2 companies)
  - High mitigation + Medium-High programs: -7.07% trend (61 companies)
  - Low mitigation + High programs: -4.55% trend (151 companies)
  - High mitigation + High programs: -4.10% trend (17 companies)
  - Medium-Low mitigation + Medium-High programs: -4.06% trend (409 companies)

====== ANALYSIS COMPLETE ======
