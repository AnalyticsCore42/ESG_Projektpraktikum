# ESG Data Column Quick Reference

This is a quick reference to find information about all datasets and columns in the ESG Analysis Project.

For complete documentation, see the following resources:

- [Documentation Index](../docs/index.md)
- [Data Dictionary Index](../docs/data_dictionaries/index.md)

## Datasets

| Dataset | Documentation | Summary Report |
|---------|---------------|----------------|
| Company Emissions | [Data Dictionary](../docs/data_dictionaries/company_emissions_data.md) | [Summary](../reports/01_emissions_analysis.md) |
| Reduction Targets | [Data Dictionary](../docs/data_dictionaries/targets_data.md) | [Summary](../reports/02_targets_analysis.md) |
| Reduction Programs 1 | [Data Dictionary](../docs/data_dictionaries/programs1_data.md) | [Summary](../reports/03_programs1_analysis.md) |
| Reduction Programs 2 | [Data Dictionary](../docs/data_dictionaries/programs2_data.md) | [Summary](../reports/04_programs2_detailed_analysis.md) |

## Key Analysis Scripts

The main analysis scripts are available in the `src/analysis/` directory:

- `analysis_04_targets_df.py` - Targets analysis
- `analysis_05_programs1_basic_analysis.py` - Basic analysis of Programs 1
- `analysis_06_program1-df_detailedv1.py` - Detailed visualization of Programs 1
- `analysis_07_programs2_detailed_analysis.py` - Detailed analysis of Programs 2
- `analysis_08_programs_combined_detailed.py` - Combined analysis of Programs 1 and 2
- `analysis_09_advanced_esg_analysis.py` - Advanced analysis of all datasets
- `analysis_10_additional_esg_insights.py` - Additional insights
- `analysis_12_program_sequencing_analysis.py` - Analysis of program implementation sequence
- `analysis_13_association_rule_mining.py` - Patterns in program implementation

## Common Analysis Tasks

| Task | Relevant Columns | Example Script |
|------|------------------|----------------|
| Emission trend analysis | `CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR` | `analysis_09_advanced_esg_analysis.py` |
| Strategy effectiveness | `CBN_GHG_MITIG_*` columns | `analysis_06_program1-df_detailedv1.py` |
| Target progress | `TARGET_CARBON_PROGRESS_PCT` | `analysis_04_targets_df.py` |
| Program implementation | `CARBON_PROGRAMS_CATEGORY`, `CARBON_PROGRAMS_IMP_YEAR` | `analysis_07_programs2_detailed_analysis.py` |
| Association rules | Multiple | `analysis_13_association_rule_mining.py` | 