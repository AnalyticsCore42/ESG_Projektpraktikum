# ESG Analysis Project Structure (Detailed)

_Last updated: 2025-05-27_

This document provides a detailed overview of the project's directory structure and organization.

## Directory Structure

```
ESG_Analysis_Project/
├── data/
│   ├── company_emissions_merged.csv
│   ├── Reduktionsprogramme 1 Results - 20241212 15_45_26.csv
│   ├── Reduktionsprogramme 2 Results - 20241212 15_51_07.csv
│   ├── Reduktionsziele Results - 20241212 15_49_29.csv
│   ├── Treibhausgasemissionen Results - 20241212 15_44_03.csv
│   └── Unternehmensdaten Results - 20241212 15_41_41.csv
├── docs/
│   ├── config/
│   │   └── data_column_reference.md
│   ├── data_dictionaries/
│   │   ├── company_emissions_data.md
│   │   ├── index.md
│   │   ├── programs1_data.md
│   │   ├── programs2_data.md
│   │   └── targets_data.md
│   ├── fastai/
│   │   ├── developer/
│   │   ├── tabular/
│   │   └── tabular_learner_guide.md
│   ├── index.md
│   ├── ml_approaches.md
│   ├── project/
│   │   └── project_structure_detailed.md
│   ├── project_structure.md
│   └── visualization_files.txt
├── output/
│   ├── summary/
│   │   ├── png/
│   │   ├── pdf/
│   │   └── others/
│   └── details/
│       ├── analysis_11/
│       ├── analysis_12/
│       └── program_focused_analysis/
├── reports/
│   ├── 01_emissions_analysis.md
│   ├── 02_targets_analysis.md
│   ├── 03_programs1_analysis.md
│   ├── 04_programs2_detailed_analysis.md
│   ├── 05_programs_combined_analysis.md
│   ├── 06_program_sequencing_analysis.md
│   ├── 07_association_rule_analysis.md
│   ├── 08_advanced_esg_analysis.md
│   ├── 09_additional_esg_insights.md
│   ├── 10_comprehensive_esg_insights.md
│   ├── 11_overall_findings.md
│   ├── esg_ml_analysis_plan.md
│   ├── esg_ml_model_summary.md
│   ├── main_report/
│   │   ├── esg_report_fixed.tex
│   │   ├── reportv1.md
│   │   └── V2_ESG_Report_final_Boldt_13April.pdf
│   └── program_focused_analysis_findings.md
├── requirements.txt
├── scripts/
│   ├── generate_report_visuals.py
│   └── scan_structure.py
├── src/
│   ├── analysis/
│   ├── config/
│   ├── tools/
│   ├── utils/
│   └── kaggle_scripts/
└── README.md
```

## Notes
- The structure above reflects the current, cleaned-up state of the project.
- All obsolete, duplicate, or empty folders have been removed.
- Output files are now consistently saved in the appropriate subfolders under `output/`.
- Kaggle/notebook-specific scripts are in `src/kaggle_scripts/`.
- Data dictionaries and documentation are in `docs/data_dictionaries/`.
- For further details, see the main README.md and docs/index.md.
