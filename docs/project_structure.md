# ESG Analysis Project Structure

_Last updated: 2025-05-27_

This document provides an overview of the project's directory structure and organization.

## Directory Structure

```
ESG_Analysis_Project/
├── config/               # Configuration files
├── data/                 # Input data files
├── docs/                 # Documentation
│   ├── data_dictionaries/# Data dictionaries for all datasets
│   ├── fastai/           # FastAI documentation references
│   ├── config/           # Data column references and config docs
│   └── project/          # Project structure and detailed docs
├── output/               # All output files
│   ├── summary/          # Key summary figures and results for reports
│   │   ├── png/          # Summary PNG figures
│   │   ├── pdf/          # Summary PDF figures
│   │   └── others/       # Other summary outputs
│   └── details/          # Detailed outputs by analysis
│       ├── analysis_11/  # Detailed outputs from script 11
│       ├── analysis_12/  # Detailed outputs from script 12
│       └── program_focused_analysis/ # Detailed outputs from program-focused analysis
├── reports/              # Analysis reports and findings
│   └── main_report/      # Main LaTeX source and compiled PDF
├── scripts/              # Entry point scripts for running analyses
├── src/                  # Source code
│   ├── analysis/         # Analysis scripts
│   ├── tools/            # Tool scripts (e.g., scraping, feature importance)
│   ├── utils/            # Utility functions and helpers
│   └── kaggle_scripts/   # Kaggle/notebook-specific scripts
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies
```

## Key Components

### Analysis Scripts

The main analysis scripts are located in `src/analysis/` with the following naming convention:
- `analysis_XX_name.py` where XX is a number indicating the sequence

Key analysis scripts include:
- `analysis_04_targets_df.py` - Analysis of reduction targets
- `analysis_05_programs1_basic_analysis.py` - Basic analysis of reduction programs 1
- `analysis_14_fastai_emissions_prediction.py` - FastAI based emissions prediction
- `analysis_15_industry_segmented_emissions.py` - Industry-segmented emissions analysis
- `analysis_16_gbm_industry_emissions.py` - GBM-based industry emissions prediction
- `analysis_17_program_focused_gbm.py` - Program-focused GBM analysis

### Reports

Analysis reports are stored in the `reports/` directory with a similar naming convention:
- `XX_report_name.md` where XX is a number indicating the sequence

Important reports include:
- `01_emissions_analysis.md` - Overview of emissions data
- `02_targets_analysis.md` - Analysis of reduction targets
- `program_focused_analysis_findings.md` - Findings from program-focused GBM analysis
- `esg_ml_model_summary.md` - Summary of ML approaches

### Running the Analysis

1. Entry point scripts in the `scripts/` directory provide easy access to running key analyses:
   - `generate_report_visuals.py` - Generate all summary figures for the report

2. Or you can run scripts directly from the src directory:
   ```bash
   python -m src.analysis.analysis_17_program_focused_gbm
   ```

### Visualization and Results

- Visualizations are stored in `output/summary/png/` and `output/summary/pdf/`
- Model results are stored in `output/details/`

## Documentation

- Dataset documentation is available in `docs/data_dictionaries/`
- Project structure is documented in this file and in the main README.md
- Machine learning approaches are described in `docs/ml_approaches.md` 