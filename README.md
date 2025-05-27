# ESG Analysis Project

## Overview
The ESG Analysis Project explores how corporate greenhouse gas emissions relate to sustainability programs and reduction targets. Using advanced data science and machine learning, the project provides modular, reproducible ESG (Environmental, Social, Governance) analyses. Results are organized for both high-level summaries and detailed, script-specific outputs, making it easy to trace findings and reproduce analyses.

_Last updated: 2025-05-27_

---

## Project Structure
```
ESG_Analysis_Project/
├── data/                      # Input data files (CSV)
├── docs/                      # Documentation and data dictionaries
│   ├── data_dictionaries/     # Data dictionary for each dataset
│   ├── fastai/                # fastai and developer documentation
│   ├── config/                # Data column references and config docs
│   └── project/               # Project structure and detailed docs
├── output/                    # All generated outputs
│   ├── summary/               # Key summary figures and results for reports
│   │   ├── png/               # Summary PNG figures
│   │   ├── pdf/               # Summary PDF figures
│   │   └── others/            # Other summary outputs
│   └── details/               # Detailed outputs by analysis
│       ├── analysis_11/       # Detailed outputs from script 11
│       ├── analysis_12/       # Detailed outputs from script 12
│       └── program_focused_analysis/ # Detailed outputs from program-focused analysis
├── reports/                   # LaTeX reports, markdown, and findings
│   └── main_report/           # Main LaTeX source and compiled PDF
├── scripts/                   # Entry-point scripts for running analyses and generating figures
├── src/                       # Source code
│   ├── analysis/              # Modular analysis scripts (e.g., industry, program, GBM)
│   ├── tools/                 # Tool scripts (e.g., scraping, feature importance)
│   ├── utils/                 # Utility modules (data, analysis, visualization)
│   └── kaggle_scripts/        # Kaggle/notebook-specific scripts
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
└── .gitignore
```

---

## Data
- All input data is stored in the `data/` directory.
- Data dictionaries and documentation are in `docs/data_dictionaries/`.
- Main datasets:
  - `company_emissions_merged.csv`
  - `Reduktionsprogramme 1 Results ...csv`
  - `Reduktionsprogramme 2 Results ...csv`
  - `Reduktionsziele Results ...csv`

---

## How to Run Analyses
1. **Set up your environment (first time only):**
   - Create and activate a Python virtual environment:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - On Windows, use:
     ```bat
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Automated project setup:**
   - Run the setup script to install all dependencies and set up the project in editable mode:
     ```bash
     python scripts/run_setup.py
     ```
   - Or, for Unix systems:
     ```bash
     bash scripts/setup_unix.sh
     ```
   - Or, for Windows:
     ```bat
     scripts\setup.bat
     ```

3. **Run analysis scripts:**
   - Each script in `src/analysis/` can be run independently. For example:
     ```bash
     python -m src.analysis.analysis_11_fastai_emissions_prediction
     python -m src.analysis.analysis_12_industry_segmented_emissions
     python -m src.analysis.analysis_14_program_focused_gbm
     ```
   - Outputs are saved to the appropriate subfolder in `output/details/`.

4. **Generate all summary figures for the report:**
   - Use the automated script:
     ```bash
     python scripts/generate_report_visuals.py
     ```
   - Figures will be saved to `output/summary/png/` and `output/summary/pdf/`.

5. **Compile the LaTeX report:**
   - The PDF report is typically compiled in Overleaf using the provided `.tex` file for convenience and collaboration.
   - If you prefer, you can also build the PDF locally. This requires a full LaTeX distribution (e.g., `texlive-full` on Linux), which may involve significant downloads and storage space.
   - To compile locally:
     ```bash
     pdflatex reports/main_report/esg_report_fixed.tex
     ```

---

## Output Organization
- **Summary outputs:**
  - All key figures and tables for the main report are in `output/summary/` (with subfolders for PNG, PDF, and others).
- **Detailed outputs:**
  - High-volume or script-specific outputs are in `output/details/`, organized by analysis (e.g., `analysis_11/`, `analysis_12/`, `program_focused_analysis/`).
- **Reports:**
  - The main LaTeX report and compiled PDF are in `reports/main_report/`.

---

## Documentation
- **Data dictionaries:**
  - `docs/data_dictionaries/` contains detailed documentation for each dataset.
- **Project documentation:**
  - `docs/index.md` and `docs/project_structure.md` provide further details.

---

## Quick Reference
- **Run a specific analysis:**
  ```bash
  python -m src.analysis.analysis_11_fastai_emissions_prediction
  ```
- **Generate all report figures:**
  ```bash
  python scripts/generate_report_visuals.py
  ```
- **Compile the report:**
  ```bash
  pdflatex reports/main_report/esg_report_fixed.tex
  ``` 