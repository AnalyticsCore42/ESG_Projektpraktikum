# src/config/paths.py
import sys
from pathlib import Path

# --- Determine Project Root ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

# --- Add Project Root to sys.path if necessary ---
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Define Core Directory Paths ---
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Output Structure ---
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_DIR = OUTPUT_DIR / "summary"
DETAILS_DIR = OUTPUT_DIR / "details"

# --- Summary Subfolders ---
SUMMARY_PNG_DIR = SUMMARY_DIR / "png"
SUMMARY_PDF_DIR = SUMMARY_DIR / "pdf"
SUMMARY_OTHERS_DIR = SUMMARY_DIR / "others"

# --- Details Subfolders (examples, add more as needed) ---
DETAILS_ANALYSIS_11_DIR = DETAILS_DIR / "analysis_11"
DETAILS_ANALYSIS_12_DIR = DETAILS_DIR / "analysis_12"
DETAILS_PROGRAM_FOCUSED_DIR = DETAILS_DIR / "program_focused_analysis"

# --- Optionally: Create directories if they don't exist ---
for d in [
    DATA_DIR, DOCS_DIR, SRC_DIR, REPORTS_DIR,
    OUTPUT_DIR, SUMMARY_DIR, DETAILS_DIR,
    SUMMARY_PNG_DIR, SUMMARY_PDF_DIR, SUMMARY_OTHERS_DIR,
    DETAILS_ANALYSIS_11_DIR, DETAILS_ANALYSIS_12_DIR, DETAILS_PROGRAM_FOCUSED_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# --- Sanity Checks (Optional) ---
if not PROJECT_ROOT.is_dir():
    print(f"Warning: PROJECT_ROOT does not seem to be a valid directory: {PROJECT_ROOT}")
if not DATA_DIR.is_dir():
    print(f"Warning: DATA_DIR does not seem to be a valid directory: {DATA_DIR}")
