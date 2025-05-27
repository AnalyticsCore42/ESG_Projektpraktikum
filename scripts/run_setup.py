# In ESG_Analysis_Project/run_setup.py
import subprocess
import sys
import os
import platform

# --- Prerequisite Check ---
print("--- Checking Setup Prerequisites ---")

# 1. Check if running inside a virtual environment
#    sys.prefix == sys.base_prefix check works on POSIX and Windows venv/virtualenv
is_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV") is not None

if not is_venv:
    print("\n❌ ERROR: Not running inside a detected virtual environment.")
    print("This script needs to run AFTER you create and activate a virtual environment.")
    print("\nPlease follow these steps in your terminal first:")
    print("  1. Navigate to the project directory: cd /path/to/ESG_Analysis_Project")
    print("  2. Create venv (if you haven't): python3 -m venv .venv  (or python -m venv .venv)")
    print("     -> If this fails, you might need to install the venv package for your system")
    print("        (e.g., 'sudo apt install python3.10-venv' on Debian/Ubuntu)")
    print("  3. Activate the venv:")
    print("     - Linux/macOS: source .venv/bin/activate")
    print("     - Windows CMD: .\\.venv\\Scripts\\activate")
    print("     - Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1")
    print("\nThen, while the venv is active (prompt shows '(.venv)'), re-run this script:")
    print("  python run_setup.py")
    sys.exit(1)
else:
    print("✅ Virtual environment detected.")
    print(f"   Using Python interpreter: {sys.executable}")

# Define pip command using the venv's Python for robustness
pip_cmd = [sys.executable, "-m", "pip"]

# 2. Check if pip is available (basic check)
try:
    # Use run instead of check_call to capture output if needed, suppress otherwise
    subprocess.run([*pip_cmd, "--version"], check=True, capture_output=True, text=True)
    print("✅ pip command is available.")
except subprocess.CalledProcessError as e:
    print(f"\n❌ ERROR: 'pip' command failed using the current Python interpreter.")
    print(f"   Attempted command: {' '.join(e.cmd)}")
    print(f"   Error: {e.stderr}")
    print("   Your virtual environment might be corrupted or pip is not installed correctly.")
    sys.exit(1)
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find Python executable '{sys.executable}' or run pip.")
    print("   Your virtual environment might be broken.")
    sys.exit(1)

# --- Installation Steps ---
print("\n--- Starting Installation ---")
print("(This might take a few minutes depending on dependencies and network speed...)")

try:
    # Ensure pip, setuptools, and wheel are up-to-date within the venv
    print("Ensuring pip, setuptools, and wheel are up-to-date...")
    subprocess.run([*pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture_output=True, text=True)

    # Install requirements first
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"Installing requirements from {req_file}...")
        subprocess.run([*pip_cmd, "install", "-r", req_file], check=True, capture_output=True, text=True)
        print("✅ Requirements installed.")
    else:
         print(f"ℹ️ No {req_file} found, skipping requirement installation.")

    # Perform the editable install
    print("Running editable install (pip install -e .)...")
    subprocess.run([*pip_cmd, "install", "-e", "."], check=True, capture_output=True, text=True)
    print("✅ Project installed in editable mode.")

    print("\n--- ✅ Setup Successful! ---")
    print("\nNext Steps:")
    print("1. Ensure VS Code is using the Python interpreter from this environment:")
    print(f"   -> '{sys.prefix}'")
    print("   (Use Command Palette -> 'Python: Select Interpreter')")
    print("2. Reload the VS Code window (Command Palette -> 'Developer: Reload Window').")
    print("3. IMPORTANT: Run analysis scripts as modules from the PROJECT ROOT directory:")
    print("   Example: python -m src.analysis.analysis_01_targets_df")
    print("   (Ensure your venv is active in the terminal first!)")
    print("4. Do NOT use 'import src'. Import specific functions/classes like:")
    print("   from utils.helpers import your_function")


except subprocess.CalledProcessError as e:
    print(f"\n--- ❌ ERROR during installation ---")
    print(f"Command failed: {' '.join(e.cmd)}")
    print(f"Return code: {e.returncode}")
    # Try to decode stderr/stdout for more info
    stderr = e.stderr.strip() if e.stderr else ""
    stdout = e.stdout.strip() if e.stdout else ""
    if stderr:
        print(f"Error Output:\n---\n{stderr}\n---")
    elif stdout:
         print(f"Output (might contain error details):\n---\n{stdout}\n---")
    print("\nPlease check the error message above.")
    print("Common issues: missing system dependencies for a package, network problems,")
    print("incorrect pyproject.toml or requirements.txt.")
    sys.exit(1)
except Exception as e:
    print(f"\n--- ❌ An unexpected error occurred ---")
    print(e)
    sys.exit(1)

sys.exit(0)