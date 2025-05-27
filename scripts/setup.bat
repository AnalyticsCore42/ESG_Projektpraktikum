@echo off
echo Attempting editable install...

REM Ensure requirements are installed first (optional)
if exist requirements.txt (
  echo Installing requirements from requirements.txt...
  pip install -r requirements.txt
  if %errorlevel% neq 0 (
    echo Error installing requirements. Please check requirements.txt and your pip setup.
    goto :error
  )
) else (
  echo No requirements.txt found, skipping requirement installation.
)

REM Perform the editable install
pip install -e .

if %errorlevel% equ 0 (
  echo Editable install successful!
  echo You may need to reload your VS Code window or restart the Python interpreter/kernel.
  goto :eof
) else (
  echo Editable install failed. Please ensure:
  echo 1. You have activated your virtual environment (.venv).
  echo 2. You have pip and setuptools installed/updated in the venv.
  echo 3. Your pyproject.toml file is correctly configured.
  goto :error
)

:error
echo Setup script failed.
exit /b 1

:eof
exit /b 0