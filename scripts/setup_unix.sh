#!/bin/bash

echo "Attempting editable install..."

# Ensure requirements are installed first (optional, but good practice)
if [ -f "requirements.txt" ]; then
  echo "Installing requirements from requirements.txt..."
  pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    echo "Error installing requirements. Please check requirements.txt and your pip setup."
    exit 1
  fi
else
  echo "No requirements.txt found, skipping requirement installation."
fi

# Perform the editable install
pip install -e .

if [ $? -eq 0 ]; then
  echo "Editable install successful!"
  echo "You may need to reload your VS Code window or restart the Python interpreter/kernel."
else
  echo "Editable install failed. Please ensure:"
  echo "1. You have activated your virtual environment (.venv)."
  echo "2. You have pip and setuptools installed/updated in the venv."
  echo "3. Your pyproject.toml file is correctly configured."
  exit 1
fi

exit 0