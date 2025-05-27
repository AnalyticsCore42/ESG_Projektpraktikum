import sys
import os

def find_project_root():
    """Finds the project root directory by looking for a marker file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, ".git")) or \
           os.path.exists(os.path.join(current_dir, "pyproject.toml")) or \
           os.path.exists(os.path.join(current_dir, "setup.py")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir

def add_project_root_to_path():
    """Adds the project root directory to sys.path."""
    project_root = find_project_root()
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Project root found and added to sys.path by src/__init__.py: {project_root}") # Optional: for debugging

add_project_root_to_path()