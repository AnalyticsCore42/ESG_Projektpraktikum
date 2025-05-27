import sys
import os

def add_project_root_to_path() -> None:
    """
    Adds the project root directory to sys.path.

    Returns:
        None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up until we find a marker file/directory indicating the project root
    while True:
        if os.path.exists(os.path.join(current_dir, ".git")) or \
           os.path.exists(os.path.join(current_dir, "pyproject.toml")) or \
           os.path.exists(os.path.join(current_dir, "setup.py")):  # Add more markers if needed
            project_root = current_dir
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                print(f"Project root found and added to sys.path: {project_root}")
            return

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            print("Error: Could not find project root.")
            return
        current_dir = parent_dir

if __name__ == "__main__":
    add_project_root_to_path()