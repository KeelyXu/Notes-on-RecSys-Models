from pathlib import Path

def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find the project root directory by looking for a .gitignore file.
    """
    # Use the current file's directory if not provided
    current = start_path or Path(__file__).resolve().parent

    # Traverse upwards until the filesystem root
    for parent in [current, *current.parents]:
        if (parent / ".gitignore").exists():
            return parent

    raise FileNotFoundError("Project root with .gitignore not found")
