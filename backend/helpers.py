from pathlib import Path

def get_project_root() -> Path:
    """Finds the root directory by looking for the .git folder."""
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if (parent / ".git").exists():
            return parent
    return curr.parent  # Fallback to current folder

ROOT = get_project_root()