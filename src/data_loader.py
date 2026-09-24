from pathlib import Path
import pandas as pd


def resolve_path(primary: Path, fallback: Path) -> Path:
    """Return the requested Windows path when available, otherwise the project fallback."""
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset not found at {primary} or {fallback}")


def load_datasets(train_primary: Path, unique_primary: Path, train_fallback: Path, unique_fallback: Path):
    """Load train.csv and unique_m.csv with clear path fallback behavior."""
    train_path = resolve_path(train_primary, train_fallback)
    unique_path = resolve_path(unique_primary, unique_fallback)
    train_df = pd.read_csv(train_path)
    unique_df = pd.read_csv(unique_path)
    return train_df, unique_df, train_path, unique_path
