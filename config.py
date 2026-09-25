from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "Superconductivity Critical Temperature API"
    app_env: str = "production"
    host: str = "0.0.0.0"
    port: int = 8000

    train_path: Path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    unique_path: Path = PROJECT_ROOT / "data" / "raw" / "unique_m.csv"
    model_path: Path = PROJECT_ROOT / "outputs" / "models" / "elasticnet_v1.joblib"
    metadata_path: Path = PROJECT_ROOT / "outputs" / "models" / "model_metadata.json"

    target: str = "critical_temp"
    random_state: int = 42
    test_size: float = 0.20

    cors_origins: str = "*"
    enable_rate_limit: bool = False
    requests_per_minute: int = 120
    require_api_key: bool = False
    api_key: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="SC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Backward-compatible constants for analysis modules.
LOCAL_TRAIN_PATH = settings.train_path
LOCAL_UNIQUE_PATH = settings.unique_path
TARGET = settings.target
RANDOM_STATE = settings.random_state
TEST_SIZE = settings.test_size
