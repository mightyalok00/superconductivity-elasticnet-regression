from pathlib import Path

from config import PROJECT_ROOT, settings


def test_settings_use_portable_project_paths():
    assert settings.train_path == PROJECT_ROOT / "data" / "raw" / "train.csv"
    assert settings.model_path == PROJECT_ROOT / "outputs" / "models" / "elasticnet_v1.joblib"
    assert "Users\\Alok Agarwal" not in str(settings.train_path)


def test_public_demo_security_controls_are_disabled_by_default():
    assert settings.enable_rate_limit is False
    assert settings.require_api_key is False
    assert settings.requests_per_minute > 0


def test_project_root_exists():
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()
