from pathlib import Path

# Exact Windows paths requested by the user.
WINDOWS_TRAIN_PATH = Path(r"C:\Users\Alok Agarwal\Downloads\superconductivty+data\train.csv")
WINDOWS_UNIQUE_PATH = Path(r"C:\Users\Alok Agarwal\Downloads\superconductivty+data\unique_m.csv")

# Portable fallback paths bundled with this project.
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
LOCAL_UNIQUE_PATH = PROJECT_ROOT / "data" / "raw" / "unique_m.csv"

TARGET = "critical_temp"
RANDOM_STATE = 42
TEST_SIZE = 0.20
