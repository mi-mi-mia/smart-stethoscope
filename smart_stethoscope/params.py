from pathlib import Path

TARGET_SAMPLING_RATE = 22050


def get_repo_root() -> Path:
    """
    Returns the repo root directory in a way that works for scripts and notebooks.
    """
    try:
        # Running from a Python script
        return Path(__file__).parent.parent.resolve()
    except NameError:
        # Running from a notebook
        return Path().resolve().parent


# Set repo_root automatically when module is imported
repo_root = get_repo_root()

RAW_AUDIO_PATH = (
    repo_root
    / "raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
)
PREPROCESSED_AUDIO_PATH = repo_root / "preprocessed_data/audio_breathing_cycles/"
DIAGNOSIS_PATH = (
    repo_root
    / "raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
)
DEMOGRAPHIC_DATA_PATH = repo_root / "raw_data/demographic_info.txt"
CACHE_PATH = repo_root / "preprocessed_data/"
