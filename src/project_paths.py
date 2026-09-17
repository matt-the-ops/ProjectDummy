from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
ROOT_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = MODELS_DIR


def _resolve_path(base_dir: Path, *parts: str) -> Path:
    candidate = base_dir.joinpath(*parts)
    if candidate.exists():
        return candidate

    legacy_candidate = ROOT_DIR.joinpath(*parts)
    if legacy_candidate.exists():
        return legacy_candidate

    return candidate


def data_path(*parts: str) -> Path:
    return _resolve_path(DATA_DIR, *parts)


def model_path(*parts: str) -> Path:
    return _resolve_path(MODELS_DIR, *parts)


def asset_path(*parts: str) -> Path:
    return _resolve_path(ASSETS_DIR, *parts)
