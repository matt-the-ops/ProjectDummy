import os

BMO_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../ProjectDummy/bmo
ROOT_DIR    = os.path.dirname(BMO_DIR)                     # .../ProjectDummy

MODELS_DIR  = os.path.join(ROOT_DIR, "models")
ASSETS_DIR  = os.path.join(ROOT_DIR, "assets")
DATA_DIR    = os.path.join(ROOT_DIR, "data")
CONFIG_DIR  = os.path.join(ROOT_DIR, "config")

SETTINGS_FILE  = os.path.join(CONFIG_DIR, "settings.json")
ICON_MAP_FILE  = os.path.join(CONFIG_DIR, "icon_map.json")
ICONS_DIR      = os.path.join(ASSETS_DIR, "fsl_icons")

BMO_VOICE_DIR  = os.path.join(MODELS_DIR, "BMO Voice")
BMO_PTH        = os.path.join(BMO_VOICE_DIR, "BMO.pth")
BMO_INDEX      = os.path.join(BMO_VOICE_DIR, "added_IVF548_Flat_nprobe_1_BMO_v2.index")

LETTER_MODEL_FILE = os.path.join(MODELS_DIR, "fsl_model.pkl")
LETTER_CSV_FILE   = os.path.join(DATA_DIR, "fsl_dataset.csv")
