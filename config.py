"""Shared constants and default parameters for Lab 06 (Face Recognition with SVM).

This file is fixed infrastructure: students should NOT modify it.
It has no Qt dependency, so the example scripts can import it as well.
"""

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
APP_NAME = "Lab06 - Face Recognition with SVM"
ORGANIZATION_NAME = "Sharif CVML Lab"
SETTINGS_NAME = "cvml-lab06-svm"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DEFAULT_DATABASE_DIR = ASSETS_DIR / "database"   # training set: one folder per person
DEFAULT_TEST_DIR = ASSETS_DIR / "test"           # test set: same structure

FACE_LANDMARKER_FILE = "face_landmarker.task"
MAGFACE_FILE = "magface_iresnet100.onnx"
MAGFACE_URL = (
    "https://drive.google.com/file/d/1kdoVMRDo__cnB2OKqJu40Qv7rwApBzHt/view?usp=sharing"
)

SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
EMBEDDING_CACHE_PATH = MODELS_DIR / "embedding_cache.pkl"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# --------------------------------------------------------------------------- #
# Input sources / camera
# --------------------------------------------------------------------------- #
SOURCE_WEBCAM = "webcam"
SOURCE_IMAGE = "image"
SOURCE_VIDEO = "video"

DEFAULT_CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 9
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
TARGET_FPS = 30
MAX_FRAME_SIDE = 1920        # larger frames are down-scaled once, right after reading

# --------------------------------------------------------------------------- #
# Face processing
# --------------------------------------------------------------------------- #
UNKNOWN_LABEL = "Unknown"
EMBEDDING_SIZE = 512
MAX_DETECTED_FACES = 10      # landmarker capacity; only the k largest are processed
MIN_FACE_SIZE = 32           # px, faces with a smaller landmark box are ignored

# Tracking / asynchronous recognition
TRACK_MIN_IOU = 0.25
TRACK_MAX_MISSED = 5         # frames a track survives without a detection
RECOGNITION_REFRESH_SEC = 1.0  # live sources: re-recognise each face this often

# --------------------------------------------------------------------------- #
# GUI parameter defaults and ranges
# --------------------------------------------------------------------------- #
THRESHOLD_MIN = 0            # slider works in percent: 0..100  ->  0.00 .. 1.00
THRESHOLD_MAX = 100
MAX_FACES_MIN = 1
MAX_FACES_MAX = MAX_DETECTED_FACES

SVM_KERNELS = ("linear", "rbf")
SVM_GAMMAS = ("scale", "auto")
SVM_C_MIN = 0.01
SVM_C_MAX = 1000.0


@dataclass
class RecognitionParameters:
    """Live parameters that change how recognition results are interpreted."""
    threshold: float = 0.45      # cosine-similarity threshold (0..1)
    max_faces: int = 2           # k: how many of the largest faces are processed
    show_landmarks: bool = True


@dataclass
class SvmParameters:
    """Hyper-parameters forwarded to svm.train_svm()."""
    kernel: str = "linear"
    C: float = 10.0
    gamma: str = "scale"


DEFAULT_THEME = "dark"

# --------------------------------------------------------------------------- #
# Keyboard shortcuts
# --------------------------------------------------------------------------- #
SHORTCUT_PAUSE = "Space"
SHORTCUT_FULLSCREEN = "F11"
SHORTCUT_RESET = "R"
SHORTCUT_QUIT = "Q"

# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
THEMES = {
    "dark": {
        "window": "#1e2327", "panel": "#262c31", "panel_alt": "#2d353b",
        "border": "#3a444b", "text": "#e6e9eb", "text_dim": "#9aa5ad",
        "accent": "#3d9be9", "accent_text": "#ffffff", "view": "#101417",
        "ok": "#4cc38a", "warn": "#e8b04a", "bad": "#e5645c",
    },
    "light": {
        "window": "#f1f3f5", "panel": "#ffffff", "panel_alt": "#e9ecef",
        "border": "#ced4da", "text": "#212529", "text_dim": "#6c757d",
        "accent": "#1c7ed6", "accent_text": "#ffffff", "view": "#dee2e6",
        "ok": "#2b9a66", "warn": "#c98a1a", "bad": "#d6453d",
    },
}

# Annotation colours (R, G, B) - identical in both themes
COLOR_MATCH = (76, 195, 138)
COLOR_UNKNOWN = (229, 100, 92)
COLOR_PENDING = (232, 176, 74)
COLOR_LANDMARK = (61, 155, 233)
