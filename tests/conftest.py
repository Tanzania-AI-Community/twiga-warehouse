import os
import sys
from pathlib import Path


os.environ.setdefault("INPUT_BOOKS_PATH", "/tmp")
os.environ.setdefault("OUTPUT_BOOKS_PATH", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
