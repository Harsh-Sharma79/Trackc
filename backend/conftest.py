"""
conftest.py – Add project root to sys.path so that `import src.*` works
from any working directory when running pytest.
"""
import sys
from pathlib import Path

# Project root is one level above this file (speech_eval/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
