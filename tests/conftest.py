import sys
from pathlib import Path

# Make ``tests/helpers.py`` importable as ``helpers`` from every test module.
sys.path.insert(0, str(Path(__file__).parent))
