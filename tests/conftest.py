import sys
from pathlib import Path

# Make the repo root importable (infidictionary package, top-level scripts).
sys.path.insert(0, str(Path(__file__).parent.parent))
