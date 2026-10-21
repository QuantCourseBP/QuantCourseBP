import sys
from pathlib import Path

# The library modules import the enums as 'src.enums', so the QuantCourseBP root folder has to be importable.
# This makes 'pytest' work from any working directory, not only from the repository root.
sys.path.append(str(Path(__file__).resolve().parents[3]))
