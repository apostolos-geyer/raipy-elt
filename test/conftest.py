import sys
from pathlib import Path

test_dir = Path(__file__).parent
proj_dir = test_dir.parent
src_dir = proj_dir / "src"

sys.path.insert(0, str(src_dir))

print(f"inserted {src_dir} to path")
