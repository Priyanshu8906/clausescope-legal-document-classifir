"""Start the ClauseScope Streamlit UI with project-local packages if present."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
local_packages = root / ".runtime"
if local_packages.exists():
    sys.path.insert(0, str(local_packages))

from streamlit.web.cli import main

sys.argv = ["streamlit", "run", str(root / "app.py"), *sys.argv[1:]]
raise SystemExit(main())
