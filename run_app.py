"""Start the ClauseScope Streamlit UI with project-local packages if present."""
from __future__ import annotations

import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parent
local_packages = root / ".runtime"
if local_packages.exists():
    sys.path.insert(0, str(local_packages))

try:
    import numpy as np
    try:
        import numpy.random  # noqa: F401
    except (ImportError, OSError):
        np.random = types.ModuleType("numpy.random")
        np.random.Generator = object
        np.random.BitGenerator = object
        np.random.RandomState = object
        sys.modules["numpy.random"] = np.random
except ImportError:
    pass

from streamlit.web.cli import main

sys.argv = ["streamlit", "run", str(root / "app.py"), *sys.argv[1:]]
raise SystemExit(main())
