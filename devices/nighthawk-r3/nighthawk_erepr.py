#!/usr/bin/env python3
"""
nighthawk_erepr.py -- moved into nighthawk_qrack.py as its `erepr` command:

  python3 nighthawk_qrack.py erepr selftest
  python3 nighthawk_qrack.py erepr -h

This stub keeps old command lines and imports working: run, it forwards its arguments to
`nighthawk_qrack.py erepr`; imported, it is nighthawk_qrack (the names that clashed there were renamed,
see the README's rename table).
"""

import os
import sys
from pathlib import Path

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

_NQ = Path(__file__).resolve().parent / "nighthawk_qrack.py"

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(_NQ), "erepr"] + sys.argv[1:])
else:
    if str(_NQ.parent) not in sys.path:
        sys.path.insert(0, str(_NQ.parent))
    import nighthawk_qrack as _nq          # noqa: E402
    sys.modules[__name__] = _nq
