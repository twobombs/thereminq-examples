#!/usr/bin/env python3
"""
nighthawk_ghz.py -- moved into nighthawk_qrack.py as its `ghz` command:

  python3 nighthawk_qrack.py ghz tiles --register nighthawk61r2 --unit 3 --out ghz61.jsonl
  python3 nighthawk_qrack.py ghz -h

This stub keeps old command lines and imports working: run, it forwards its arguments to
`nighthawk_qrack.py ghz`; imported, it is nighthawk_qrack (every former nighthawk_ghz name is there,
main / make_parser / cmd_selftest as ghz_main / ghz_make_parser / cmd_ghz_selftest).
"""

import os
import sys
from pathlib import Path

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

_NQ = Path(__file__).resolve().parent / "nighthawk_qrack.py"

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(_NQ), "ghz"] + sys.argv[1:])
else:
    if str(_NQ.parent) not in sys.path:
        sys.path.insert(0, str(_NQ.parent))
    import nighthawk_qrack as _nq          # noqa: E402
    sys.modules[__name__] = _nq
