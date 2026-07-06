"""cuxray — hardware-free static analyzer for CUDA kernel binaries.

Public API:
    build_report(path, toolchain, ...)   full analysis → schema/1 dict
    diff_reports(old, new)               compare two report dicts
    parse_gate / eval_gate               CI gate expressions
    resolve()                            locate/fetch the CUDA binary utilities
    lookup(arch) / compute(...)          occupancy engine
    analyze_accesses(func, block_dims)   Layer B access analysis
"""

__version__ = "0.3.1"

SCHEMA_VERSION = "cuxray.schema/1"

from .analyze.access import analyze_accesses          # noqa: E402,F401
from .archspec import lookup                          # noqa: E402,F401
from .diffgate import diff_reports, eval_gate, parse_gate  # noqa: E402,F401
from .occupancy import compute                        # noqa: E402,F401
from .report import build_report                      # noqa: E402,F401
from .toolchain import resolve                        # noqa: E402,F401
