"""Every module and script must at least import / compile: a syntax error in a file that no
other test touches would otherwise only show up when that step is run."""
import importlib
import pkgutil
import py_compile
from pathlib import Path

import pianovam_vision


def test_all_modules_import():
    for m in pkgutil.iter_modules(pianovam_vision.__path__):
        importlib.import_module(f"pianovam_vision.{m.name}")


def test_paper_scripts_compile():
    for f in sorted((Path(__file__).resolve().parent.parent / "paper").glob("*.py")):
        py_compile.compile(str(f), doraise=True)
