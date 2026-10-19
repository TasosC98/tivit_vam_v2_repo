"""Every module must at least import: a syntax error in a module that no other
test touches (evaluate.py) once made every evaluation on the server fail."""
import importlib
import pkgutil

import pianovam_vision


def test_all_modules_import():
    for m in pkgutil.iter_modules(pianovam_vision.__path__):
        importlib.import_module(f"pianovam_vision.{m.name}")
