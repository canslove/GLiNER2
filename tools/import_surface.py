"""Snapshot the public import surface of a package for refactor verification."""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys


def surface(pkg_name: str = "gliner2") -> dict:
    pkg = importlib.import_module(pkg_name)
    names = [pkg_name] + [
        module.name for module in pkgutil.walk_packages(pkg.__path__, pkg_name + ".")
    ]
    out: dict = {}
    for name in sorted(names):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"__error__": f"{type(exc).__name__}: {exc}"}
            continue
        entry: dict = {}
        for attr in sorted(attr for attr in dir(mod) if not attr.startswith("_")):
            obj = getattr(mod, attr)
            if inspect.isclass(obj):
                entry[attr] = ["class"] + sorted(
                    member for member in dir(obj) if not member.startswith("_")
                )
            elif inspect.isfunction(obj):
                try:
                    entry[attr] = ["func", str(inspect.signature(obj))]
                except (ValueError, TypeError):
                    entry[attr] = ["func", "?"]
            else:
                entry[attr] = [type(obj).__name__]
        entry["__all__"] = sorted(getattr(mod, "__all__", []) or [])
        out[name] = entry
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as handle:
        json.dump(surface(), handle, indent=1, sort_keys=True)
