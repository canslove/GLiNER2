#!/usr/bin/env python3
"""Snapshot and compare a package's source-level Python API without importing it."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, Optional, Union


def _annotation(node: ast.arg) -> Optional[str]:
    return ast.unparse(node.annotation) if node.annotation else None


def _signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[str, Any]:
    args = node.args
    return {
        "posonly": [(arg.arg, _annotation(arg)) for arg in args.posonlyargs],
        "args": [(arg.arg, _annotation(arg)) for arg in args.args],
        "kwonly": [(arg.arg, _annotation(arg)) for arg in args.kwonlyargs],
        "defaults": [ast.unparse(value) for value in args.defaults],
        "kw_defaults": [
            ast.unparse(value) if value else None for value in args.kw_defaults
        ],
        "vararg": args.vararg.arg if args.vararg else None,
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "returns": ast.unparse(node.returns) if node.returns else None,
        "decorators": [ast.unparse(value) for value in node.decorator_list],
        "is_async": isinstance(node, ast.AsyncFunctionDef),
    }


def _dunder_all(tree: ast.Module):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in node.targets):
            continue
        try:
            return list(ast.literal_eval(node.value))
        except (ValueError, TypeError):
            return "<non-literal>"
    return None


def snapshot_module(path: str, module: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    out: Dict[str, Any] = {
        "module": module,
        "__all__": _dunder_all(tree),
        "classes": {},
        "functions": {},
        "assignments": [],
        "imports": [],
        "dataclass_fields": {},
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out["functions"][node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            entry = {
                "bases": [ast.unparse(base) for base in node.bases],
                "decorators": [ast.unparse(value) for value in node.decorator_list],
                "methods": {},
                "class_attrs": [],
            }
            fields = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entry["methods"][member.name] = _signature(member)
                elif isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    fields.append((
                        member.target.id,
                        ast.unparse(member.annotation),
                        ast.unparse(member.value) if member.value else None,
                    ))
                elif isinstance(member, ast.Assign):
                    entry["class_attrs"].extend(
                        target.id
                        for target in member.targets
                        if isinstance(target, ast.Name)
                    )
            out["classes"][node.name] = entry
            if fields:
                out["dataclass_fields"][node.name] = fields
        elif isinstance(node, ast.Assign):
            out["assignments"].extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id != "__all__"
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out["assignments"].append(node.target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            source = "." * node.level + (node.module or "")
            out["imports"].extend(
                f"from {source} import {alias.name}" for alias in node.names
            )
        elif isinstance(node, ast.Import):
            out["imports"].extend(f"import {alias.name}" for alias in node.names)
    out["imports"] = sorted(set(out["imports"]))
    out["assignments"] = sorted(set(out["assignments"]))
    return out


def snapshot(root: str) -> Dict[str, Any]:
    root = root.rstrip("/")
    package = os.path.basename(root)
    modules = {}
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in {"__pycache__", ".git", "tests", "test"}
        ]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(directory, filename)
            parts = os.path.relpath(path, root)[:-3].split(os.sep)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join([package] + parts)
            modules[module] = snapshot_module(path, module)
    return {"root": package, "module_count": len(modules), "modules": modules}


def _flatten(data: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for module, details in data["modules"].items():
        if details["__all__"] not in (None, "<non-literal>"):
            for name in details["__all__"]:
                flat[f"{module}::__all__::{name}"] = "exported"
        for name, signature in details["functions"].items():
            flat[f"{module}.{name}"] = signature
        for name, cls in details["classes"].items():
            flat[f"{module}.{name}"] = {
                "bases": cls["bases"],
                "decorators": cls["decorators"],
            }
            for method, signature in cls["methods"].items():
                flat[f"{module}.{name}.{method}"] = signature
        for name, fields in details["dataclass_fields"].items():
            flat[f"{module}.{name}::fields"] = fields
        for name in details["assignments"]:
            flat[f"{module}.{name}"] = "assignment"
    return flat


def _public(key: str) -> bool:
    return not any(
        part.startswith("_") and not part.startswith("__")
        for part in key.split("::", 1)[0].split(".")
    )


def _tail(key: str) -> str:
    """Return the caller-visible symbol suffix, excluding its module path."""
    head, separator, marker = key.partition("::")
    parts = head.split(".")
    for index, part in enumerate(parts):
        if part[:1].isupper():
            suffix = ".".join(parts[index:])
            break
    else:
        suffix = parts[-1]
    return suffix + (separator + marker if separator else "")


def compare(
    before: Dict[str, Any],
    after: Dict[str, Any],
    strict: bool,
) -> int:
    old, new = _flatten(before), _flatten(after)
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])

    added_by_shape = {}
    for key in added:
        shape = (_tail(key), json.dumps(new[key], sort_keys=True))
        added_by_shape.setdefault(shape, []).append(key)
    moved = []
    truly_removed = []
    used_destinations = set()
    for key in removed:
        shape = (_tail(key), json.dumps(old[key], sort_keys=True))
        destinations = added_by_shape.get(shape, [])
        destination = next(
            (candidate for candidate in destinations if candidate not in used_destinations),
            None,
        )
        if destination is None:
            truly_removed.append(key)
        else:
            moved.append((key, destination))
            used_destinations.add(destination)
    removed = truly_removed
    added = [key for key in added if key not in used_destinations]

    public_removed = [key for key in removed if _public(key)]
    public_changed = [key for key in changed if _public(key)]
    private_removed = [key for key in removed if not _public(key)]

    print(f"modules: {before['module_count']} -> {after['module_count']}")
    print(f"surface entries: {len(old)} -> {len(new)}")
    if moved:
        print("MOVED (same signature):")
        for source, destination in moved:
            print(f"  {source} -> {destination}")
    print("REMOVED (public):", public_removed or "none")
    print("CHANGED SIGNATURE (public):", public_changed or "none")
    print(f"ADDED: {len(added)}; REMOVED (private): {len(private_removed)}")
    failed = bool(public_removed or public_changed)
    if strict:
        failed = failed or bool(added or private_removed)
    print("RESULT:", "FAIL" if failed else "OK")
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snap")
    snap.add_argument("root")
    diff = commands.add_parser("diff")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.command == "snap":
        json.dump(snapshot(args.root), sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    with open(args.before) as before_handle, open(args.after) as after_handle:
        return compare(
            json.load(before_handle),
            json.load(after_handle),
            args.strict,
        )


if __name__ == "__main__":
    raise SystemExit(main())
