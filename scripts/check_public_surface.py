"""Compare public class and function signatures in two Python source files."""
from __future__ import annotations

import ast
import sys


def _function_surface(node: ast.FunctionDef) -> dict:
    args = node.args
    return {
        "params": [
            (arg.arg, ast.unparse(arg.annotation) if arg.annotation else None)
            for arg in args.posonlyargs + args.args + args.kwonlyargs
        ],
        "defaults": [ast.unparse(default) for default in args.defaults]
        + [ast.unparse(default) if default else None for default in args.kw_defaults],
        "returns": ast.unparse(node.returns) if node.returns else None,
        "vararg": bool(args.vararg),
        "kwarg": bool(args.kwarg),
    }


def surface(path: str) -> dict:
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[f"<module>.{node.name}"] = "ClassDef"
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                    out[f"{node.name}.{method.name}"] = _function_surface(method)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            out[f"<module>.{node.name}"] = _function_surface(node)
    return out


before, after = surface(sys.argv[1]), surface(sys.argv[2])
missing = sorted(set(before) - set(after))
added = sorted(set(after) - set(before))
changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
print(f"public symbols before={len(before)} after={len(after)}")
print("REMOVED:", missing or "none")
print("ADDED:", added or "none")
print("CHANGED SIGNATURES:", changed or "none")
sys.exit(1 if missing or changed else 0)
