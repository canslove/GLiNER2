"""Diff two import-surface snapshots; expected module removals are passed in."""
from __future__ import annotations

import json
import sys


before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))
expected_removed = set(sys.argv[3:])

removed = set(before) - set(after)
added = set(after) - set(before)
failed = False

if removed - expected_removed:
    print("UNEXPECTED module removals:", sorted(removed - expected_removed))
    failed = True
if expected_removed - removed:
    print("expected removed but still present:", sorted(expected_removed - removed))
    failed = True
if added:
    print("new modules (informational):", sorted(added))

for module in sorted(removed & expected_removed):
    parent = module.rsplit(".", 1)[0]
    lost = {symbol for symbol in before[module] if symbol != "__all__"} - set(
        after.get(parent, {})
    )
    if lost:
        print(f"{module}: not re-exported by {parent}: {sorted(lost)}")
        failed = True

for module in sorted(set(before) & set(after)):
    for symbol, shape in before[module].items():
        if symbol not in after[module]:
            print(f"{module}.{symbol}: GONE")
            failed = True
        elif after[module][symbol] != shape:
            print(f"{module}.{symbol}: CHANGED {shape} -> {after[module][symbol]}")
            failed = True

print("FAIL" if failed else "OK")
sys.exit(1 if failed else 0)
