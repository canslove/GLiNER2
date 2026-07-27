"""Phase 6: 500-seed differential test - the exact decoder's utility equals the
brute-force optimum over every feasible assignment (or both are infeasible).

This is the proof of exactness. Must complete well under 60s.
"""
from __future__ import annotations

import math
import random
from itertools import combinations, product
from types import SimpleNamespace

from gliner2.classification import constraints as C
from gliner2.classification.compiler import compile_schema
from gliner2.classification.candidates import task_utilities
from gliner2.classification.constraints import DictAssignment
from gliner2.classification.decoding import build_problem
from gliner2.classification.decoding.exact import ExactDecoder
from gliner2.classification.errors import SchemaError
from gliner2.classification.schema import ClassificationSchema
from gliner2.classification.scoring import ClassificationScores


def _cfg():
    return SimpleNamespace(decoder="exact", exact_node_budget=10_000_000,
                           beam_size=32, candidate_threshold=0.0,
                           max_candidates_per_task=64, on_infeasible="raise")


def _cardinality_subsets(labels, spec):
    minl = spec.min_labels
    maxl = spec.effective_max_labels()
    for size in range(len(labels) + 1):
        if not (minl <= size <= maxl):
            continue
        for combo in combinations(labels, size):
            yield frozenset(combo)


def _brute_force(compiled, scores):
    tasks = compiled.task_order
    per_task = []
    utils = {}
    for t in tasks:
        spec = compiled.task(t)
        per_task.append(list(_cardinality_subsets(spec.label_names, spec)))
        utils[t] = task_utilities(spec, dict(scores.tasks[t]))
    best_score = -math.inf
    feasible_found = False
    for combo in product(*per_task):
        chosen = {t: combo[i] for i, t in enumerate(tasks)}
        a = DictAssignment(compiled, chosen, decided=tasks)
        if all(c.evaluate(a) is True for c in compiled.constraints):
            feasible_found = True
            score = sum(utils[t][l] for t in tasks for l in chosen[t])
            best_score = max(best_score, score)
    return feasible_found, best_score


def _random_schema(rng):
    schema = ClassificationSchema()
    n_tasks = rng.randint(2, 3)
    task_names = [f"t{i}" for i in range(n_tasks)]
    for name in task_names:
        n_labels = rng.randint(2, 4)
        labels = [f"{name}_{j}" for j in range(n_labels)]
        kind = rng.choice(["single", "multi", "multi"])
        if kind == "single":
            schema.single(name, labels)
        else:
            maxl = rng.randint(1, n_labels)
            minl = rng.randint(0, maxl)
            schema.multi(name, labels, min_labels=minl, max_labels=maxl)
    # a couple of random cross-task constraints
    for _ in range(rng.randint(0, 3)):
        kind = rng.choice(["implies", "excludes", "iff"])
        ta, tb = rng.sample(task_names, 2)
        la = rng.choice(schema.task_spec(ta).label_names)
        lb = rng.choice(schema.task_spec(tb).label_names)
        try:
            if kind == "implies":
                schema.constrain(C.implies((ta, la), (tb, lb)))
            elif kind == "excludes":
                schema.constrain(C.excludes((ta, la), (tb, lb)))
            else:
                schema.constrain(C.iff((ta, la), (tb, lb)))
        except SchemaError:
            pass
    return schema


def test_exact_matches_brute_force_over_500_seeds():
    cfg = _cfg()
    checked = 0
    for seed in range(500):
        rng = random.Random(seed)
        schema = _random_schema(rng)
        try:
            compiled = compile_schema(schema)
        except SchemaError:
            continue  # statically infeasible; skip
        tasks = {t: {l: round(rng.uniform(-3, 3), 3)
                     for l in compiled.task(t).label_names}
                 for t in compiled.task_order}
        scores = ClassificationScores(
            text="x", tasks=tasks, fingerprint=compiled.fingerprint,
            specs={s.name: s for s in compiled.task_specs})

        problem = build_problem(compiled, scores, cfg)
        sol = ExactDecoder().decode(problem, budget=cfg.exact_node_budget)

        feasible, brute_score = _brute_force(compiled, scores)
        if not feasible:
            assert sol is None, f"seed {seed}: exact found a solution but brute did not"
        else:
            assert sol is not None, f"seed {seed}: exact missed a feasible optimum"
            assert math.isclose(sol.score, brute_score, rel_tol=1e-9, abs_tol=1e-9), (
                f"seed {seed}: exact {sol.score} != brute {brute_score}")
        checked += 1
    assert checked > 400  # the vast majority of seeds are valid schemas
