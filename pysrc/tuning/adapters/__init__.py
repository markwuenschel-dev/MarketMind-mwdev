"""
Adapters for the tuning subsystem.

Adapters provide thin boundary-translation layers between the canonical engine
API (``pysrc.tuning._facade.tune``) and callers that carry sklearn estimators, raw
objective functions, or legacy YAML-format search spaces.

Modules:
    sklearn.py      — ``tune_estimator()`` for sklearn estimator + (X, y) data.
    objective.py    — ``tune_objective()`` for pure objective_fn callers.
    legacy_yaml.py  — ``parse_yaml_grid()`` for list-of-dicts YAML spaces.
"""

__all__: list[str] = []
