"""
Utility functions for loading and validating the settings for a global fit run.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

def find_class(name: str, modules: tuple[str, ...]) -> type:
     for module in modules:
         cls = getattr(importlib.import_module(module), name, None)
         if isinstance(cls, type):
             return cls
     raise ValueError(f"no class {name!r} in {list(modules)}.")

def accepted_kwargs(target: type) -> set[str] | None:
    """Keyword names ``target(...)`` accepts, following ``**kwargs`` up the MRO for classes; ``None`` = anything."""
    if not isinstance(target, type):
        params = inspect.signature(target).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return None
        return {n for n, p in params.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}

    names: set[str] = set()
    for klass in target.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if klass is object:
            # **kwargs passed all the way down: object takes none, and a base that swallows them
            # (eryn's Move) would silently ignore a misspelled name
            return names - {"self"}
        params = inspect.signature(init).parameters
        names |= {n for n, p in params.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return names - {"self"}
    return None

def check_kwargs(given: dict, target: type | Any, engine_filled: frozenset[str]) -> None:
    names = set(given)
    if names & engine_filled:
        raise ValueError(f"{sorted(names & engine_filled)} are filled in by the builder; remove them.")
    accepted = accepted_kwargs(target)
    name = (
            target.__self__.__name__
            if hasattr(target, "__self__") and isinstance(target.__self__, type)
            else getattr(target, "__name__", str(target))
        )
    if accepted is not None and names - accepted:
        raise ValueError(f"{name} does not accept {sorted(names - accepted)}.")

def prior_class(name: str) -> type:
    from ereborpriors.base import GFBasePrior

    module, _, clsname = name.rpartition(".")
    try:
        cls = find_class(clsname, (f"ereborpriors.stock.{module}",)) if module else None
    except ImportError as err:  # pydantic only turns ValueError into a validation error
        raise ValueError(f"no module ereborpriors.stock.{module}.") from err

    if cls is None or not issubclass(cls, GFBasePrior):
        raise ValueError(f"{name!r} is not a GFBasePrior subclass in ereborpriors.stock (use <module>.<Class>).")

    return cls