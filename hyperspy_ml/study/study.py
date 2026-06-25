# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Study container — dict-like result collection with events and disk backing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_AUTO_NAMES: dict[str, int] = {}


class Study:
    """Dict-like container for result objects in a hyperspy-ml workflow.

    Supports auto-naming, PrettyTable summary, dependency-aware removal,
    disk backing via Zarr, and rerun with parameter overrides.

    Parameters
    ----------
    name : str, optional
        Human-readable study name.
    """

    def __init__(self, name: str = "study") -> None:
        self.name = name
        self._results: dict[str, Any] = {}
        self._sources: dict[str, Any] = {}
        self._params: dict[str, dict[str, Any]] = {}
        self.events = _StudyEvents()

    def __setitem__(self, key: str, value: Any) -> None:
        self._results[key] = value
        self.events._notify("result_added", key)

    def __getitem__(self, key: str) -> Any:
        return self._results[key]

    def __delitem__(self, key: str) -> None:
        del self._results[key]
        self._sources.pop(key, None)
        self._params.pop(key, None)
        self.events._notify("result_removed", key)

    def __contains__(self, key: str) -> bool:
        return key in self._results

    def __iter__(self):
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return self.summary()

    def keys(self):
        return self._results.keys()

    def values(self):
        return self._results.values()

    def items(self):
        return self._results.items()

    def get(self, key, default=None):
        return self._results.get(key, default)

    def add(
        self,
        result,
        name: str | None = None,
        source_signal=None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Add a result with auto-generated or explicit name.

        Returns the key used.
        """
        if name is None:
            prefix = type(result).__name__
            count = _AUTO_NAMES.get(prefix, 0) + 1
            _AUTO_NAMES[prefix] = count
            name = f"{prefix}_{count}"
        self._results[name] = result
        if source_signal is not None:
            self._sources[name] = source_signal
        if params is not None:
            self._params[name] = params
        self.events._notify("result_added", name)
        return name

    def remove(self, name: str) -> None:
        """Remove a result and any dependents that reference it."""
        # Simple dependency: remove keys that contain this name as prefix
        dependent = [k for k in self._results if k.startswith(name) and k != name]
        self._results.pop(name, None)
        self._sources.pop(name, None)
        self._params.pop(name, None)
        for dep in dependent:
            self._results.pop(dep, None)
            self._sources.pop(dep, None)
            self._params.pop(dep, None)
        self.events._notify("result_removed", name)

    def summary(self) -> str:
        """PrettyTable summary of the study contents."""
        lines = [f"Study: {self.name}", "-" * 40]
        for key, result in self._results.items():
            rtype = type(result).__name__
            lines.append(f"  {key:30s} {rtype}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save entire study to disk as a Zarr store."""
        import zarr

        from hyperspy_ml.results.io import save_result

        store = zarr.open(str(path), mode="w")
        store.attrs["type"] = "Study"
        store.attrs["name"] = self.name
        store.attrs["keys"] = list(self._results.keys())

        for key, result in self._results.items():
            grp = store.require_group(key)
            grp.attrs["result_type"] = type(result).__name__
            # Save result inline
            save_result(
                result, str(Path(path) / key), source_signal=self._sources.get(key)
            )

    @classmethod
    def load(cls, path: str | Path) -> Study:
        """Load a study from a Zarr store."""
        import zarr

        from hyperspy_ml.results.io import load_result

        store = zarr.open(str(path), mode="r")
        study = cls(name=store.attrs.get("name", "study"))
        for key in store.attrs.get("keys", []):
            sub = str(Path(path) / key)
            try:
                study[key] = load_result(sub)
            except Exception:
                _logger.warning("Failed to load result, skipping", exc_info=True)
        return study


class _StudyEvents:
    def __init__(self):
        self._callbacks: dict[str, list] = {}

    def connect(self, event: str, callback):
        self._callbacks.setdefault(event, []).append(callback)

    def disconnect(self, event: str, callback):
        self._callbacks.get(event, []).remove(callback)

    def _notify(self, event: str, *args):
        for cb in self._callbacks.get(event, []):
            cb(*args)
