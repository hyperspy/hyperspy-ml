# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Linear Pipeline with contract validation and partial_fit dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Pipeline:
    """Linear sequence of ML stages with input/output type contracts.

    Each stage is a callable ``(input) -> output``. Contracts are
    optionally specified as tuples of accepted input types.

    Supports indexing, slicing, and ``partial_fit`` dispatch to the
    last stage that supports it.

    Parameters
    ----------
    stages : list[tuple[str, callable, tuple | None]]
        List of ``(name, stage, contract)`` tuples.
        *contract* is ``None`` (no check) or a tuple of accepted input types.
    """

    def __init__(
        self,
        stages: list[tuple[str, Callable, tuple[type, ...] | None]],
    ) -> None:
        self._stages = [(name, stage, contract) for name, stage, contract in stages]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Pipeline(self._stages[idx])
        name, stage, contract = self._stages[idx]
        return stage

    def __len__(self):
        return len(self._stages)

    def __iter__(self):
        for name, stage, _ in self._stages:
            yield name, stage

    @property
    def stage_names(self) -> list[str]:
        return [name for name, _, _ in self._stages]

    def run(self, initial_input: Any) -> Any:
        """Execute all stages in sequence, checking contracts.

        Parameters
        ----------
        initial_input : Any
            Input to the first stage.

        Returns
        -------
        Any
            Output of the last stage.

        Raises
        ------
        TypeError
            If any stage's contract check fails.
        """
        current = initial_input
        for name, stage, contract in self._stages:
            if contract is not None and not isinstance(current, contract):
                raise TypeError(
                    f"Stage '{name}' requires input type "
                    f"{contract}, got {type(current).__name__}"
                )
            current = stage(current)
        return current

    def partial_fit(self, data: Any) -> Any:
        """Dispatch partial_fit to the last stage that supports it.

        Parameters
        ----------
        data : Any
            Data chunk to stream through the incremental stage.

        Returns
        -------
        Any
            Stage output, or None if no stage supports partial_fit.

        Raises
        ------
        NotImplementedError
            If no stage has partial_fit.
        """
        for name, stage, _ in reversed(self._stages):
            if hasattr(stage, "partial_fit"):
                return stage.partial_fit(data)
        raise NotImplementedError("No stage in pipeline supports partial_fit")
