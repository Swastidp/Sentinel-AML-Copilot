"""Smoke test for the package skeleton.

Cheap, but it catches the two things that actually break at this stage: a `src`
layout that was never installed (`pip install -e .` skipped), and a subpackage
directory missing its `__init__.py`.
"""

from __future__ import annotations

import importlib

import sentinel

EXPECTED_SUBPACKAGES = [
    "sentinel.config",
    "sentinel.domain",
    "sentinel.domain.ports",
    "sentinel.application",
    "sentinel.infrastructure",
    "sentinel.pipeline",
    "sentinel.agent",
    "sentinel.api",
    "sentinel.ui",
    "sentinel.reporting",
]


def test_package_is_importable_and_versioned():
    assert sentinel.__version__ == "0.1.0"


def test_every_layer_is_an_importable_package():
    for name in EXPECTED_SUBPACKAGES:
        assert importlib.import_module(name) is not None, name
