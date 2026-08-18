"""Authoritative fixture loaders for ADR-007 hashing vectors.

The source of truth for golden vectors is language-neutral data under
``tests/golden/adr007``: ``manifest.json`` plus raw ``.bin``/``.json`` case
files. Python helpers exist only to load those shared fixtures; they are not the
canonical representation.
"""

from .golden_loader import (
    GoldenCase,
    GoldenManifest,
    GoldenSuite,
    GoldenVectorError,
    iter_suite_cases,
    load_case_bytes,
    load_manifest,
    load_suite,
)

__all__ = [
    "GoldenCase",
    "GoldenManifest",
    "GoldenSuite",
    "GoldenVectorError",
    "iter_suite_cases",
    "load_case_bytes",
    "load_manifest",
    "load_suite",
]
