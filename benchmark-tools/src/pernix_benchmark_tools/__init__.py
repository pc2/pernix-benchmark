"""Automation for the Pernix benchmark repository."""

from __future__ import annotations

from .cmake import CMakeProject, build_cmake_project

__version__ = "0.1.0"


__all__ = [
    "CMakeProject",
    "build_cmake_project",
]
