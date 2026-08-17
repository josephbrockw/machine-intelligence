"""Shared helpers for the whole learning repo.

Anything in here is importable from any notebook in any resource directory
(`from mi.gpu import check_gpu_availability`), because the root project is
installed into .venv in editable mode. Helpers used by only one resource's
notebooks are better off sitting next to those notebooks instead.
"""
