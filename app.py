"""Vercel's Python runtime looks for `app` in app.py/index.py/main.py — api.py isn't
on that list, and pointing at it from pyproject.toml drags in uv, which then demands a
full [project] table and ignores requirements.txt. One re-export is cheaper."""

from api import app

__all__ = ["app"]
