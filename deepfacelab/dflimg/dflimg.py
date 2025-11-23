from __future__ import annotations

import logging

from .dfljpg import DFLJPG

logger = logging.getLogger(__name__)


def load(filepath, loader_func=None):
    if filepath.suffix == ".jpg":
        return DFLJPG.load(str(filepath), loader_func=loader_func)
    else:
        return None
