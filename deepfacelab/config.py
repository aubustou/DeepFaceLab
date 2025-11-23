from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass

logger = logging.getLogger(__name__)


CONFIG: Config

@dataclass(kw_only=True)
class Config:
    cpu_number: int = multiprocessing.cpu_count()
