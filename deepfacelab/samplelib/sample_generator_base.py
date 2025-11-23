from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
"""
You can implement your own SampleGenerator
"""


class SampleGeneratorBase:

    def __init__(self, debug=False, batch_size=1):
        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size
        self.last_generation = None
        self.active = True

    def set_active(self, is_active):
        self.active = is_active

    def generate_next(self):
        if not self.active and self.last_generation is not None:
            return self.last_generation
        self.last_generation = next(self)
        return self.last_generation

    # overridable
    def __iter__(self):
        # implement your own iterator
        return self

    def __next__(self):
        # implement your own iterator
        return None

    # overridable
    def is_initialized(self):
        return True
