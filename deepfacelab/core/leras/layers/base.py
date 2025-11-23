
from __future__ import annotations
import logging
from .saveable import Saveable

logger = logging.getLogger(__name__)

class LayerBase(Saveable):
    # override
    def build_weights(self):
        pass

    # override
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


