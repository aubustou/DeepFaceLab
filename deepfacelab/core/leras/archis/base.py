from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ArchiBase:

    def __init__(self, *args, name=None, **kwargs):
        self.name = name

    # overridable
    def flow(self, *args, **kwargs):
        raise Exception("this archi does not support flow. Use model classes directly.")

    # overridable
    def get_weights(self):
        pass
