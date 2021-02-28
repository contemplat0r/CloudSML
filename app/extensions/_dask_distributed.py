# encoding: utf-8
"""
Dask Distributed adapter
========================
"""

import dask.distributed


class DaskDistributed(object):

    def __init__(self):
        self.client = None

    def init_app(self, app):
        # pylint: disable=unused-argument
        self.client = dask.distributed.Client()
