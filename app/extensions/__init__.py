# encoding: utf-8
# pylint: disable=invalid-name,wrong-import-position
"""
Extensions setup
================

Extensions provide access to common resources of the application.

Please, put new extension instantiations and initializations here.
"""

from ._dask_distributed import DaskDistributed
dask_distributed = DaskDistributed()

from ._seaweedfs import SeaweedFS
seaweedfs = SeaweedFS()

from ._cloudsml import CloudSML
cloudsml = CloudSML()


def init_app(app):
    """
    Initiates extensions by calling ``init_app()`` on each of them
    """
    for extensions in (
            dask_distributed,
            seaweedfs,
            cloudsml,
        ):
        extensions.init_app(app)
