# encoding: utf-8
"""
Helper JSON utils
-----------------

Example:

>>> from app.utils import json
>>> dumps(pandas.Series([u'1', numpy.int64(2), b'3.0', numpy.float32(4)], dtype='object').items())
'[[0, "1"], [1, "2"], [2, "3.0"], [3, "4.0"]]'

"""

import functools
import json

import numpy
import pandas


class StreamArray(list):
    """
    A helper to wrap any iterable to make it JSON-serializable.

    It is based on this snippet: http://stackoverflow.com/a/24033219/1178806
    """

    def __init__(self, a_list):
        # pylint: disable=super-init-not-called
        self._a_list = a_list

    def __iter__(self):
        return self._a_list

    def __len__(self):
        return 1


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder dealing with numpy types and iterators.
    """

    def default(self, obj):
        # pylint: disable=method-hidden
        if isinstance(obj, zip):
            return StreamArray(obj)
        if isinstance(obj, (set, numpy.ndarray)):
            return StreamArray(iter(obj))
        if isinstance(obj, numpy.generic):
            return obj.item()
        if isinstance(obj, pandas.Series):
            return StreamArray(obj.items())
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super(NumpyAwareJSONEncoder, self).default(obj)


@functools.wraps(json.dumps)
def dumps(*args, **kwargs):
    """
    This is a customized JSON dumps implementation enabling numpy aware
    encoder (serializer) by default.
    """
    if 'cls' not in kwargs:
        kwargs['cls'] = NumpyAwareJSONEncoder
    if 'allow_nan' not in kwargs:
        # Forbid NaNs serialization as they are not valid JSON
        kwargs['allow_nan'] = False
    if 'separators' not in kwargs:
        # Reduce the size of JSON output by avoiding extra space after ":"
        kwargs['separators'] = (',', ':')
    return json.dumps(*args, **kwargs)


loads = json.loads  # pylint: disable=invalid-name
