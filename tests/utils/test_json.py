import pytest

import numpy
import pandas

from app.utils import json


@pytest.mark.parametrize(
        'input_object, expected_output',
        (
            ([1], '[1]'),
            ([1.0], '[1.0]'),
            ({'a': 1}, '{"a":1}'),
            ({1, 2, 3}, '[1,2,3]'),
            ([numpy.int64(1)], '[1]'),
            ([numpy.int32(1)], '[1]'),
            ([numpy.int16(1)], '[1]'),
            ([numpy.uint64(1)], '[1]'),
            ([numpy.uint32(1)], '[1]'),
            ([numpy.uint16(1)], '[1]'),
            ([numpy.float64(1)], '[1.0]'),
            ([numpy.float32(1)], '[1.0]'),
            ([numpy.float16(1)], '[1.0]'),
            ([numpy.bool(True)], '[true]'),
            ([numpy.str('str')], '["str"]'),
            ([numpy.str_('str')], '["str"]'),
            (numpy.array([1, 2]), '[1,2]'),
            (numpy.array([[1, 2], [3, 4]]), '[[1,2],[3,4]]'),
            (pandas.Series([1]).items(), '[[0,1]]'),
            (pandas.Series([1.0]).items(), '[[0,1.0]]'),
            (pandas.Series([1.0]), '[[0,1.0]]'),
        )
    )
def test_serialization(input_object, expected_output):
    assert json.dumps(input_object) == expected_output
