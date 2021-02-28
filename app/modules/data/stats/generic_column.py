# encoding: utf-8

import numpy

from cloudsml_computational_backend_common.data.consts import (
    DataFormats,
    DataTypes,
)

from . import constants


def detect_column_type(uniques_count):
    """
    Detect column type - categorical or continuous
    """
    # TODO Add Binary?

    if uniques_count >= constants.UNIQUES_LIMIT:
        return DataTypes.continuous
    else:
        return DataTypes.categorical


def detect_and_cast_column_format(column):
    """
    Detect column format (numerical or text) and cast to
    detected format.
    """
    if numpy.issubdtype(column.dtype, numpy.number):
        return column, DataFormats.numerical
    try:
        column = column.astype(numpy.float32)
    except ValueError:
        pass
    else:
        return column, DataFormats.numerical

    return column, DataFormats.character
