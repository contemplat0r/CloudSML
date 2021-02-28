import pytest
import pandas

from cloudsml_computational_backend_common.data.consts import DataTypes, DataFormats

from app.modules.data.stats import generic_column
from app.modules.data.stats import constants


def test_detect_and_cast_column_format():
    column = pandas.Series([b'0', b'1', b'2', b'1'], dtype='S8')
    column, column_format = generic_column.detect_and_cast_column_format(column)
    assert column_format is DataFormats.numerical
    assert column.loc[0] == 0

    column = pandas.Series([b'0.8', b'1', b'2.0', b'0'], dtype='S8')
    column, column_format = generic_column.detect_and_cast_column_format(column)
    assert column_format is DataFormats.numerical
    assert column.loc[1] == 1.0

    column = pandas.Series([b's', b'1', b'2.0', b''], dtype='S8')
    column, column_format = generic_column.detect_and_cast_column_format(column)
    assert column_format is DataFormats.character
    assert column.loc[0] == b's'


def tests_detect_column_type_continuous():
    assert generic_column.detect_column_type(1000) is DataTypes.continuous


def tests_detect_column_type_categorical():
    assert generic_column.detect_column_type(100) is DataTypes.categorical
