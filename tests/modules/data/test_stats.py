# encoding: utf-8
from collections import namedtuple
import mock
import string

import pytest

import dask.dataframe
import numpy
import pandas

from cloudsml_computational_backend_common.data.consts import DataTypes, DataFormats

from app.extensions import cloudsml
from app.modules.data import stats


def test_get_column_statistics():
    column = pandas.Series([float('NaN'), '1', '2.0', float('NaN')])
    stats_values = stats.get_column_statistics(column)
    assert set(stats_values.keys()) == {
            'missing_values_count',
            'format',
            'format_specific_statistics',
            'type'
        } or set(stats_values.keys()) == {
            'missing_values_count',
            'format',
            'format_specific_statistics',
            'type',
            'uniques_count',
            'uniques_stats'
        }

    column = pandas.Series(['a', 'cd', 'etx', 'a', '22', '22', 'n'])
    stats_values = stats.get_column_statistics(column)
    assert set(stats_values.keys()) == {
            'missing_values_count',
            'format',
            'format_specific_statistics',
            'type'
        } or set(stats_values.keys()) == {
            'missing_values_count',
            'format',
            'format_specific_statistics',
            'type',
            'uniques_count',
            'uniques_stats'
        }


def test_get_column_statistics_check_missings():
    column = pandas.Series([float('NaN'), '1', '2.0', float('NaN')])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['missing_values_count'] == 2

    column = pandas.Series(['0', '1', '2.0', '-1'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['missing_values_count'] == 0


def test_get_column_statistics_check_format():
    column = pandas.Series(['0', '1', '2', '-1'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format'] is DataFormats.numerical

    column = pandas.Series(['0.8', '1', '2.0', '0'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format'] is DataFormats.numerical

    column = pandas.Series(['s', '1', '2.0', float('NaN')])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format'] is DataFormats.character


def test_get_column_statistics_detect_continuous():
    column = pandas.Series(numpy.arange(1002).astype(str))
    stats_values = stats.get_column_statistics(column)
    assert stats_values['type'] is DataTypes.continuous


def test_get_column_statistics_detect_categorical():
    column = pandas.Series(['a', 'cd', 'etx', 'a', '22', '22', 'n'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['type'] is DataTypes.categorical
    assert stats_values['uniques_count'] == 5


def test_get_column_statistics_numeric_specific_statistics_min_max():
    column = pandas.Series(['0.8', '1', '2', '-1'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format_specific_statistics']['min'] == -1
    assert stats_values['format_specific_statistics']['max'] == 2


def test_get_column_statistics_numeric_specific_statistics_sum():
    column = pandas.Series(['0.8', '1', '2', '-1'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format_specific_statistics']['sum'] == numpy.float32(2.8)


def test_get_column_statistics_text_specific_statistics_min_len_max_len():
    column = pandas.Series(['a', float('NaN'), 'etx09', 'a', '22', '22', 'n'])
    stats_values = stats.get_column_statistics(column)
    assert stats_values['format_specific_statistics']['min_len'] == 1
    assert stats_values['format_specific_statistics']['max_len'] == 5


def test_extract_pandas_dataframe_info():
    dataset_df = pandas.DataFrame(
            {
                'A': ['1', '2.0', float('NaN'), '2.0', '7', '0.7', '1'],
                'B': ['1', float('NaN'), float('NaN'), '1', float('NaN'), float('NaN'), '3'],
                'C': [float('NaN'), 't', '1', 'Hi!', float('NaN'), 'Pandas', 'x3'],
            }
        )

    pandas_dataframe_info = stats.extract_pandas_dataframe_info(dataset_df)
    uniques_stats = {}
    per_column_statistic = {}
    for key, value in pandas_dataframe_info['per_column_statistic'].items():
        uniques_stats[key] = value.pop('uniques_stats').to_dict()
        per_column_statistic[key] = value
    assert pandas_dataframe_info['rows_count'] == 7
    assert per_column_statistic == {
            'C': {
                    'format': DataFormats.character,
                    'format_specific_statistics': {
                            'max_len': 6,
                            'min_len': 1
                        },
                    'type': DataTypes.categorical,
                    'uniques_count': 5,
                    'missing_values_count': 2,
                },
            'A': {
                    'format': DataFormats.numerical,
                    'format_specific_statistics': {
                            'max': 7.0,
                            'min': numpy.float32(0.69999999),
                            'sum': numpy.float32(13.7)
                        },
                    'type': DataTypes.categorical,
                    'uniques_count': 4,
                    'missing_values_count': 1,
                },
            'B': {
                    'format': DataFormats.numerical,
                    'format_specific_statistics': {
                            'max': 3,
                            'min': 1,
                            'sum': 5
                        },
                    'type': DataTypes.categorical,
                    'uniques_count': 2,
                    'missing_values_count': 4
                }
        }
    assert uniques_stats == {
            'B': {1: 2, 3: 1},
            'A': {0.69999998807907104: 1, 1.0: 2, 2.0: 2, 7.0: 1},
            'C': {'1': 1, 'Hi!': 1, 't': 1, 'x3': 1, 'Pandas': 1}
        }


@pytest.mark.parametrize('npartitions', (1, 2, 6))
def test_extract_dataframe_info(boston_dataset, npartitions):
    pandas_df = pandas.read_csv('file://' + boston_dataset['path'], dtype=str)
    pandas_df_info = stats.extract_pandas_dataframe_info(pandas_df)
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=npartitions)
    dask_df_info = stats.extract_dataframe_info(dask_df)
    pandas_df_per_column_statistic = pandas_df_info['per_column_statistic']
    dask_df_per_column_statistic = dask_df_info['per_column_statistic']

    pandas_df_uniques_stats = {}
    for column_name, column_stats in pandas_df_per_column_statistic.items():
        pandas_df_uniques_stats[column_name] = \
                column_stats.pop('uniques_stats').to_dict()
        del column_stats['format_specific_statistics']['sum']
    dask_df_uniques_stats = {}
    for column_name, column_stats in dask_df_per_column_statistic.items():
        dask_df_uniques_stats[column_name] = \
                column_stats.pop('uniques_stats').to_dict()
        del column_stats['format_specific_statistics']['sum']
    assert pandas_df_per_column_statistic == dask_df_per_column_statistic
    assert pandas_df_uniques_stats == dask_df_uniques_stats


@pytest.mark.parametrize('npartitions', (1, 3))
def test_extract_dataframe_info_nan_column(npartitions):
    size = 1000
    pandas_df = pandas.DataFrame({'A':  numpy.tile(float('NaN'), 3 * size)})
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=npartitions)
    dask_df_info = stats.extract_dataframe_info(dask_df)
    column_statistic = dask_df_info['per_column_statistic']['A']
    assert column_statistic['format'] is DataFormats.numerical
    assert column_statistic['type'] is DataTypes.categorical
    assert column_statistic['format_specific_statistics'] == {}


def test_extract_dataframe_info_nan_partition():
    size = 1000
    part_one = pandas.DataFrame({ 'A': numpy.tile(float('NaN'), size)})
    part_two = pandas.DataFrame({ 'A': numpy.linspace(0, size, size)}).astype(str)
    part_three = pandas.DataFrame({ 'A': numpy.tile(float('NaN'), size)})
    delayed = [dask.delayed(part_one), dask.delayed(part_two), dask.delayed(part_three)]
    dask_df = dask.dataframe.from_delayed(delayed)
    dask_df_info = stats.extract_dataframe_info(dask_df)
    column_statistic = dask_df_info['per_column_statistic']['A']
    assert column_statistic['format'] is DataFormats.numerical
    assert column_statistic['type'] is DataTypes.continuous
    assert column_statistic['missing_values_count'] == 2000
    assert column_statistic['format_specific_statistics'] == {
            'min': 0.0,
            'max': 1000.0,
            'sum': 500000.0
        }


def test_extract_dataframe_info_continuous():
    loc = 40
    scale = 60
    size = 1600

    numpy.random.seed(0)
    part_one_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2)).astype(str)
    numpy.random.seed(10)
    part_two_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2)).astype(str)

    categorical_samples = numpy.random.choice(
            numpy.array(['a', 'a', 'c', 'dce', 'B', '1', '2pq']),
            size=size
        )
    part_one_categorical_samples = categorical_samples[:int(size / 2)]
    part_two_categorical_samples = categorical_samples[int(size / 2):]

    alphabet = numpy.array(list(string.ascii_lowercase))
    text_continuous_samples = numpy.array(
            [
                ''.join(numpy.random.choice(alphabet, size=8).tolist()) \
                    for _ in range(0, size)
            ]
        )
    part_one_text_continuous_samples = text_continuous_samples[:int(size / 2)]
    part_two_text_continuous_samples = text_continuous_samples[int(size / 2):]

    part_one = pandas.DataFrame(
            {
                'A': part_one_samples,
                'B': part_one_categorical_samples,
                'C': part_one_text_continuous_samples
            },
            dtype=str
        )
    part_two = pandas.DataFrame(
            {
                'A': part_two_samples,
                'B': part_two_categorical_samples,
                'C': part_two_text_continuous_samples
            },
            dtype=str
        )

    pandas_df = pandas.concat([part_one, part_two])
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=1)
    dask_df_info = stats.extract_dataframe_info(dask_df)
    one_part_dask_df_per_column_statistic = dask_df_info['per_column_statistic']

    delayed = [dask.delayed(part_one), dask.delayed(part_two)]
    dask_df = dask.dataframe.from_delayed(delayed)
    dask_df_info = stats.extract_dataframe_info(dask_df)
    many_part_dask_df_per_column_statistic = dask_df_info['per_column_statistic']

    assert(
            many_part_dask_df_per_column_statistic['A']['binning_stats'] ==
            one_part_dask_df_per_column_statistic['A']['binning_stats']
        )
    assert (
            many_part_dask_df_per_column_statistic['B']['uniques_count'] ==
            one_part_dask_df_per_column_statistic['B']['uniques_count']
        )
    assert many_part_dask_df_per_column_statistic['C']['format'] == DataFormats.character
    assert many_part_dask_df_per_column_statistic['C']['type'] == DataTypes.continuous
