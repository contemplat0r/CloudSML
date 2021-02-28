import pandas
import numpy

from cloudsml_computational_backend_common.data.consts import DataFormats, DataTypes

from app.modules.data import stats
from app.modules.data.stats import merge
from app.modules.data.stats import constants


def test_merge_column_partitions_stats():
    column = pandas.Series(
            [float('NaN'), '1', '2.0', '3', float('NaN'), '2', float('NaN')],
            dtype=str
        )
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    merged_stats = merge.merge_column_partitions_stats(
            [first_part_stats_values, second_part_stats_values]
        )
    uniques_stats = merged_stats.pop('uniques_stats')
    assert merged_stats == {
            'format': DataFormats.numerical,
            'missing_values_count': 3,
            'type': DataTypes.categorical,
            'uniques_count': 3,
            'format_specific_statistics': {
                    'sum': 8.0,
                    'min': 1.0,
                    'max': 3,
                },
        }
    assert all(
            uniques_stats.sort_index() == pandas.Series(
                [1.0, 2.0, 1.0],
                index=pandas.Index([1.0, 2.0, 3.0])
            ).sort_index()
        )
    
    column = pandas.Series(
            [float('NaN'), float('NaN'), float('NaN'), 'a', '2.0', 'abc', 'ddddd', '2'],
            dtype=str
        )
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    merged_stats = merge.merge_column_partitions_stats(
            [first_part_stats_values, second_part_stats_values]
        )
    uniques_stats = merged_stats.pop('uniques_stats')

    assert merged_stats == {
            'format': DataFormats.character,
            'missing_values_count': 3,
            'type': DataTypes.categorical,
            'uniques_count': 5,
            'format_specific_statistics': {
                    'min_len': 1,
                    'max_len': 5
                },
        }

    assert all(
            uniques_stats.sort_index() == pandas.Series(
                [1.0, 1.0, 1.0, 1.0, 1.0],
                index=pandas.Index(['2', '2.0', 'a', 'abc', 'ddddd'])
            ).sort_index()
        )


def test_merge_column_missing_values_count():
    column = pandas.Series(
            [float('NaN'), '1', '2.0', '3', float('NaN'), '2', float('NaN')],
            dtype=str
        )
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    assert merge.merge_column_missing_values_count(
            [first_part_stats_values, second_part_stats_values]
        ) == 3


def test_merge():
    dataset_df = pandas.DataFrame(
            {
                'A': ['1', '2.0', float('NaN'), '2.0', '7', '1.0', '1'],
                'B': ['1', float('NaN'), float('NaN'), '1', float('NaN'), float('NaN'), '3'],
                'C': [float('NaN'), 't', '1', 'Hi!', float('NaN'), 'Pandas', 'x3'],
            }
        )
    first_chunk_stats = stats.extract_pandas_dataframe_info(dataset_df[:3])
    second_chunk_stats = stats.extract_pandas_dataframe_info(dataset_df[3:])
    merged_stats = merge.merge([first_chunk_stats, second_chunk_stats])

    per_column_statistic = merged_stats['per_column_statistic']
    a_uniques_stats = per_column_statistic['A'].pop('uniques_stats')
    b_uniques_stats = per_column_statistic['B'].pop('uniques_stats')
    c_uniques_stats = per_column_statistic['C'].pop('uniques_stats')

    assert merged_stats['rows_count'] == 7
    assert per_column_statistic ==  {
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
                            'max': numpy.float32(7.0),
                            'min': numpy.float32(1.0),
                            'sum': numpy.float32(14.0)
                        },
                    'type': DataTypes.categorical,
                    'uniques_count': 3,
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
                    'missing_values_count': 4,
            }
        }
    assert tuple(a_uniques_stats.sort_index().items()) == (
            (1.0, 3.0),
            (2.0, 2.0),
            (7.0, 1.0)
        )
    assert tuple(b_uniques_stats.sort_index().items()) == (
            (1, 2.0),
            (3, 1.0)
        )
    assert tuple(c_uniques_stats.sort_index().items()) == (
            ('1', 1.0),
            ('Hi!', 1.0),
            ('Pandas', 1.0),
            ('t', 1.0),
            ('x3', 1.0)
        )
