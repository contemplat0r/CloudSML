import pandas

from app.modules.data.stats import numerical_column
from app.modules.data import stats


def test_compute_column_stats_for_numerical_format():
    column = pandas.Series([0, 1, 2, -1])
    column_stats = numerical_column.compute_column_stats_for_numerical_format(column)
    assert set(column_stats.keys()) == {
            'min',
            'max',
            'sum',
        }
    column = pandas.Series([])
    column_stats = numerical_column.compute_column_stats_for_numerical_format(column)
    assert column_stats == {}


def test_compute_column_stats_for_numerical_format_min_max():
    column = pandas.Series([0.8, 1, 2, -1])
    column_stats = numerical_column.compute_column_stats_for_numerical_format(column)
    assert column_stats['min'] == -1
    assert column_stats['max'] == 2


def test_compute_column_stats_for_numerical_format_sum():
    column = pandas.Series([0.8, 1, 2.0, 0])
    column_stats = numerical_column.compute_column_stats_for_numerical_format(column)
    assert column_stats['sum'] == 3.80


def test_merge_numerical_column_partitions_stats():
    column = pandas.Series([float('NaN'), '1', '2.0', '3', '2', float('NaN')], dtype=str)
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    merged_numerical_stats = numerical_column.merge_numerical_column_partitions_stats(
            [
                first_part_stats_values,
                second_part_stats_values
            ]
        )
    assert merged_numerical_stats == {'min': 1, 'max': 3, 'sum': 8.0}
