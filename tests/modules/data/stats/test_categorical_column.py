import numpy
import pandas

from app.modules.data import stats 
from app.modules.data.stats import categorical_column


def tests_get_uniques_and_frequency_numeric():
    column = pandas.Series([1.1, 0, 1, 2.02, 1, 0, 3, 2.02])
    uniques_data = categorical_column.get_uniques_and_frequency(column)
    uniques_count = uniques_data['uniques_count']
    uniques_stats = uniques_data['uniques_stats'].sort_index()
   
    assert uniques_count == 5
    assert all(uniques_stats.index == pandas.Index([0.0, 1.0, 1.1, 2.02, 3.0]))
    assert all(uniques_stats.values == numpy.array([2, 2, 1, 2, 1]))

def tests_get_uniques_and_frequency_text():
    column = pandas.Series(['ab', 'c', 'd', 'ab', '1', '1'])
    uniques_data = categorical_column.get_uniques_and_frequency(column)
    uniques_count = uniques_data['uniques_count']
    uniques_stats = uniques_data['uniques_stats'].sort_index()
   
    assert uniques_count == 4
    assert all(uniques_stats.index == pandas.Index(['1', 'ab', 'c', 'd']))
    assert all(uniques_stats.values == numpy.array([2, 2, 1, 1]))

def test_merge_column_uniques_count():
    first_column_part_stats = stats.get_column_statistics(
            pandas.Series([float('NaN'), '1', '2.0', float('NaN'), '3'], dtype=str)
        )
    second_column_part_stats = stats.get_column_statistics(
            pandas.Series([float('NaN'), '2', '3', float('NaN')], dtype=str)
        )
    uniques_data = categorical_column.merge_column_uniques_count(
            [first_column_part_stats, second_column_part_stats]
        )
    uniques_count = uniques_data['uniques_count']
    uniques_stats = uniques_data['uniques_stats']
    
    assert uniques_count == 3
    assert all(uniques_stats.sort_index() == pandas.Series(
               [1.0, 2.0, 2.0],
               index=[1.0, 2.0, 3.0]
           ).sort_index()
        )
