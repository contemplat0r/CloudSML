import pandas
from app.modules.data import stats
from app.modules.data.stats import text_column

def test_compute_column_stats_for_string_format():
    column = pandas.Series(['s', '1', '2.0'], dtype=str)
    column_stats = text_column.compute_column_stats_for_text_format(column)
    assert set(column_stats.keys()) == {'min_len', 'max_len'}


def test_merge_text_column_min_len_max_len():
    column = pandas.Series(
            ['a', '2.0', 'abc', 'ddddd', '2', float('NaN')], dtype=str
        )
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    merged_min_len, merged_max_len = text_column.merge_text_column_min_len_max_len(
            [
                    first_part_stats_values['format_specific_statistics'],
                    second_part_stats_values['format_specific_statistics']
            ]
        )
    assert merged_min_len == 1 and merged_max_len == 5


def test_merge_text_column_partitions_stats():
    column = pandas.Series([float('NaN'), 'a', '2.0', 'abc', 'ddddd', '2'], dtype=str)
    first_part_stats_values = stats.get_column_statistics(column[:3])
    second_part_stats_values = stats.get_column_statistics(column[3:])
    merged_text_stats = text_column.merge_text_column_partitions_stats(
            [
                first_part_stats_values,
                second_part_stats_values
            ]
        )
    assert merged_text_stats == {'min_len': 1, 'max_len': 5}
