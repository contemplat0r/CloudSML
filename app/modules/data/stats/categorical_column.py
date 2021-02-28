import numpy
import pandas


def get_uniques_and_frequency(column):
    """
    Get uniques values and frequency for each value
    """
    # TODO, uniques_frequency for detecting dense/condensed?
    uniques_stats = column.value_counts()
    return {
            'uniques_count': uniques_stats.shape[0],
            'uniques_stats': uniques_stats
        }

def merge_column_uniques_count(per_column_partitions_stats):
    """
    Merge a list of unique values statistics for categorical data
    into a single aggregated result.
    """
    cumulative_stats = per_column_partitions_stats[0]['uniques_stats']
    for stats in per_column_partitions_stats[1:]:
        partition_uniques_stats = stats['uniques_stats']
        cumulative_stats = cumulative_stats.add(partition_uniques_stats, fill_value=0)
    return {
            'uniques_count': cumulative_stats.shape[0],
            'uniques_stats': cumulative_stats
        }
