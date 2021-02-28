# encoding: utf-8

"""
Merge previously computed per partitions dataset statistics.
------------------------------------------------------------
"""

from cloudsml_computational_backend_common.data.consts import DataFormats, DataTypes

from . import (
        constants,
        generic_column,
        numerical_column,
        text_column,
        categorical_column,
        continuous_column
    )


MERGE_COLUMN_STATS_BY_FORMAT = {
        DataFormats.numerical: numerical_column.\
            merge_numerical_column_partitions_stats,
        DataFormats.character: text_column.\
            merge_text_column_partitions_stats
    }


def merge_column_missing_values_count(per_column_partitions_stats):
    """
    Merge column missing values count for all partitions
    """
    return sum(
            column_statistics['missing_values_count']\
                for column_statistics in per_column_partitions_stats
        )


def merge_column_partitions_stats(per_column_partitions_stats):
    """
    Collect all statistics computed by other functions.
    """
    merged_column_statistics = {
            'missing_values_count': merge_column_missing_values_count(
                    per_column_partitions_stats
                )
        }

    # Filter out the partitions will all data missing
    first_column_partition_stats = per_column_partitions_stats[0]
    per_column_partitions_stats = [
            stats for stats in per_column_partitions_stats \
                if stats['format_specific_statistics']
        ]
    if not per_column_partitions_stats:
        first_column_partition_stats.update(merged_column_statistics)
        return first_column_partition_stats

    merged_format = per_column_partitions_stats[0]['format']
    merged_column_statistics['format'] = merged_format
    assert all(
            partition_stats['format'] is merged_format \
                for partition_stats in per_column_partitions_stats
        )
    if any(
            partition_stats['format'] is DataFormats.character \
                for partition_stats in per_column_partitions_stats
        ):
        merged_column_statistics['format'] = DataFormats.character
    elif all(
            partition_stats['format'] is DataFormats.numerical \
                for partition_stats in per_column_partitions_stats
        ):
        merged_column_statistics['format'] = DataFormats.numerical
    else:
        raise ValueError("Column format is neither numerical nor character")
    merged_column_statistics['type'] = per_column_partitions_stats[0]['type']
    # XXX Presence uniques_count in general will be checked for each chunk/partition!
    if all('uniques_count' in stats for stats in per_column_partitions_stats):
        uniques_data = (
                categorical_column.merge_column_uniques_count(
                        per_column_partitions_stats
                    )
            )
        merged_column_statistics['type'] = column_type = generic_column.detect_column_type(
                uniques_data['uniques_count']
            )
        if column_type is DataTypes.categorical:
            merged_column_statistics['uniques_count'] = uniques_data['uniques_count']
            merged_column_statistics['uniques_stats'] = uniques_data['uniques_stats']
    else:
        merged_column_statistics['type'] = DataTypes.continuous
    merged_column_statistics['format_specific_statistics'] = (
            MERGE_COLUMN_STATS_BY_FORMAT[merged_format](per_column_partitions_stats)
        )
    return merged_column_statistics


def merge(dataset_per_partitions_stats):
    """
    Main merge function
    """
    dataset_merged_stats = {
            'rows_count': sum(
                    stats['rows_count'] for stats in dataset_per_partitions_stats
                )
        }
    collected_per_column_statistic = [
            stats['per_column_statistic']\
                for stats in dataset_per_partitions_stats
        ]
    columns = dataset_per_partitions_stats[0]['per_column_statistic'].keys()
    merged_per_column_statistic = {}
    for column_label in columns:
        merged_per_column_statistic[column_label] = merge_column_partitions_stats(
                [partition_stats[column_label]\
                    for partition_stats in collected_per_column_statistic]
            )
    dataset_merged_stats['per_column_statistic'] = merged_per_column_statistic
    return dataset_merged_stats


def merge_binned_stats(dataset_per_partitions_binned_stats):
    collected_per_column_binned_stats = [
            stats['per_column_binned_stats'] \
                for stats in dataset_per_partitions_binned_stats
        ]
    columns = dataset_per_partitions_binned_stats[0]['per_column_binned_stats'].keys()
    merged_per_column_binned_stats = {}
    for column_label in columns:
        merged_per_column_binned_stats[column_label] = \
            continuous_column.merge_binned_data(
                [
                    partition_binned_stats[column_label] \
                        for partition_binned_stats in \
                        collected_per_column_binned_stats
                ]
            )
    return {'per_column_binned_stats': merged_per_column_binned_stats}
