# encoding: utf-8

"""
This module compute column statistics (per partition).
------------------------------------------------------
"""
import dask
import dask.bag
import numpy

from cloudsml_computational_backend_common.data.consts import DataFormats, DataTypes

from . import (
        constants,
        generic_column,
        categorical_column,
        continuous_column,
        numerical_column,
        text_column,
        merge
    )


COMPUTE_COLUMN_STATS_BY_FORMAT = {
        DataFormats.numerical: numerical_column.\
            compute_column_stats_for_numerical_format,
        DataFormats.character: text_column.\
            compute_column_stats_for_text_format
    }


def get_column_statistics(column):
    """
    Collect all computed by other functions statistic.
    """
    missing_values_mask = column.isnull()

    number_of_missing_values = numpy.count_nonzero(missing_values_mask)
    if number_of_missing_values > 0:
        column = column[~missing_values_mask]

    column_statistics = {
            'missing_values_count': number_of_missing_values
        }
    column, column_format = generic_column.detect_and_cast_column_format(column)
    column_statistics['format'] = column_format
    column_statistics['format_specific_statistics'] = (
            COMPUTE_COLUMN_STATS_BY_FORMAT[column_format](column)
        )

    uniques_data = categorical_column.get_uniques_and_frequency(column)
    uniques_count = uniques_data['uniques_count']
    column_statistics['type'] = column_type = generic_column.detect_column_type(uniques_count)
    if column_type is DataTypes.categorical:
        column_statistics['uniques_count'] = uniques_count
        column_statistics['uniques_stats'] = uniques_data['uniques_stats']

    # TODO: Implement rest of the stats
    return column_statistics


def extract_pandas_dataframe_info(dataset_df):
    """
    Get statistics for all columns in pandas dataframe (partition).
    """
    return {
            'rows_count': dataset_df.shape[0],
            'per_column_statistic': {
                    column: get_column_statistics(dataset_df[column])\
                        for column in dataset_df.columns
                }
        }


def extract_pandas_dataframe_binned_info(bins_dict):
    def wrapper(dataset_df):
        binning_stats = {}
        for column_label in dataset_df.columns:
            column = dataset_df[column_label]
            missing_values_mask = column.isnull()
            if missing_values_mask.any():
                column = column[~missing_values_mask]
            binning_stats[column_label] = continuous_column.get_binned_data(
                    column.astype(numpy.float64),
                    bins=bins_dict[column_label]
                )
        return {'per_column_binned_stats': binning_stats}
    return wrapper


def extract_dataframe_info(dataset_df):
    """
    Run parallel stats computation on partitions with piramid merging the
    partial results.
    """
    dataset_columns = dataset_df.columns
    df_bag_of_pandas_df = dask.bag\
        .from_delayed(dataset_df.to_delayed())\
        .map_partitions(lambda df: [df])

    dataframes_statistics = df_bag_of_pandas_df.map(extract_pandas_dataframe_info)

    while dataframes_statistics.npartitions > 1:
        dataframes_statistics = dataframes_statistics\
            .repartition(max(1, dataframes_statistics.npartitions // 4))\
            .map_partitions(
                    lambda stats_data_map: [merge.merge(list(stats_data_map))]
                )

    statistics = dataframes_statistics.compute()[0]

    per_column_statistic = statistics['per_column_statistic']
    for column_statistic in per_column_statistic.values():
        if 'uniques_stats' in column_statistic:
            column_statistic['uniques_stats'].sort_index(inplace=True)

    binned_columns = {}
    for column_name, column_stats in per_column_statistic.items():
        if (
                column_stats['format'] is DataFormats.numerical and
                column_stats['type'] is DataTypes.continuous
            ):
            format_specific_statistics = column_stats['format_specific_statistics']
            if format_specific_statistics:
                binned_columns[column_name] = numpy.linspace(
                        format_specific_statistics['min'],
                        format_specific_statistics['max'],
                        constants.BINS_COUNT
                    )

    if binned_columns:
        binned_columns_labels = list(binned_columns.keys())
        for_binning_dataset_df = dataset_df[binned_columns_labels]
        df_bag_of_pandas_df = dask.bag\
            .from_delayed(for_binning_dataset_df.to_delayed())\
            .map_partitions(lambda df: [df])

        dataframes_binned_stats = df_bag_of_pandas_df.map(
                extract_pandas_dataframe_binned_info(binned_columns)
            )
        while dataframes_binned_stats.npartitions > 1:
            dataframes_binned_stats = dataframes_binned_stats\
                .repartition(max(1, dataframes_binned_stats.npartitions // 4))\
                .map_partitions(
                        lambda stats_data_map: [merge.merge_binned_stats(list(stats_data_map))]
                    )
        binned_stats = dataframes_binned_stats.compute()[0]['per_column_binned_stats']
        for column_label, column_binned_stats in binned_stats.items():
            binning_stats = column_binned_stats['binning_stats']
            ranges = binning_stats['ranges']
            ranges = [
                    (left_margin, right_margin) \
                        for left_margin, right_margin in zip(ranges[:-1], ranges[1:])
                ]
            per_column_statistic[column_label]['binning_stats'] = tuple(
                    zip(ranges, binning_stats['stats'])
                )

    return {
            'columns': dataset_columns.tolist(),
            'columns_count': dataset_columns.shape[0],
            'rows_count': statistics.pop('rows_count'),
            'per_column_statistic': statistics['per_column_statistic']
        }
