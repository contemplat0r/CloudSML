# encoding: utf-8

"""
One-hot encoding categorical data for dask dataframe
------------------------------------------------------------
"""
from collections import namedtuple

import numpy

from cloudsml_computational_backend_common.data.consts import DataFormats

VirtualColumnInfo = namedtuple('VirtualColumnInfo', ['label', 'category_value'])
CategoricalColumnInfo = namedtuple(
        'CategoricalColumnInfo',
        [
            'categorical_column_id',
            'one_hot_encoded_column_labels',
            'one_hot_encoded_column_categories',
        ]
    )


class OneHotEncoder(object):

    def __init__(self, categorical_columns_ids, columns_info):
        self.categorical_columns_ids = categorical_columns_ids
        self.categorical_columns_info = []
        for column_id in self.categorical_columns_ids:
            column_statictics = columns_info[column_id].statistics
            column_format = column_statictics['format']
            uniques_count = column_statictics['uniques_count']
            if column_format == DataFormats.character:
                if column_statictics['missing_values_count'] > 0:
                    one_hot_encoded_column_ids = range(-1, uniques_count)
                else:
                    one_hot_encoded_column_ids = range(uniques_count)
                one_hot_encoded_column_categories = one_hot_encoded_column_ids
            elif column_format == DataFormats.numerical:
                one_hot_encoded_column_ids = range(uniques_count)
                one_hot_encoded_column_categories = [
                        unique_value for unique_value, _ in column_statictics['uniques_stats']
                    ]
            else:
                raise TypeError(
                        "Column format '%s' is not supported for one-hot encoding.",
                        column_format
                    )
            self.categorical_columns_info.append(
                        CategoricalColumnInfo(
                            categorical_column_id=column_id,
                            one_hot_encoded_column_labels=[
                                "%s__%s" % (
                                    column_id,
                                    one_hot_encoded_column_id
                                ) for one_hot_encoded_column_id in one_hot_encoded_column_ids
                            ],
                        one_hot_encoded_column_categories=one_hot_encoded_column_categories
                    )
                )

    def _update_partition(self, partition):
        one_hot_encoded_partition = partition.copy(deep=False)
        for categorical_column_info in self.categorical_columns_info:
            column = one_hot_encoded_partition[categorical_column_info.categorical_column_id]
            for one_hot_encoded_column_label, category_value in zip(
                    categorical_column_info.one_hot_encoded_column_labels,
                    categorical_column_info.one_hot_encoded_column_categories
                ):
                one_hot_encoded_partition[one_hot_encoded_column_label] = \
                        (column == category_value).astype(numpy.int16)
        one_hot_encoded_partition.drop(
                self.categorical_columns_ids,
                axis=1,
                inplace=True
            )
        return one_hot_encoded_partition

    def _update_columns_info(self, columns_info):
        for categorical_column_info in self.categorical_columns_info:
            categorical_column_id = categorical_column_info.categorical_column_id
            columns_info[categorical_column_id].virtual_columns = [
                    VirtualColumnInfo(
                            label=one_hot_encoded_colum_label,
                            category_value=category_value
                        )  for one_hot_encoded_colum_label, category_value in zip(
                            categorical_column_info.one_hot_encoded_column_labels,
                            categorical_column_info.one_hot_encoded_column_categories
                    )
                ]
        return columns_info

    def update(self, dask_df, columns_info):
        columns_info = self._update_columns_info(columns_info)
        return (
                dask_df.map_partitions(
                    self._update_partition,
                    meta=self._update_partition(dask_df._meta)
                ),
                columns_info
            )
