# encoding: utf-8

import tempfile
import string

import pytest

import dask.dataframe
import numpy
import pandas


from app.modules.data import stats
from app.modules.data.preprocessing import one_hot_encoding
from cloudsml.models.base_data_transformation_column import BaseDataTransformationColumn


@pytest.mark.parametrize('npartitions', (1, 2, 3))
def test_one_hot_encoding(npartitions):
    loc = 40
    scale = 60
    size = 1600
    
    numpy.random.seed(0)
    uniques_values_numerical = numpy.array([0.1, 0.0, 2.7, 3.03, 5., 2.4, 4.2, 7.8])
    categorical_samples_numerical = numpy.random.choice(
            uniques_values_numerical,
            size=size
        )
    uniques_values_character = numpy.array([0, 1, 2, 3, 4, 5, 6])
    categorical_samples_character = numpy.random.choice(
            uniques_values_character,
            size=size
        )

    pandas_df = pandas.DataFrame(
            {
                1: categorical_samples_numerical,
                2: categorical_samples_character,
            },
        )
    columns_info = {
            1: BaseDataTransformationColumn(
                title=None,
                data_type=None,
                name='A',
                id=1,
                data_transformation=None,
                statistics={
                    'uniques_stats' : [
                        item for item in zip(
                            *numpy.unique(categorical_samples_numerical, return_counts=True)
                            )
                        ],
                    'uniques_count': len(uniques_values_numerical),
                    'format': 'numerical',
                    'type': 'categorical',
                    'missing_values_count': 0
                }
            ),
            2: BaseDataTransformationColumn(
                title=None,
                data_type=None,
                name='B',
                id=2,
                data_transformation=None,
                statistics={
                    'uniques_stats' : [
                        item for item in zip(
                            *numpy.unique(categorical_samples_character, return_counts=True)
                            )
                        ],
                    'uniques_count': len(uniques_values_character),
                    'format': 'character',
                    'type': 'categorical',
                    'missing_values_count': 0
                },
            )
        }

    categorical_columns_ids = [1, 2]
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=npartitions)
    one_hot_encoded_dask_df, updated_columns_info = one_hot_encoding.OneHotEncoder(
            categorical_columns_ids=categorical_columns_ids,
            columns_info=columns_info
        ).update(dask_df, columns_info)
    one_hot_encoded_pandas_df = one_hot_encoded_dask_df.compute()
    assert (
            one_hot_encoded_pandas_df.columns.shape[0] == updated_columns_info[1].statistics['uniques_count'] + updated_columns_info[2].statistics['uniques_count']
        )
    uniques_stats_numerical_values = [
            unique_value for unique_value, _ in columns_info[1].statistics['uniques_stats']
        ]
    for i in range(len(uniques_values_numerical)):
        assert all(
                (one_hot_encoded_pandas_df['1__%s' % i] == 1) == (
                    pandas_df[1] == uniques_stats_numerical_values[i]
            )
        )

    for i in range(7):
        assert all(
                (one_hot_encoded_pandas_df['2__%d' % i] == 1) == (pandas_df[2] == i)
            )
    virtual_column_info = updated_columns_info[1].virtual_columns[2]
    assert virtual_column_info.label == '1__2'
    assert virtual_column_info.category_value == 2.4
    virtual_column_info = updated_columns_info[2].virtual_columns[2]
    assert virtual_column_info.label == '2__2'
    assert virtual_column_info.category_value == 2
