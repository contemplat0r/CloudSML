# encoding: utf-8
# Copyright 2016 Salford Systems, San Diego, CA, USA.

import functools
import random
import string

import mock
import pytest

import dask
import dask.dataframe
import numpy
import pandas

import cloudsml
from app import create_app
from app.modules.data import stats
from app.utils import json


def update_dataset_with_computed_values(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dataset = func(*args, **kwargs)

        raw_dataframe = dataset['raw_dataframe']
        dataset['dataframe_info'] = json.loads(json.dumps(
                stats.extract_dataframe_info(raw_dataframe)
            ))
        if 'feature_column_labels' not in dataset:
            dataset['feature_column_labels'] = raw_dataframe.columns.drop(
                    dataset['target_column_label']
                )

        if 'MSE_baseline' not in dataset:
            y = raw_dataframe[dataset['target_column_label']]
            dataset['MSE_baseline'] = ((y - y.mean())**2).mean().compute()

        per_column_statistic = dataset['dataframe_info']['per_column_statistic']
        dataset['columns_info'] = {
                column_id: cloudsml.models.BaseDataTransformationColumn(
                        id=column_id,
                        name=column_name,
                        statistics=per_column_statistic[column_name],
                        data_type=per_column_statistic[column_name]['type'],
                        data_format=per_column_statistic[column_name]['format']
                    ) for column_id, column_name in zip(
                            sorted(random.sample(range(100000), len(raw_dataframe.columns))),
                            dataset['dataframe_info']['columns']
                        )
            }
        dataset['columns_info_by_name'] = {
                column.name: column for column in dataset['columns_info'].values()
            }
        dataframe = raw_dataframe.rename(
                columns={
                        column.name: column.id for column in dataset['columns_info'].values()
                    }
            )
        dataset['dataframe'] = dataframe
        dataset['target_column_id'] = dataset['columns_info_by_name'][
                dataset['target_column_label']
            ].id
        dataset['feature_column_ids'] = dataset['dataframe'].columns.drop(
                dataset['target_column_id']
            ).values.tolist()
        return dataset
    return wrapper


@pytest.fixture(scope='session')
@update_dataset_with_computed_values
def boston_dataset():
    boston_file_path = str(pytest.config.rootdir.join("tests", "_data", "boston.csv"))
    boston_df = dask.dataframe.read_csv(boston_file_path, dtype=numpy.float32)
    return {
            'path': boston_file_path,
            'raw_dataframe': boston_df,
            'target_column_label': 'MV',
        }


@pytest.fixture(scope='session')
@update_dataset_with_computed_values
def iris_dataset():
    iris_file_path = str(pytest.config.rootdir.join("tests", "_data", "iris.csv"))
    iris_df = dask.dataframe.read_csv(iris_file_path, dtype=numpy.float32)
    return {
            'path': iris_file_path,
            'raw_dataframe': iris_df,
            'target_column_label': 'class',
            'MSE_baseline': 4.31,
        }


@pytest.fixture(scope='session')
@update_dataset_with_computed_values
def titanic_dataset():
    titanic_file_path = str(pytest.config.rootdir.join("tests", "_data", "titanic_train.csv"))
    titanic_df = dask.dataframe.read_csv(titanic_file_path)
    return {
            'path': titanic_file_path,
            'raw_dataframe': titanic_df,
            'target_column_label': 'Survived',
        }


@pytest.fixture(scope='session')
def generate_dataframe(request):
    df_type, dtype = request.param
    loc = 40
    scale = 60
    size = 1600

    numpy.random.seed(0)
    part_one_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2))
    numpy.random.seed(10)
    part_two_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2))

    categorical_samples = numpy.random.choice(
            numpy.array(['a', 'ab', 'c', 'dce', 'b', '1', '2pq']),
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
            dtype=dtype
        )
    part_two = pandas.DataFrame(
            {
                'A': part_two_samples,
                'B': part_two_categorical_samples,
                'C': part_two_text_continuous_samples
            },
            dtype=dtype
        )

    if df_type == 'pandas':
        return pandas.concat([part_one, part_two])
    elif df_type == 'dask':
        return dask.dataframe.from_delayed([dask.delayed(part_one), dask.delayed(part_two)])
    else:
        raise ValueError("Unknown dataframe type: %s" % df_type)


@pytest.yield_fixture(scope='session')
def cb_app():
    class MockedApiClient(cloudsml.ApiClient):
        def call_api(self, *args, **kwargs):
            return True

    with mock.patch.object(cloudsml, 'ApiClient', MockedApiClient):
        with mock.patch.object(
                cloudsml.Configuration, 'get_oauth2_token', lambda *args, **kwargs: True
            ):
            yield create_app(config_name='development')
