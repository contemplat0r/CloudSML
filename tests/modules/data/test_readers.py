import pytest

import dask.dataframe
import numpy
import pandas

from cloudsml.models import BaseDataTransformationColumn
from cloudsml_computational_backend_common.data.consts import DataFormats, DataTypes

from app.modules.data import readers


@pytest.mark.parametrize('category_column', ('class', 'petal_width_cm'))
def test_CategoriesUnifier_update(iris_dataset, category_column):
    pandas_df = iris_dataset['dataframe'].compute()
    category_column_id = iris_dataset['columns_info_by_name'][category_column].id
    pandas_df[category_column] = pandas_df[category_column_id].astype('category')
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=3)
    categories_unifier = readers.CategoriesUnifier(
            categories_by_column={
                category_column: pandas_df[category_column].cat.categories[::-1]
            }
        )
    df_with_replaced_categories = categories_unifier.update(dask_df).compute()

    n_categories = len(pandas_df[category_column].cat.categories)
    expected_codes = pandas_df[category_column].cat.codes.replace(
            {i: n_categories - i - 1 for i in range(n_categories)}
        )
    assert all(df_with_replaced_categories[category_column] == expected_codes)


def test_dask_universal_read_into_strings():
    pandas_df = pandas.read_csv('tests/_data/iris.csv', dtype=str)
    dask_df = readers.dask_universal_read('tests/_data/iris.csv', columns_info=None)
    assert all(dask_df.dtypes.values == numpy.dtype('O'))
    assert all(dask_df.compute()['petal_width_cm'] == pandas_df['petal_width_cm'])


def test_dask_universal_read_with_columns_info():
    pandas_df_float32 = pandas.read_csv('tests/_data/iris.csv', dtype=numpy.float32)
    dask_df = readers.dask_universal_read(
            'tests/_data/iris.csv',
            columns_info=[
                    BaseDataTransformationColumn(
                            id=index,
                            name=name,
                            data_format=DataFormats.numerical,
                            data_type=DataTypes.continuous
                        ) for index, name in enumerate(pandas_df_float32.columns)
                ]
        )
    dtypes = pandas.Series(
            data=[numpy.float32] * 5,
            index=pandas.Index(range(len(pandas_df_float32.columns)))
        )
    assert all(dask_df.dtypes == dtypes)
    assert all(dask_df.compute()[3] == pandas_df_float32['petal_width_cm'])


def test_dask_universal_read_with_categorical():
    columns_info = [
            BaseDataTransformationColumn(
                    id=0,
                    name='sepal_length_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=1,
                    name='sepal_width_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=2,
                    name='petal_length_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=3,
                    name='petal_width_cm',
                    data_format=DataFormats.numerical,
                    data_type=DataTypes.categorical,
                    statistics={
                            'uniques_stats': [
                                    (0.1, 1),
                                    (0.2, 1),
                                    (0.3, 1),
                                    (0.4, 1),
                                    (0.5, 1),
                                    (0.6, 1),
                                    (1.0, 1),
                                    (1.1, 1),
                                    (1.2, 1),
                                    (1.3, 1),
                                    (1.4, 1),
                                    (1.5, 1),
                                    (1.6, 1),
                                    (1.7, 1),
                                    (1.8, 1),
                                    (1.9, 1),
                                    (2.0, 1),
                                    (2.1, 1),
                                    (2.2, 1),
                                    (2.3, 1),
                                    (2.4, 1),
                                    (2.5, 1),
                                ]
                        }
                ),
            BaseDataTransformationColumn(
                    id=4,
                    name='class',
                    data_format=DataFormats.character,
                    data_type=DataTypes.categorical,
                    statistics={
                            'uniques_stats': [
                                    ('0', 1),
                                    ('1', 1),
                                    ('2', 1),
                                ]
                        }
                ),
        ]
    dask_df = readers.dask_universal_read('tests/_data/iris.csv', columns_info=columns_info)
    df_with_replaced_categories = dask_df.compute()
    pandas_df_with_category_dtype = pandas.read_csv(
            'tests/_data/iris.csv',
            dtype={
                    column.name: (
                            numpy.float32
                            if column.data_format is DataFormats.numerical else
                            numpy.int16
                        ) for column in columns_info
                }
        )
    assert all(dask_df.dtypes.values == pandas_df_with_category_dtype.dtypes.values)
    assert all(
            df_with_replaced_categories[3].values ==
            pandas_df_with_category_dtype['petal_width_cm'].values
        )
    assert all(
            df_with_replaced_categories[4].values ==
            pandas_df_with_category_dtype['class'].astype(numpy.int16)
        )

def test_dask_universal_read_with_numeric_categorical():
    columns_info = [
            BaseDataTransformationColumn(
                    id=0,
                    name='sepal_length_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=1,
                    name='sepal_width_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=2,
                    name='petal_length_cm',
                    data_format=DataFormats.numerical
                ),
            BaseDataTransformationColumn(
                    id=3,
                    name='petal_width_cm',
                    data_format=DataFormats.numerical,
                    data_type=DataTypes.categorical,
                    statistics={
                            'uniques_stats': [
                                    (0.1, 1),
                                    (0.2, 1),
                                    (0.3, 1),
                                    (0.4, 1),
                                    (0.5, 1),
                                    (0.6, 1),
                                    (1.0, 1),
                                    (1.1, 1),
                                    (1.2, 1),
                                    (1.3, 1),
                                    (1.4, 1),
                                    (1.5, 1),
                                    (1.6, 1),
                                    (1.7, 1),
                                    (1.8, 1),
                                    (1.9, 1),
                                    (2.0, 1),
                                    (2.1, 1),
                                    (2.2, 1),
                                    (2.3, 1),
                                    (2.4, 1),
                                    (2.5, 1),
                                ]
                        }
                ),
            BaseDataTransformationColumn(
                    id=4,
                    name='class',
                    data_format=DataFormats.character,
                    data_type=DataTypes.categorical,
                    statistics={
                            'uniques_stats': [
                                    ('0', 1),
                                    ('1', 1),
                                    ('2', 1),
                                ]
                        }
                ),
        ]
    dask_df = readers.dask_universal_read('tests/_data/iris.csv', columns_info=columns_info)
    df_with_replaced_categories = dask_df.compute()
    pandas_df_with_category_dtype = pandas.read_csv(
            'tests/_data/iris.csv',
            dtype={
                    column.name: (
                            numpy.float32
                            if column.data_format is DataFormats.numerical else
                            numpy.int16
                        ) for column in columns_info
                }
        )
    assert all(dask_df.dtypes.values == pandas_df_with_category_dtype.dtypes.values)
    assert all(
            df_with_replaced_categories[3].values ==
            pandas_df_with_category_dtype['petal_width_cm'].values
        )
    assert all(
            df_with_replaced_categories[4].values ==
            pandas_df_with_category_dtype['class'].astype(numpy.int16)
        )
