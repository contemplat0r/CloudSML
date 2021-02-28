import pytest

import dask.dataframe
import mock
import numpy
import pandas

from app.extensions import cloudsml
from app.modules.data import transformations


def test_transform_pandas_dataframe():
    pandas_dataframe = pandas.DataFrame(
            [
                {'BIRTH_YEAR': 1900},
                {'BIRTH_YEAR': 1910},
                {'BIRTH_YEAR': 1920},
                {'BIRTH_YEAR': 1930},
                {'BIRTH_YEAR': 2000},
                {'BIRTH_YEAR': 2020},
            ],
            dtype=numpy.float32
        )
    transformation = {
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'varBIRTH_YEAR', 'type': 'double'}
                ]
            },
            'output': {'name': 'AGE', 'type': 'double'},
            'action': {
                '-': [2016, 'input.varBIRTH_YEAR']
            },
        }

    transformed_pandas_dataframe = transformations.transform_pandas_dataframe(
            pandas_df=pandas_dataframe,
            transformation=transformation,
            new_column_name='AGE'
        )
    assert 'AGE' in transformed_pandas_dataframe.columns
    assert (
            transformed_pandas_dataframe['AGE'].tolist() == [116, 106, 96, 86, 16, -4]
        )


def test_transform_pandas_dataframe_wrong_pfa_record_field_type():
    pandas_dataframe = pandas.DataFrame(
            {
                'BIRTH_YEAR': [1900, 1910, 1930, 2000, 2020],
                'ID': [1, 2, 3, 4, 5]
            },
            dtype=numpy.float32
        )
    transformation = {
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'varBIRTH_YEAR', 'type': 'double'},
                    {'name': 'varID', 'type': 'integer'}
                ]
            },
            'output': {'name': 'AGE', 'type': 'double'},
            'action': {
                '-': [2016, 'input.varBIRTH_YEAR']
            },
        }
    with pytest.raises(AssertionError) as assertion_error:
        transformations.transform_pandas_dataframe(
                pandas_df=pandas_dataframe,
                transformation=transformation,
                new_column_name='NEW'
            )
    assert "Some PFA field records are not of double type:" in str(assertion_error.value)


def test_transform_pandas_dataframe_wrong_dtype():
    pandas_dataframe = pandas.DataFrame(
            {
                'BIRTH_YEAR': [1900, 1910, 1930, 2000, 2020],
                'ID': [1, 2, 3, 4, 5]
            },
            dtype=numpy.int32
        )
    transformation = {
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'varBIRTH_YEAR', 'type': 'double'},
                    {'name': 'varID', 'type': 'double'}
                ]
            },
            'output': {'name': 'AGE', 'type': 'double'},
            'action': {
                '-': [2016, 'input.varBIRTH_YEAR']
            },
        }
    with pytest.raises(AssertionError) as assertion_error:
        transformations.transform_pandas_dataframe(
                pandas_df=pandas_dataframe,
                transformation=transformation,
                new_column_name='NEW'
            )
    assert str(assertion_error.value)[:60] == (
            "Some pandas dataframe columns are not in numpy.float32 dtype"
        )


def test_transform_dask_dataframe():
    dask_dataframe = dask.dataframe.from_pandas(
            pandas.DataFrame(
                    [
                        {'BIRTH_YEAR': 1900},
                        {'BIRTH_YEAR': 1910},
                        {'BIRTH_YEAR': 1920},
                        {'BIRTH_YEAR': 1930},
                        {'BIRTH_YEAR': 2000},
                        {'BIRTH_YEAR': 2020},
                    ],
                    dtype=numpy.float32
                ),
            npartitions=3
        )
    new_dask_dataframe = transformations.transform_dask_dataframe(
            dask_dataframe,
            transformations=[
                {
                    'input': {
                        'type': 'record',
                        'fields': [
                            {'name': 'varBIRTH_YEAR', 'type': 'double'}
                        ]
                    },
                    'output': {'name': 'AGE', 'type': 'double'},
                    'action': {
                        '-': [2016, 'input.varBIRTH_YEAR']
                    },
                },
            ]
        )
    assert 'BIRTH_YEAR' in new_dask_dataframe.columns
    assert 'AGE' in new_dask_dataframe.columns
    assert (new_dask_dataframe['AGE'].compute() == [116, 106, 96, 86, 16, -4]).all()


def test_transform_dask_dataframe_with_more_than_one_transformation():
    dask_dataframe = dask.dataframe.from_pandas(
            pandas.DataFrame(
                    [
                        {'BIRTH_YEAR': 1900},
                        {'BIRTH_YEAR': 1910},
                        {'BIRTH_YEAR': 1920},
                        {'BIRTH_YEAR': 1930},
                        {'BIRTH_YEAR': 2000},
                        {'BIRTH_YEAR': 2020},
                    ],
                    dtype=numpy.float32
                ),
            npartitions=3
        )
    new_dask_dataframe = transformations.transform_dask_dataframe(
            dask_dataframe,
            transformations=[
                    {
                        'input': {
                            'type': 'record',
                            'fields': [
                                {'name': 'varBIRTH_YEAR', 'type': 'double'}
                            ]
                        },
                        'output': {'name': 'AGE', 'type': 'double'},
                        'action': {
                            '-': [2016, 'input.varBIRTH_YEAR']
                        },
                    },
                    {
                        'input': {
                            'type': 'record',
                            'fields': [
                                {'name': 'varAGE', 'type': 'double'}
                            ]
                        },
                        'output': {'name': 'DIFFERENT', 'type': 'double'},
                        'action': {
                            '-': [18, 'input.varAGE']
                        }
                    }
                ]
        )
    assert 'BIRTH_YEAR' in new_dask_dataframe.columns
    assert 'AGE' in new_dask_dataframe.columns
    assert 'DIFFERENT' in new_dask_dataframe.columns
    assert (new_dask_dataframe['AGE'].compute() == [116, 106, 96, 86, 16, -4]).all()
    assert (new_dask_dataframe['DIFFERENT'].compute() == [-98, -88, -78, -68, 2, 22]).all()


@pytest.mark.parametrize('use_raw', (True, False))
def test_boston_dataset_transformation(boston_dataset, use_raw):
    if use_raw:
        dataframe = boston_dataset['raw_dataframe']
        get_column_label = lambda name: name
    else:
        dataframe = boston_dataset['dataframe']
        get_column_label = lambda name: boston_dataset['columns_info_by_name'][name].id
    new_dask_dataframe = transformations.transform_dask_dataframe(
            dataframe,
            transformations=[
                    {
                        'input': {
                            'type': 'record',
                            'fields': [
                                {'name': 'var%s' % get_column_label('CRIM'), 'type':'double'}
                            ]
                        },
                        'output': {'name': 'CRIM_TEST', 'type': 'double'},
                        'action': {
                            '*': [2, 'input.var%s' % get_column_label('CRIM')]
                        }
                    },
                    {
                        'input': {
                            'type': 'record',
                            'fields': [
                                {'name': 'var%s' % get_column_label('CRIM'), 'type': 'double'}
                            ]
                        },
                        'output': {'name': 'CRIM_IF', 'type': 'double'},
                        'action': {
                            'if': {
                                '>=': ['input.var%s' % get_column_label('CRIM'), 1]
                            },
                            'then': 1,
                            'else': 0
                        }
                    },
                    {
                        'input': {
                            'type': 'record',
                            'fields': [
                                {'name': 'var%s' % get_column_label('ZN'), 'type': 'double'},
                                {'name': 'var%s' % get_column_label('INDUS'), 'type': 'double'}
                            ]
                        },
                        'output': {'name': -1, 'type': 'double'},
                        'action': {
                            'if': {
                                '>': ['input.var%s' % get_column_label('INDUS'), 0]
                            },
                            'then': {
                                '/': [
                                    'input.var%s' % get_column_label('ZN'),
                                    'input.var%s' % get_column_label('INDUS')
                                ]
                            },
                            'else': 0.0
                        }
                    }
                ]
        )
    assert 'CRIM_TEST' in new_dask_dataframe.columns
    assert 'CRIM_IF' in new_dask_dataframe.columns
    assert -1 in new_dask_dataframe.columns
    assert (new_dask_dataframe['CRIM_TEST'].compute()[:4] < 1).all()
    assert (new_dask_dataframe['CRIM_IF'].compute()[:4] == 0).all()
    assert numpy.isclose(
            new_dask_dataframe[-1].compute()[:4],
            [7.792207792207792, 0, 0, 0]
        ).all()


def test_fetch_data_transformation_by_id_in_pfa(cb_app):

    def patched_call_api(*args, **kwargs):
        patched_call_api.call_count = getattr(patched_call_api, 'call_count', 0) + 1
        return True
    
    with mock.patch.object(
            cloudsml.data_api.api_client, 'call_api', patched_call_api
        ) as mocked_call:
        for _ in transformations.fetch_data_transformations_by_id_in_pfa([3, 2, 1]):
            pass

        assert mocked_call.call_count == 3
