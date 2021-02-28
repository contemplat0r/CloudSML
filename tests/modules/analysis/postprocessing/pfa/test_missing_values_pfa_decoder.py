import math

import json
import dask.dataframe
import numpy
import pytest

from app.modules.analysis.postprocessing.pfa import missing_values_pfa_decoder
from app.modules.analysis.predictive_analysis import gbm
from app.modules.data import transformations
from app.modules.data.preprocessing.missing_values_encoding import FILL_MISSING_CONST


def test_missing_values_pfa_decoder():
    pfa_decoder = missing_values_pfa_decoder.MissingValuesPFADecoder({'3' : FILL_MISSING_CONST})
    decoded_pfa_document = pfa_decoder.transform({
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'var0', 'type': 'double'},
                    {'name': 'var1', 'type': 'double'},
                    {'name': 'var2', 'type': 'double'},
                    {'name': 'var3', 'type': 'double'},
                ],
            },
            'action': {
                'if': {'<=': ['input.var0', 1.0]},
                'then': {
                    'if': {'<=': ['input.var1', 0.5]},
                    'then': {'double': 1.0},
                    'else': {'double': 2.0},
                },
                'else': {
                    'if': {'<=': ['input.var2', 1.5]},
                    'then': {'double': 'input.var3'},
                    'else': {'double': 2.0},
                },
            }
        })
    assert decoded_pfa_document['input']['fields'] == [
            {'name': 'var0', 'type': 'double'},
            {'name': 'var1', 'type': 'double'},
            {'name': 'var2', 'type': 'double'},
            {'name': 'var3', 'type': ['double', 'null']}
        ]
    assert decoded_pfa_document['action'] == {
            'if': {'<=': ['input.var0', 1.0]},
            'then': {
                'if': {'<=': ['input.var1',0.5]},
                'then': {'double': 1.0},
                'else': { 'double': 2.0}
            },
            'else': {
                'if': {'<=': ['input.var2', 1.5]},
                'then': {
                    'double': {
                        'ifnotnull': {'input.var3': 'input.var3'},
                        'then': 'input.var3',
                        'else': FILL_MISSING_CONST
                    }
                },
                'else': {'double': 2.0}
            }
        }


def test_missing_values_pfa_decoder_on_boston(boston_dataset):
    dask_df = boston_dataset['dataframe']
    pandas_df = dask_df.compute()
    target_column_id = boston_dataset['target_column_id']
    feature_column_ids = boston_dataset['feature_column_ids']
    column_with_nans_id = feature_column_ids[-1]
    nan_logical_vector_pattern = numpy.random.choice(numpy.array([True, False]), pandas_df.shape[0])
    imitating_nan_zero_filled_pandas_df = pandas_df.copy()
    imitating_nan_zero_filled_pandas_df[column_with_nans_id][nan_logical_vector_pattern] = FILL_MISSING_CONST
    imitating_nan_zero_filled_dask_df = dask.dataframe.from_pandas(
            imitating_nan_zero_filled_pandas_df,
            npartitions=1
        )
    regressor = gbm.GBMRegressor.build(
            imitating_nan_zero_filled_dask_df,
            target_column_id=target_column_id,
            method_parameters={
                'n_estimators': 100,
                'max_depth': 6,
                'max_leaf_nodes': None,
                'min_samples_split': 2,
                'learning_rate': 0.01,
                'loss': 'ls',
                'random_state': 0
            }
        )
    missing_values_pfa = regressor.to_pfa()

    pfa_decoder = missing_values_pfa_decoder.MissingValuesPFADecoder({column_with_nans_id: FILL_MISSING_CONST})
    decoded_pfa_document = pfa_decoder.transform(missing_values_pfa)

    nan_filled_pandas_df = pandas_df.copy()
    nan_filled_pandas_df[column_with_nans_id][nan_logical_vector_pattern] = numpy.nan
    nan_filled_dask_df = dask.dataframe.from_pandas(
            imitating_nan_zero_filled_pandas_df,
            npartitions=1
        )
    decoded_pfa_document['output'] = {'name': target_column_id, 'type': 'double'}
    transformed_dask_dataframe = transformations.transform_dask_dataframe(
            nan_filled_dask_df,
            transformations=[decoded_pfa_document]
        )
    predictions = transformed_dask_dataframe.compute()[target_column_id].values
    features_values = imitating_nan_zero_filled_pandas_df[feature_column_ids].values
    scikit_regressor_predictions = regressor.estimator.predict(features_values)
    assert all(
                math.isclose(prediction, scikit_prediction, rel_tol=0.0001)\
                    for prediction, scikit_prediction in zip(
                            predictions,
                            scikit_regressor_predictions
                        )
        )
