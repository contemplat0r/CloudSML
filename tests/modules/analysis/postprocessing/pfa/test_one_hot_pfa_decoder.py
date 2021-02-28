import copy
import math

from app.modules.data.preprocessing import one_hot_encoding
from app.modules.analysis.postprocessing.pfa import one_hot_pfa_decoder
from app.modules.analysis.predictive_analysis import gbm
from app.modules.data import transformations


def test_one_hot_pfa_decoder():
    pfa_decoder = one_hot_pfa_decoder.OneHotPFADecoder({
            1: [
                    one_hot_pfa_decoder.ColumnInfo(label='1__0', category_value=1.0),
                    one_hot_pfa_decoder.ColumnInfo(label='1__1', category_value=10.0),
                    one_hot_pfa_decoder.ColumnInfo(label='1__2', category_value=100.0),
                ],
            2: [
                    one_hot_pfa_decoder.ColumnInfo(label='2__0', category_value=0.0),
                    one_hot_pfa_decoder.ColumnInfo(label='2__1', category_value=1.0),
                ],
        })
    decoded_pfa_document = pfa_decoder.transform({
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'var0', 'type': 'double'},
                    {'name': 'var1__0', 'type': 'double'},
                    {'name': 'var1__1', 'type': 'double'},
                    {'name': 'var1__2', 'type': 'double'},
                    {'name': 'var2__0', 'type': 'double'},
                    {'name': 'var2__1', 'type': 'double'},
                    {'name': 'var3', 'type': 'double'},
                ],
            },
            'action': {
                'if': {'<=': ['input.var0', 1.0]},
                'then': {
                    'if': {'<=': ['input.var1__2', 0.5]},
                    'then': {'double': 1.0},
                    'else': {'double': 2.0},
                },
                'else': {
                    'if': {'<=': ['input.var2__0', 0.5]},
                    'then': {'double': 3.0},
                    'else': {'double': 4.0},
                },
            }
        })
    assert decoded_pfa_document['input']['fields'] == [
            {'name': 'var0', 'type': 'double'},
            {'name': 'var1', 'type': 'double'},
            {'name': 'var2', 'type': 'double'},
            {'name': 'var3', 'type': 'double'}
        ]
    assert decoded_pfa_document['action'] == {
            'if': {'<=': ['input.var0', 1.0]},
            'then': {
                'if': {'<=': [
                    {
                        'if': {'==': ['input.var1', 100.0]},
                        'then': {'double': 1.0},
                        'else': {'double': 0.0},
                    },
                    0.5
                ]},
                'then': {'double': 1.0},
                'else': {'double': 2.0},
            },
            'else': {
                'if': {'<=': [
                    {
                        'if': {'==': ['input.var2', 0.0]},
                        'then': {'double': 1.0},
                        'else': {'double': 0.0},
                    },
                    0.5
                ]},
                'then': {'double': 3.0},
                'else': {'double': 4.0},
            },
        }


def test_one_hot_pfa_decoder_on_iris(iris_dataset):
    target_column_id = iris_dataset['columns_info_by_name']['petal_width_cm'].id
    class_column_id = iris_dataset['columns_info_by_name']['class'].id

    columns_info = copy.deepcopy(iris_dataset['columns_info'])
    one_hot_encoded_dask_df, _ = one_hot_encoding.OneHotEncoder(
            categorical_columns_ids=[class_column_id],
            columns_info=columns_info
        ).update(iris_dataset['dataframe'], columns_info)

    regressor = gbm.GBMRegressor.build(
            one_hot_encoded_dask_df,
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
    one_hot_encoded_pfa = regressor.to_pfa()

    pfa_decoder = one_hot_pfa_decoder.OneHotPFADecoder({
            class_column_id: columns_info[class_column_id].virtual_columns
        })
    decoded_pfa_document = pfa_decoder.transform(one_hot_encoded_pfa)

    decoded_pfa_document['output'] = {'name': 'petal_width_cm', 'type': 'double'}
    transformed_dask_dataframe = transformations.transform_dask_dataframe(
            iris_dataset['dataframe'],
            transformations=[decoded_pfa_document]
        )

    hot_encoded_predictor_labels = one_hot_encoded_dask_df.columns.drop([target_column_id])
    scikit_regressor_predictions = regressor.estimator.predict(
            one_hot_encoded_dask_df.compute()[hot_encoded_predictor_labels].values
        )
    assert all(
                math.isclose(prediction, scikit_prediction, rel_tol=0.013)\
                    for prediction, scikit_prediction in zip(
                            transformed_dask_dataframe.compute()['petal_width_cm'].values,
                            scikit_regressor_predictions
                        )
        )
