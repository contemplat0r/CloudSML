import copy
import math

from app.modules.analysis.predictive_analysis import gbm
from app.modules.data.utils import SplitSampling
from app.modules.data.preprocessing import one_hot_encoding


def test_gbm_regressor_build(boston_dataset):
    regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    assert isinstance(regressor, gbm.GBMRegressor)
    assert regressor.estimator is not None


def test_gbm_regressor_to_pfa(boston_dataset):
    regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    pfa_as_python_structure = regressor.to_pfa()
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}
    assert round(
            pfa_as_python_structure['action']\
                    ['+'][1]['*'][1]['a.sum']['new'][0]['if']['<='][1], 3) == 6.941


def test_compute_staged_performance_stats(boston_dataset):
    regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id'],
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
    learn_dataset_df = boston_dataset['dataframe'].compute()
    staged_performance_stats = regressor._compute_staged_performance_stats(
            learn_dataset_df[regressor.feature_column_labels].values,
            learn_dataset_df[boston_dataset['target_column_id']].values
        )
    performance_stats_2 = staged_performance_stats['performance_stats'][2]
    assert set(staged_performance_stats.keys()) == {'estimators_count', 'performance_stats'}
    assert round(performance_stats_2['MSE'], 3) == 79.751


def test_compute_staged_performance_stats_learn_test(boston_dataset):
    test_learn_splitter = SplitSampling(split_ratio=0.4, random_state=0)
    test_dataset_df, learn_dataset_df = test_learn_splitter.split(boston_dataset['dataframe'])
    regressor = gbm.GBMRegressor.build(
            learn_dataset_df,
            target_column_id=boston_dataset['target_column_id'],
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
    test_dataset_df = test_dataset_df.compute()
    staged_performance_stats = regressor._compute_staged_performance_stats(
            test_dataset_df[regressor.feature_column_labels].values,
            test_dataset_df[boston_dataset['target_column_id']].values
        )
    performance_stats_2 = staged_performance_stats['performance_stats'][2]
    assert set(staged_performance_stats.keys()) == {'estimators_count', 'performance_stats'}
    assert math.isclose(performance_stats_2['MSE'], 81.690813183593107)


def test_gbm_regressor_get_partial_dependence(iris_dataset):
    target_column_id = iris_dataset['columns_info_by_name']['petal_width_cm'].id
    class_column_id = iris_dataset['columns_info_by_name']['class'].id
    dask_df = iris_dataset['dataframe']
    feature_column_ids = iris_dataset['feature_column_ids'][:]
    feature_column_ids.remove(target_column_id)
    feature_column_ids.append(class_column_id)

    columns_info = copy.deepcopy(iris_dataset['columns_info'])
    one_hot_encoded_dask_df, _  = one_hot_encoding.OneHotEncoder(
            categorical_columns_ids=[class_column_id],
            columns_info=columns_info
        ).update(dask_df, columns_info)
    regressor = gbm.GBMRegressor.build(
            one_hot_encoded_dask_df,
            target_column_id=target_column_id,
            columns_info=columns_info,
            categorical_column_ids=[class_column_id],
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
    partial_dependence_for_column = regressor._get_partial_dependence_for_column(
            one_hot_encoded_dask_df.compute().drop(target_column_id, axis=1),
            class_column_id,
            grid_resolution=100
        )
    assert partial_dependence_for_column == (
            [0, 1, 2],
            [
                {
                    'feature_id': class_column_id,
                    'output': [
                            -0.27731963908713814,
                            -0.069759987080723743,
                            0.078165796932529161
                        ]
                }
            ]
        )
    partial_dependence = regressor._get_partial_dependence(
            one_hot_encoded_dask_df.compute().drop([target_column_id], axis=1),
            feature_column_ids
        )

    assert set(partial_dependence.keys()) == {'feature_axes', 'outputs'}
    assert set(partial_dependence['feature_axes'].keys()) == set(feature_column_ids)
    sepal_length_cm_column_id = iris_dataset['columns_info_by_name']['sepal_length_cm'].id
    assert len(partial_dependence['feature_axes'][sepal_length_cm_column_id]) == 35
    assert math.isclose(
            partial_dependence['feature_axes'][sepal_length_cm_column_id][1],
            4.4,
            abs_tol=0.001
        )
    sepal_width_cm_column_id = iris_dataset['columns_info_by_name']['sepal_width_cm'].id
    assert len(partial_dependence['feature_axes'][sepal_width_cm_column_id]) == 23
    assert math.isclose(
            partial_dependence['feature_axes'][sepal_width_cm_column_id][1],
            2.2,
            abs_tol=0.001
        )
    assert partial_dependence['feature_axes'][class_column_id] == [0, 1, 2]
    assert set(partial_dependence['outputs'][0].keys()) == {'feature_id', 'output'}
    assert partial_dependence['outputs'][0]['feature_id'] == sepal_length_cm_column_id
    assert math.isclose(
            partial_dependence['outputs'][0]['output'][0],
            -0.005,
            abs_tol=0.001
        )
    assert partial_dependence['outputs'][3]['feature_id'] == class_column_id
    assert partial_dependence['outputs'][3]['output'] == [
            -0.27731963908713814,
            -0.069759987080723743,
            0.078165796932529161
        ]

def test_gbm_regressor_get_info(boston_dataset):
    columns_info=boston_dataset['columns_info']
    for value in columns_info.values():
        if hasattr(value, 'virtual_columns'):
            del value.virtual_columns
    regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id'],
            feature_column_ids=boston_dataset['feature_column_ids'],
            columns_info=columns_info,
            categorical_column_ids=[],
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
    info = regressor.get_info(boston_dataset['dataframe'])
    assert set(info.keys()) == {
            'MSE',
            'R_squared',
            'staged_performance_stats',
            'variable_importance',
            'partial_dependence'
        }
    assert round(info['MSE'], 3) == 14.083
    assert round(info['R_squared'], 3) == 0.833


def test_gbm_regressor_get_info_variable_importance(boston_dataset):
    columns_info=boston_dataset['columns_info']
    for value in columns_info.values():
        if hasattr(value, 'virtual_columns'):
            del value.virtual_columns
    regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id'],
            feature_column_ids=boston_dataset['feature_column_ids'],
            columns_info=columns_info,
            categorical_column_ids=[],
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
    info = regressor.get_info(boston_dataset['dataframe'])
    assert info['variable_importance'] == {
            boston_dataset['columns_info_by_name']['CHAS'].id: 0.020192924297734748,
            boston_dataset['columns_info_by_name']['RAD'].id: 0.28074291456472322,
            boston_dataset['columns_info_by_name']['PT'].id: 1.3631822580944224,
            boston_dataset['columns_info_by_name']['LSTAT'].id: 37.040548557882232,
            boston_dataset['columns_info_by_name']['ZN'].id: 0.014326340820187248,
            boston_dataset['columns_info_by_name']['AGE'].id: 0.82418387333869014,
            boston_dataset['columns_info_by_name']['TAX'].id: 1.4008009039805327,
            boston_dataset['columns_info_by_name']['DIS'].id: 7.639515046460482,
            boston_dataset['columns_info_by_name']['NOX'].id: 2.8036245653414609,
            boston_dataset['columns_info_by_name']['CRIM'].id: 3.4721726110079731,
            boston_dataset['columns_info_by_name']['B'].id: 1.3346684411192249,
            boston_dataset['columns_info_by_name']['RM'].id: 43.539915383412357,
            boston_dataset['columns_info_by_name']['INDUS'].id: 0.26612617967995023
        }
