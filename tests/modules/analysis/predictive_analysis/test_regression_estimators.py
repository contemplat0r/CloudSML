import math

import pytest

from app.modules.analysis.predictive_analysis import decision_tree, gbm, linear_regression
from app.modules.data.utils import SplitSampling


@pytest.mark.parametrize(
        'estimator, method_parameters, expected_metrics',
        (
            (
                linear_regression.LinearRegressor,
                {},
                {'MSE': 23.757473106970494, 'R_squared': 0.72193081683662719}
            ),
            (
                decision_tree.DecisionTreeRegressor,
                {'max_depth': 6, 'min_samples_split': 2, 'random_state': 0},
                {'MSE': 24.515592635476111, 'R_squared': 0.71305741194703465}
            ),
            (
                gbm.GBMRegressor,
                {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'max_leaf_nodes': None,
                    'min_samples_split': 2,
                    'learning_rate': 0.01,
                    'loss': 'ls',
                    'random_state': 0
                },
                {'MSE': 25.901296338917131, 'R_squared': 0.69683847513600783}
            ),
        )
    )
def test_regression_predictive_analysis_method_performance_on_test(
        estimator,
        method_parameters,
        expected_metrics,
        boston_dataset
    ):
    test_learn_splitter = SplitSampling(split_ratio=0.4, random_state=0)
    test_dataset_df, learn_dataset_df = test_learn_splitter.split(boston_dataset['dataframe'])
    columns_info=boston_dataset['columns_info']
    for value in columns_info.values():
        if hasattr(value, 'virtual_columns'):
            del value.virtual_columns
    regressor = estimator.build(
            learn_dataset_df,
            target_column_id=boston_dataset['target_column_id'],
            feature_column_ids=boston_dataset['feature_column_ids'],
            columns_info=columns_info,
            categorical_column_ids=[],
            method_parameters=method_parameters
    )
    info = regressor.get_info(test_dataset_df)
    assert set(info.keys()) >= set(expected_metrics.keys())

    for metric_name, expected_metric_value in expected_metrics.items():
        assert math.isclose(info[metric_name], expected_metric_value, rel_tol=1e-05)
