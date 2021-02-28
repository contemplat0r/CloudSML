import math

import pytest


from app.modules.analysis.predictive_analysis import decision_tree, gbm, logistic_regression
from app.modules.data.utils import SplitSampling

@pytest.mark.parametrize(
        'estimator, method_parameters, expected_metrics',
        (
            (
                logistic_regression.LogisticRegressionClassifier,
                {},
                {'ROC': 0.9555555555555556}
            ),
            (
                decision_tree.DecisionTreeClassifier,
                {},
                {'ROC': 0.97777777777777775}
            ),
            (
                gbm.GBMClassifier,
                {},
                {'ROC': 0.96666666666666667}
            ),
        )
    )
def test_classification_predictive_analysis_method_performance_on_test(
        estimator,
        method_parameters,
        expected_metrics,
        iris_dataset
    ):
    test_learn_splitter = SplitSampling(split_ratio=0.4, random_state=0)
    test_dataset_df, learn_dataset_df = test_learn_splitter.split(iris_dataset['dataframe'])
    classifier = estimator.build(
            learn_dataset_df,
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            columns_info=iris_dataset['columns_info'],
            categorical_column_ids=[],
            method_parameters=method_parameters
        )
    info = classifier.get_info(test_dataset_df)

    assert set(info.keys()) >= set(expected_metrics.keys())

    for metric_name, expected_metric_value in expected_metrics.items():
        assert math.isclose(info[metric_name], expected_metric_value)
