import copy
import math

import dask.dataframe
import numpy
import pandas

from app.modules.analysis.predictive_analysis import gbm
from app.modules.data.utils import SplitSampling
from app.modules.data.preprocessing import one_hot_encoding


def test_gbm_classifier_build(iris_dataset):
    classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    assert isinstance(classifier, gbm.GBMClassifier)
    assert classifier.estimator is not None


def test_gbm_classifier_to_pfa(iris_dataset):
    classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    pfa_as_python_structure = classifier.to_pfa()
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}


def test_compute_staged_performance_stats(iris_dataset):
    classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    learn_dataset_df = iris_dataset['dataframe'].compute()
    staged_performance_stats = classifier._compute_staged_performance_stats(
            learn_dataset_df[iris_dataset['feature_column_ids']].values,
            learn_dataset_df[iris_dataset['target_column_id']].values
        )
    performance_stats_2 = staged_performance_stats['performance_stats'][2]
    assert set(staged_performance_stats.keys()) == {'estimators_count', 'performance_stats'}
    assert performance_stats_2['ROC'] == 1.0


def test_compute_staged_performance_stats_learn_test(iris_dataset):
    test_learn_splitter = SplitSampling(split_ratio=0.4, random_state=0)
    test_dataset_df, learn_dataset_df = test_learn_splitter.split(iris_dataset['dataframe'])
    classifier = gbm.GBMClassifier.build(
            learn_dataset_df,
            target_column_id=iris_dataset['target_column_id']
        )
    test_dataset_df = test_dataset_df.compute()
    staged_performance_stats = classifier._compute_staged_performance_stats(
            test_dataset_df[iris_dataset['feature_column_ids']].values,
            test_dataset_df[iris_dataset['target_column_id']].values
        )
    performance_stats_2 = staged_performance_stats['performance_stats'][2]
    assert set(staged_performance_stats.keys()) == {'estimators_count', 'performance_stats'}
    assert math.isclose(performance_stats_2['ROC'], 0.58888888888888891)


def test_gbm_classifier_get_partial_dependence(iris_dataset):
    dask_df = iris_dataset['dataframe']

    petal_width_cm_column_id = iris_dataset['columns_info_by_name']['petal_width_cm'].id

    dask_df = dask_df.astype({
            petal_width_cm_column_id: pandas.core.dtypes.dtypes.CategoricalDtype()
        })
    pandas_df = dask_df.compute()
    column = pandas_df[petal_width_cm_column_id]
    all_column_categories = column.cat.categories
    new_column = column.cat.codes.astype(numpy.int16, copy=False)

    new_column.replace(
            to_replace={
                code: all_column_categories.get_loc(category) \
                        for code, category in enumerate(column.cat.categories)
                    },
            inplace=True
        )
    pandas_df[petal_width_cm_column_id] = new_column
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=1)

    columns_info = copy.deepcopy(iris_dataset['columns_info'])
    one_hot_encoded_dask_df, _ = one_hot_encoding.OneHotEncoder(
            categorical_columns_ids=[petal_width_cm_column_id],
            columns_info=columns_info
        ).update(dask_df, columns_info)

    classifier = gbm.GBMClassifier.build(
            one_hot_encoded_dask_df,
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            categorical_column_ids=[petal_width_cm_column_id],
            columns_info=columns_info,
            method_parameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 2,
                    'max_leaf_nodes': None,
                    'learning_rate': 0.01,
                    'random_state': 0
                }
        )
    partial_dependence = classifier._get_partial_dependence(
            one_hot_encoded_dask_df.compute().drop([iris_dataset['target_column_id']], axis=1),
            iris_dataset['feature_column_ids']
        )
    assert set(partial_dependence.keys()) == {'feature_axes', 'outputs'}
    assert set(partial_dependence['feature_axes'].keys()) == {
            iris_dataset['columns_info_by_name']['petal_length_cm'].id,
            iris_dataset['columns_info_by_name']['sepal_width_cm'].id,
            iris_dataset['columns_info_by_name']['sepal_length_cm'].id,
            iris_dataset['columns_info_by_name']['petal_width_cm'].id
        }
    sepal_width_cm_column_id = iris_dataset['columns_info_by_name']['sepal_width_cm'].id
    assert math.isclose(
            partial_dependence['feature_axes'][sepal_width_cm_column_id][1],
            2.2,
            abs_tol=0.001
        )
    petal_length_cm_column_id = iris_dataset['columns_info_by_name']['petal_length_cm'].id
    assert math.isclose(
            partial_dependence['feature_axes'][petal_length_cm_column_id][2],
            1.2,
            abs_tol=0.001
        )
    petal_width_cm_column_id = iris_dataset['columns_info_by_name']['petal_width_cm'].id
    assert partial_dependence['feature_axes'][petal_width_cm_column_id] == [
            0.10000000149011612,
            0.20000000298023224,
            0.30000001192092896,
            0.4000000059604645,
            0.5,
            0.6000000238418579,
            1.0,
            1.100000023841858,
            1.2000000476837158,
            1.2999999523162842,
            1.399999976158142,
            1.5,
            1.600000023841858,
            1.7000000476837158,
            1.7999999523162842,
            1.899999976158142,
            2.0,
            2.0999999046325684,
            2.200000047683716,
            2.299999952316284,
            2.4000000953674316,
            2.5
        ]
    assert set(partial_dependence['outputs'][0].keys()) == {'target_class', 'feature_id', 'output'}
    assert partial_dependence['outputs'][0]['target_class'] == 0.0
    sepal_length_cm_column_id = iris_dataset['columns_info_by_name']['sepal_length_cm'].id
    assert partial_dependence['outputs'][0]['feature_id'] == sepal_length_cm_column_id
    assert math.isclose(
            partial_dependence['outputs'][0]['output'][0],
            -0.166,
            abs_tol=0.001
        )
    assert partial_dependence['outputs'][9]['target_class'] == 0.0
    petal_width_cm_column_id = iris_dataset['columns_info_by_name']['petal_width_cm'].id
    assert partial_dependence['outputs'][9]['feature_id'] == petal_width_cm_column_id
    assert len(partial_dependence['outputs'][9]['output']) == 22
    assert partial_dependence['outputs'][9]['output'][6] == -0.16603830108992307


def test_gbm_classifier_get_info(iris_dataset):
    classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=iris_dataset['columns_info']
        )
    info = classifier.get_info(iris_dataset['dataframe'])
    assert set(info.keys()) == {
            'ROC', 'staged_performance_stats', 'variable_importance', 'partial_dependence'
        }
    assert info['ROC'] == 1.0


def test_gbm_classifier_get_info_variable_importance(iris_dataset):
    classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=iris_dataset['columns_info'],
            method_parameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 2,
                    'max_leaf_nodes': None,
                    'learning_rate': 0.01,
                    'random_state': 0
                }
        )
    info = classifier.get_info(iris_dataset['dataframe'])
    assert info['variable_importance'] == {
            iris_dataset['columns_info_by_name']['sepal_width_cm'].id: 0.75463653098457362,
            iris_dataset['columns_info_by_name']['petal_length_cm'].id: 26.817314571562552,
            iris_dataset['columns_info_by_name']['petal_width_cm'].id: 71.159321192076376,
            iris_dataset['columns_info_by_name']['sepal_length_cm'].id: 1.2687277053764612
        }
