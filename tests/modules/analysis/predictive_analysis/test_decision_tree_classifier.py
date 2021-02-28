import math

from app.modules.analysis.predictive_analysis import decision_tree


def test_decision_tree_classifier_build(iris_dataset):
    classifier = decision_tree.DecisionTreeClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    assert isinstance(classifier, decision_tree.DecisionTreeClassifier)
    assert classifier.estimator is not None


def test_decision_tree_classifier_to_pfa(iris_dataset):
    classifier = decision_tree.DecisionTreeClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    pfa_as_python_structure = classifier.to_pfa()
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}
    assert set(pfa_as_python_structure['action'].keys()) == {'else', 'if', 'then'}


def test_decision_tree_classifier_info(iris_dataset):
    dataframe = iris_dataset['dataframe']
    classifier = decision_tree.DecisionTreeClassifier.build(
            dataframe,
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=iris_dataset['columns_info'],
            method_parameters={'min_samples_split': 2}
        )
    info = classifier.get_info(dataframe)
    assert math.isclose(info['ROC'], 1.0)

def test_decision_tree_classifier_get_info_variable_importance(iris_dataset):
    dataframe = iris_dataset['dataframe']
    classifier = decision_tree.DecisionTreeClassifier.build(
            dataframe,
            target_column_id=iris_dataset['target_column_id'],
            feature_column_ids=iris_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=iris_dataset['columns_info'],
            method_parameters={
                'max_depth': 6,
                'min_samples_split': 2,
                'random_state': 0,
            }
        )
    info = classifier.get_info(dataframe)
    assert info['variable_importance'] == {
            iris_dataset['columns_info_by_name']['sepal_width_cm'].id: 1.3333333333333328,
            iris_dataset['columns_info_by_name']['petal_length_cm'].id: 6.4055958132045054,
            iris_dataset['columns_info_by_name']['petal_width_cm'].id: 92.261070853462158,
            iris_dataset['columns_info_by_name']['sepal_length_cm'].id: 0.0
        }
