from app.modules.analysis.predictive_analysis import logistic_regression

def test_logistic_regression_build(iris_dataset):
    regressor = logistic_regression.LogisticRegressionClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    assert isinstance(regressor, logistic_regression.LogisticRegressionClassifier)
    assert regressor.estimator is not None

def test_logistic_regression_to_pfa(iris_dataset):
    regressor = logistic_regression.LogisticRegressionClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    pfa_as_python_structure = regressor.to_pfa()
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}

def test_logistic_regression_info(iris_dataset):
    dataframe = iris_dataset['dataframe']
    regressor = logistic_regression.LogisticRegressionClassifier.build(
            dataframe,
            target_column_id=iris_dataset['target_column_id']
        )
    info = regressor.get_info(dataframe)
    assert round(info['ROC'], 3) == 0.793
