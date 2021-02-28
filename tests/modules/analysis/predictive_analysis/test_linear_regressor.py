import json

from app.modules.analysis.predictive_analysis import linear_regression


def test_linear_regression_build(boston_dataset):
    regressor = linear_regression.LinearRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    assert isinstance(regressor, linear_regression.LinearRegressor)
    assert regressor.estimator is not None

def test_linear_regression_to_pfa(boston_dataset):
    regressor = linear_regression.LinearRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    pfa_as_python_structure = regressor.to_pfa()
    with open('linear_regression_pfa.json', 'w') as pfa_file:
        pfa_file.write(json.dumps(pfa_as_python_structure, indent=4))
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}

def test_linear_regression_info(boston_dataset):
    dataframe = boston_dataset['dataframe']
    regressor = linear_regression.LinearRegressor.build(
            dataframe,
            target_column_id=boston_dataset['target_column_id']
        )
    info = regressor.get_info(dataframe)
    assert round(info['MSE'], 3) == 21.895
    assert round(info['R_squared'], 3) == 0.741
