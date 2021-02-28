from app.modules.analysis.predictive_analysis import decision_tree


def test_decision_tree_regressor_build(boston_dataset):
    regressor = decision_tree.DecisionTreeRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    assert isinstance(regressor, decision_tree.DecisionTreeRegressor)
    assert regressor.estimator is not None


def test_decision_tree_regressor_to_pfa(boston_dataset):
    regressor = decision_tree.DecisionTreeRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    pfa_as_python_structure = regressor.to_pfa()
    assert set(pfa_as_python_structure.keys()) == {'action', 'input', 'output'}
    assert set(pfa_as_python_structure['action'].keys()) == {'else', 'if', 'then'}


def test_decision_tree_regressor_info(boston_dataset):
    columns_info=boston_dataset['columns_info']
    for value in columns_info.values():
        if hasattr(value, 'virtual_columns'):
            del value.virtual_columns
    dataframe = boston_dataset['dataframe']
    regressor = decision_tree.DecisionTreeRegressor.build(
            dataframe,
            target_column_id=boston_dataset['target_column_id'],
            feature_column_ids=boston_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=columns_info,
            method_parameters={
                'max_depth': 6,
                'min_samples_split': 2,
                'random_state': 0,
            }
        )
    info = regressor.get_info(dataframe)
    assert round(info['MSE'], 3) == 4.647
    assert round(info['R_squared'], 3) == 0.945


def test_decision_tree_regressor_get_info_variable_importance(boston_dataset):
    columns_info=boston_dataset['columns_info']
    for value in columns_info.values():
        if hasattr(value, 'virtual_columns'):
            del value.virtual_columns
    regressor = decision_tree.DecisionTreeRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id'],
            feature_column_ids=boston_dataset['feature_column_ids'],
            categorical_column_ids=[],
            columns_info=columns_info,
            method_parameters={
                'max_depth': 6,
                'min_samples_split': 2,
                'random_state': 0,
            }
        )
    info = regressor.get_info(boston_dataset['dataframe'])
    assert info['variable_importance'] == {
            boston_dataset['columns_info_by_name']['PT'].id: 0.5171481780726106,
            boston_dataset['columns_info_by_name']['CHAS'].id: 0.0,
            boston_dataset['columns_info_by_name']['CRIM'].id: 5.9421847856209862,
            boston_dataset['columns_info_by_name']['RM'].id: 59.911062840275711,
            boston_dataset['columns_info_by_name']['DIS'].id: 7.2682023320025655,
            boston_dataset['columns_info_by_name']['AGE'].id: 0.43048791983020973,
            boston_dataset['columns_info_by_name']['NOX'].id: 2.2546708882373165,
            boston_dataset['columns_info_by_name']['ZN'].id: 0.091913180959621962,
            boston_dataset['columns_info_by_name']['INDUS'].id: 0.22778493200710082,
            boston_dataset['columns_info_by_name']['B'].id: 0.22847792513830856,
            boston_dataset['columns_info_by_name']['TAX'].id: 2.3691653111793274,
            boston_dataset['columns_info_by_name']['RAD'].id: 0.0,
            boston_dataset['columns_info_by_name']['LSTAT'].id: 20.758901706676241
        }
