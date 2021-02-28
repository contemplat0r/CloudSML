import pytest

import json
import csv

from app.modules.data.readers import dask_universal_read
from app.modules.translator import translator


@pytest.fixture(scope='session')
def empty_translator():
    return translator.Translator()

@pytest.fixture(scope='session')
def regression_gbm_titus_prediction():
    with open(
            str(pytest.config.rootdir.join(
                'tests',
                '_data',
                'regression_gbm_titus_prediction.json'
                ))
        ) as titus_prediction_file:
        titus_prediction_str = titus_prediction_file.read()
        return json.loads(titus_prediction_str)

@pytest.fixture(scope='session')
def linear_regression_titus_prediction():
    with open(
            str(pytest.config.rootdir.join('tests', '_data', 'linear_regression_titus_prediction.json'))
        ) as titus_prediction_file:
        titus_prediction_str = titus_prediction_file.read()
        return json.loads(titus_prediction_str)

@pytest.fixture(scope='session')
def logistic_regression_titus_prediction():
    with open(
            str(pytest.config.rootdir.join('tests', '_data', 'logistic_regression_titus_prediction.json'))
        ) as titus_prediction_file:
        titus_prediction_str = titus_prediction_file.read()
        return json.loads(titus_prediction_str)

@pytest.fixture(scope='session')
def gbm_classifier_titus_prediction():
    with open(
            str(pytest.config.rootdir.join('tests', '_data', 'gbm_classifier_titus_prediction.json'))
        ) as titus_prediction_file:
        titus_prediction_str = titus_prediction_file.read()
        return json.loads(titus_prediction_str)

@pytest.fixture(scope='session')
def pfa_model_min():
    with open(
            str(pytest.config.rootdir.join('tests', '_models', 'gbm-min-pretty.json'))
        ) as pfa_model_file:
        boston_gbm_model_str = pfa_model_file.read()
        return json.loads(boston_gbm_model_str)

@pytest.fixture(scope='session')
def pfa_model_100_estimators():
    with open(
            str(pytest.config.rootdir.join('tests', '_models', 'gbm-100-estimators.pfa'
            ))
        ) as pfa_model_file:
        boston_gbm_model_str = pfa_model_file.read()
        return json.loads(boston_gbm_model_str)

@pytest.fixture(scope='session')
def pfa_model_linear_regression():
    with open(
            str(pytest.config.rootdir.join(
                'tests',
                '_models',
                'linear_regressor_pfa.json'
            ))
        ) as pfa_model_file:
        model_str = pfa_model_file.read()
        return json.loads(model_str)

@pytest.fixture(scope='session')
def pfa_model_logistic_regression():
    with open(
            str(pytest.config.rootdir.join('tests', '_models', 'logistic_regressor_pfa.json'
            ))
        ) as pfa_model_file:
        model_str = pfa_model_file.read()
        return json.loads(model_str)

@pytest.fixture(scope='session')
def pfa_model_gbm_classifier():
    with open(
            str(pytest.config.rootdir.join('tests', '_models', 'gbm_classifier_pfa.json'
            ))
        ) as pfa_model_file:
        model_str = pfa_model_file.read()
        return json.loads(model_str)
