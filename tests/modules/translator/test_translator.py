import math
import tempfile
import subprocess
import json
import os

from cloudsml_computational_backend_common.analysis.consts import PredictiveModelExportFormats

from app.modules.analysis.predictive_analysis import linear_regression, logistic_regression, gbm


def test_translate_if_form(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate(
            {
                'if': {'<': [2, {'+': [1, 2]}]},
                'then' : {'-': [5.0, 'x']},
                'else': -7
            }
        ) == "(((5.0 - x)) if ((2 < (1 + 2))) else (-7))"


def test_translate_new_form(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate(
            {'new': [
                20,
                {
                    'if': {'<=': [0, {'+': [1, 2]}]},
                    'then' : {'-': [4.0, 'x']},
                    'else': -7
                },
                {
                    'if': {'<': [2, {'+': [1, 2]}]},
                    'then' : {'-': [5.0, 'x']},
                    'else': -7
                }
            ]
        }
    ) == "[20, (((4.0 - x)) if ((0 <= (1 + 2))) else (-7)), (((5.0 - x)) if ((2 < (1 + 2))) else (-7))]"
    empty_translator.lang = None


def test_translate_ifnotnull_form_python(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    translation_result = empty_translator.translate(
            {
                'ifnotnull': {'x': 1, 'y': 2},
                'then' : 1,
                'else': 0 
            }
        )
    assert (
            (translation_result == '((1) if not (math.isnan(1) or math.isnan(2)) else (0))') or
            (translation_result == '((1) if not (math.isnan(2) or math.isnan(1)) else (0))')
        )
    assert empty_translator.translate(
            {
                'ifnotnull': {'x': 'math.nan'},
                'then' : 1,
                'else': 0 
            }
        ) == '((1) if not (math.isnan(math.nan)) else (0))'
    empty_translator.lang = None


def test_translate_ifnotnull_form_c(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.c
    translation_result = empty_translator.translate(
            {
                'ifnotnull': {'x': 1, 'y': 2},
                'then' : 1,
                'else': 0 
            }
        )
    assert  (
            (translation_result == '(!(isnan(2) || isnan(1)) ? (1) : (0))') or
            (translation_result == '(!(isnan(1) || isnan(2)) ? (1) : (0))')
        )
    translation_result = empty_translator.translate(
            {
                'ifnotnull': {'x': math.nan},
                'then' : 1,
                'else': 0 
            }
        )
    assert translation_result == '(!(isnan(nan)) ? (1) : (0))'
    translation_result = empty_translator.translate(
            {
                'ifnotnull': {'x': 'input.X'},
                'then' : 1,
                'else': 0 
            }
        )
    assert translation_result == '(!(isnan(input.X)) ? (1) : (0))'
    empty_translator.lang = None


def test_translate_array_sum(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate(
        {
            'a.sum': {
                'type': 'array',
                'new': [
                    20,
                    {
                        'if': {'<=': [0, {'+': [1, 2]}]},
                        'then' : {'-': [4.0, 'x']},
                        'else': -7
                    },
                    {
                        'if': {'<': [2, {'+': [1, 2]}]},
                        'then' : {'-': [5.0, 'x']},
                        'else': -7
                    }
                ]
            }
        }
    ) == "sum([20, (((4.0 - x)) if ((0 <= (1 + 2))) else (-7)), (((5.0 - x)) if ((2 < (1 + 2))) else (-7))])"
    empty_translator.lang = None


def test_translate_math_link_function_logit(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate(
            {
                'm.link.logit': {
                    'type': {
                        'type': 'array',
                        'items': 'double'
                    },
                    'new': [1.0, 2.0, 3.0]
                }
            }
        ) == "logit([1.0, 2.0, 3.0])"
    empty_translator.lang = None


def test_translate_math_fuction_argmax(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate(
            {
                'a.argmax': {
                    'type': {
                        'type': 'array',
                        'items': 'double'
                    },
                    'new': [2.7, 2.0, 3.3, 3.1]
                }
            }
        ) == "argmax([2.7, 2.0, 3.3, 3.1])"
    empty_translator.lang = None


def test_translate_function_la_add(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate({
        'la.add': [
            {
                'type': {
                    'type': 'array',
                    'items': {
                        'items': 'double',
                        'type': 'array'
                        }
                    },
                'new': [
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [1.0, 2.0, 3.0]
                        },
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': ['x.a', 'x.b', 'x.c']
                        }
                    ]
                },
            {
                'type': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': 'double'
                        }
                    },
                'new': [
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': ['y.a', 'y.b', 'y.c']
                        },
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [8.0, 11, 16.0]
                        }
                    ]
                }
            ]}) == "add([[1.0, 2.0, 3.0], [x.a, x.b, x.c]], [[y.a, y.b, y.c], [8.0, 11, 16.0]])"
    empty_translator.lang = None


def test_translate_function_la_dot(empty_translator):
    empty_translator.lang = PredictiveModelExportFormats.python
    assert empty_translator.translate({
        'la.dot': [
            {
                'type': {
                    'type': 'array',
                    'items': {
                        'items': 'double',
                        'type': 'array'
                        }
                    },
                'new': [
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': ['x.a', 'x.b', 'x.c']
                        },
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [4.0, 5.0, 6.0]
                        }
                    ]
                },
            {
                'type': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': 'double'
                        }
                    },
                'new': [
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [7.0, 10]
                        },
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [8.0, 11]
                        },
                    {
                        'type': {
                            'type': 'array',
                            'items': 'double'
                            },
                        'new': [9.0, 12]
                        }
                    ]
                }
            ]}) == "dot([[x.a, x.b, x.c], [4.0, 5.0, 6.0]], [[7.0, 10], [8.0, 11], [9.0, 12]])"
    empty_translator.lang = None


def test_translate_pfa_document_gbm_regression_python(
        boston_dataset,
        empty_translator,
    ):
    scikit_regressor = gbm.GBMRegressor.build(
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
    output_file = tempfile.NamedTemporaryFile(mode='w')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.python,
            output_file=output_file,
            pfa_document=scikit_regressor.to_pfa()
        )
    scoring_result = json.loads(
            subprocess.check_output(
                ['python', output_file.name, boston_dataset['path'], '/dev/null']
            ).decode()
        )
    dataset_df = boston_dataset['dataframe'].compute()
    features_values = dataset_df[boston_dataset['feature_column_ids']].values
    scikit_regressor_predictions = scikit_regressor.estimator.predict(features_values)
   
    assert all(
            math.isclose(prediction, scikit_prediction, rel_tol=0.001) \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_regressor_predictions)
        )

def test_translate_pfa_document_gbm_classification(
        iris_dataset,
        empty_translator,
    ):
    scikit_classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.python,
            output_file=output_file,
            pfa_document=scikit_classifier.to_pfa()
        )
    scoring_result = json.loads(
            subprocess.check_output(
                ['python', output_file.name, iris_dataset['path'], '/dev/null']
            ).decode()
        )
    dataset_df = iris_dataset['dataframe'].compute()
    features_values = dataset_df[iris_dataset['feature_column_ids']].values
    scikit_classifier_predictions = scikit_classifier.estimator.predict(features_values)

    assert all(
            prediction == scikit_prediction \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_classifier_predictions)
        )


def test_translate_pfa_document_linear_regression(
        boston_dataset,
        empty_translator,
    ):
    scikit_regressor = linear_regression.LinearRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.python,
            output_file=output_file,
            pfa_document=scikit_regressor.to_pfa()
        )
    scoring_result = json.loads(
            subprocess.check_output(
                ['python', output_file.name, boston_dataset['path'], '/dev/null']
            ).decode()
        )
    dataset_df = boston_dataset['dataframe'].compute()
    features_values = dataset_df[boston_dataset['feature_column_ids']].values
    scikit_regressor_predictions = scikit_regressor.estimator.predict(features_values)

    assert all(
            math.isclose(prediction, scikit_prediction, rel_tol=0.001) \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_regressor_predictions)
        )


def test_translate_pfa_document_logistic_regression(
        iris_dataset,
        empty_translator,
    ):
    scikit_classifier = logistic_regression.LogisticRegressionClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.python,
            output_file=output_file,
            pfa_document=scikit_classifier.to_pfa()
        )
    scoring_result = json.loads(
            subprocess.check_output(
                ['python', output_file.name, iris_dataset['path'], '/dev/null']
            ).decode()
        )
    dataset_df = iris_dataset['dataframe'].compute()
    features_values = dataset_df[iris_dataset['feature_column_ids']].values
    scikit_classifier_predictions = scikit_classifier.estimator.predict(features_values)

    assert all(
            prediction == scikit_prediction \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_classifier_predictions)
        )

def test_translate_pfa_document_gbm_regression_c(
        boston_dataset,
        empty_translator,
        pfa_model_100_estimators
    ):
    scikit_regressor = gbm.GBMRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id'],
            method_parameters = {
                'n_estimators': 100,
                'max_depth': 6,
                'max_leaf_nodes': None,
                'min_samples_split': 2,
                'learning_rate': 0.01,
                'loss': 'ls',
                'random_state': 0
            }
        )
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.c,
            output_file=output_file,
            pfa_document=scikit_regressor.to_pfa()
        )
    exe_filename = output_file.name[:-2] + '.out'
    subprocess.call(
            ['gcc', output_file.name, '-lm', '-o', exe_filename]
        )
    scoring_result = json.loads(
            "[%s]" % subprocess.check_output(
                [exe_filename, boston_dataset['path'], '/dev/null']
            ).decode()[:-1]
        )
    os.remove(exe_filename)
    dataset_df = boston_dataset['dataframe'].compute()
    features_values = dataset_df[boston_dataset['feature_column_ids']].values
    scikit_regressor_predictions = scikit_regressor.estimator.predict(features_values)
    assert all(
            math.isclose(prediction, scikit_prediction, rel_tol=0.0001) \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_regressor_predictions)
        )


def test_translate_pfa_document_gbm_classification_c(
        iris_dataset,
        empty_translator,
    ):
    scikit_classifier = gbm.GBMClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.c,
            output_file=output_file,
            pfa_document=scikit_classifier.to_pfa()
        )
    exe_filename = output_file.name[:-2] + '.out'
    subprocess.call(
            ['gcc', output_file.name, '-lm', '-o', exe_filename]
        )
    scoring_result = json.loads(
            "[%s]" % subprocess.check_output(
                [exe_filename, iris_dataset['path'], '/dev/null']
            ).decode()[:-1]
        )
    os.remove(exe_filename)
    dataset_df = iris_dataset['dataframe'].compute()
    features_values = dataset_df[iris_dataset['feature_column_ids']].values
    scikit_classifier_predictions = scikit_classifier.estimator.predict(features_values)

    assert all(
            prediction == scikit_prediction \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_classifier_predictions)
        )


def test_translate_pfa_document_linear_regression_c(
        boston_dataset,
        empty_translator,
    ):
    scikit_regressor = linear_regression.LinearRegressor.build(
            boston_dataset['dataframe'],
            target_column_id=boston_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.c,
            output_file=output_file,
            pfa_document=scikit_regressor.to_pfa()
        )
    exe_filename = output_file.name[:-2] + '.out'
    subprocess.call(
            ['gcc', output_file.name, '-lm', '-o', exe_filename]
        )
    scoring_result = json.loads(
            "[%s]" % subprocess.check_output(
                [exe_filename, boston_dataset['path'], '/dev/null']
            ).decode()[:-1]
        )
    os.remove(exe_filename)
    dataset_df = boston_dataset['dataframe'].compute()
    features_values = dataset_df[boston_dataset['feature_column_ids']].values
    scikit_regressor_predictions = scikit_regressor.estimator.predict(features_values)

    assert all(
            math.isclose(prediction, scikit_prediction, rel_tol=0.001) \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_regressor_predictions)
        )


def test_translate_pfa_document_logistic_regression_c(
        iris_dataset,
        empty_translator,
    ):
    scikit_classifier = logistic_regression.LogisticRegressionClassifier.build(
            iris_dataset['dataframe'],
            target_column_id=iris_dataset['target_column_id']
        )
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c')
    empty_translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.c,
            output_file=output_file,
            pfa_document=scikit_classifier.to_pfa()
        )
    exe_filename = output_file.name[:-2] + '.out'
    subprocess.call(
            ['gcc', output_file.name, '-lm', '-o', exe_filename]
        )
    scoring_result = json.loads(
            "[%s]" % subprocess.check_output(
                [exe_filename, iris_dataset['path'], '/dev/null']
            ).decode()[:-1]
        )
    os.remove(exe_filename)
    dataset_df = iris_dataset['dataframe'].compute()
    features_values = dataset_df[iris_dataset['feature_column_ids']].values
    scikit_classifier_predictions = scikit_classifier.estimator.predict(features_values)

    assert all(
            prediction == scikit_prediction \
                for prediction, scikit_prediction in \
                    zip(scoring_result, scikit_classifier_predictions)
        )
