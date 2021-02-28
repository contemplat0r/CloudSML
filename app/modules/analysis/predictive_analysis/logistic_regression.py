# encoding: utf-8

from collections import OrderedDict

from sklearn import linear_model

from . import base


class LogisticRegressionClassifier(base.BaseEstimator):
    """
    Attributes:
        estimator (sklearn.linear.LogisticRegression):
            Underalying object that train model using linear regression algorithm.
        options (dict):
            Saved predictive_analysis_options.
        target_column_id (int):
            Unique id of the target column.
        feature_column_labels (list):
            Names of features columns.
    """

    def __init__(
            self,
            estimator=None,
            options=None,
            target_column_id=None,
            feature_column_labels=None
        ):
        self.estimator = estimator
        self.options = options
        self.target_column_id = target_column_id
        self.feature_column_labels = feature_column_labels

    @classmethod
    def build(cls, dask_learn_dataset_df, **predictive_analysis_options):
        """
        Args:
            learn_dataset_df (dask.dataframe):
                Learn Dataset.
            predictive_analysis_options (dict):
                kwargs to predictive analysis method.

        Returns:
            model (object): new model instance.
        """
        # TODO: handle this intelligently by offloading the workload to
        # Dask Workers. Even if we cannot compute in parallel, we should
        # distribute the load on Dask.
        learn_dataset_df = dask_learn_dataset_df.compute()

        target_column_id = predictive_analysis_options['target_column_id']
        feature_column_labels = learn_dataset_df.columns.drop([target_column_id])
        features_values = learn_dataset_df[feature_column_labels].values
        target_values = learn_dataset_df[target_column_id].values

        method_parameters = {
                'C': 0.01,
                'solver': 'lbfgs'
            }
        method_parameters.update(
                predictive_analysis_options.get('method_parameters', {})
            )
        regressor = linear_model.LogisticRegression(**method_parameters)
        regressor = regressor.fit(features_values, target_values)

        return cls(
                estimator=regressor,
                options=predictive_analysis_options,
                target_column_id=target_column_id,
                feature_column_labels=feature_column_labels
            )

    def to_pfa(self):
        """
        Convert a model to PFA.

        Returns:
            dict: PFA logistic regression model representation.
        """
        regression_coefficients = self.estimator.coef_
        intercept = self.estimator.intercept_

        return {
                'input': {
                    'type': 'record',
                    'fields': [
                        {'name': 'var%s' % feature_column_label, 'type': 'double'} \
                            for feature_column_label in self.feature_column_labels
                    ],
                },
                'output': 'double',
                'action': {
                    'a.argmax': {
                        'm.link.logit': {
                            'la.add': [
                                {
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
                                                    'new': row
                                                } for row in regression_coefficients.tolist()
                                            ]
                                        },
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
                                                    'new': ['input.var%s' % feature_column_label]
                                                } for feature_column_label in (
                                                        self.feature_column_labels
                                                    )
                                            ]
                                        }
                                    ]
                                },
                                {
                                    'type': {
                                        'type': 'array',
                                        'items': 'double'
                                    },
                                    'new': intercept.tolist()
                                }
                            ]
                        }
                    }
                }
            }

    def get_info(self, input_dataset):
        """
        Args:
            input_dataset (dask.dataframe):
                Dataset containing features and target values.

        Returns:
            info (dict): Various values that measure model peformance.
            Currently, ROC score.
        """
        # TODO: Implement distributed versions of getting performance stats.
        dataset_df = input_dataset.compute()

        features_values = dataset_df[self.feature_column_labels].values
        target_values = dataset_df[self.target_column_id].values
        return {
                'ROC': self.estimator.score(features_values, target_values)
            }
