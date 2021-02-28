# encoding: utf-8

import sklearn
from sklearn import linear_model

from . import base


class LinearRegressor(base.BaseEstimator):
    """
    Attributes:
        estimator (sklearn.linear.LinearRegression):
            Underalying object that train model using linear regression algorithm.
        options (dict):
            Saved predictive_analysis_options.
        target_column_id (int):
            Unique ID of the target column.
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

        method_parameters = {
                'normalize': False,
            }
        method_parameters.update(
                predictive_analysis_options.get('method_parameters', {})
            )
        regressor = linear_model.LinearRegression(**method_parameters)

        target_column_id = predictive_analysis_options['target_column_id']
        feature_column_labels = learn_dataset_df.columns.drop([target_column_id])
        regressor = regressor.fit(
                learn_dataset_df[feature_column_labels].values,
                learn_dataset_df[target_column_id].values
            )

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
            dict: PFA linear regression representation.
        """

        linear_regression_coefficients = self.estimator.coef_
        intercept = self.estimator.intercept_

        return {
                'input': {
                    'type': 'record',
                    'fields': [
                        {
                            'name': 'var%s' % feature_column_label,
                            'type': 'double'
                        } for feature_column_label in self.feature_column_labels
                    ]
                },
                'output': 'double',
                'action': {
                    '+': [
                        {
                            'm.kernel.linear': [
                                {
                                    'type': {
                                        'type': 'array',
                                        'items': 'double'
                                    },
                                    'new' : linear_regression_coefficients.tolist(),
                                },
                                {
                                    'type': {
                                        'type': 'array',
                                        'items': 'double'
                                    },
                                    'new': [
                                        'input.var%s' % feature_column_label \
                                            for feature_column_label in self.feature_column_labels
                                    ]
                                }
                            ]
                        },
                        intercept
                    ]
                }
            }

    def get_info(self, input_dataset):
        """
        Args:
            input_dataset (dask.dataframe):
                Dataset containing features and target values.

        Returns:
            info (dict): Various values that measure model peformance.
            Currently, only MSE and R^2 score.
        """
        # TODO: Implement distributed versions of getting performance stats.
        dataset_df = input_dataset.compute()

        features_values = dataset_df[self.feature_column_labels].values
        target_values = dataset_df[self.target_column_id].values
        return {
                'MSE': sklearn.metrics.mean_squared_error(
                        target_values,
                        self.estimator.predict(features_values)
                    ),
                'R_squared': self.estimator.score(features_values, target_values)
            }
