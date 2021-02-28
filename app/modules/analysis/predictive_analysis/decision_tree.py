# encoding: utf-8

from collections import OrderedDict
import numpy

import sklearn
from sklearn import tree as decision_tree
from sklearn.tree._tree import TREE_UNDEFINED

from . import base


class BaseDecisionTree(base.BaseEstimator):
    """
    Attributes:
        estimator (sklearn.tree.DecisionTreeClassifier):
            Underalying object that train model using CART algorithm.
        options (dict):
            Saved predictive_analysis_options.
        target_column_id (int):
            Unique ID of the target column.
        feature_column_labels (list):
            Names of feature columns.
    """
    ESTIMATOR = lambda *args, **kwags: None

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
            dask_learn_dataset_df (dask.dataframe):
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
        feature_column_labels = learn_dataset_df\
            .columns.drop([target_column_id])
        features_values = learn_dataset_df[feature_column_labels].values
        target_values = learn_dataset_df[target_column_id].values

        method_parameters = {
                'min_samples_split': 10,
                'min_samples_leaf': 1,
                'max_depth': None,
                'max_leaf_nodes': None,
                'random_state': 0,
            }
        method_parameters.update(
                predictive_analysis_options.get('method_parameters', {})
            )
        estimator = cls.ESTIMATOR(**method_parameters)
        estimator = estimator.fit(features_values, target_values)

        return cls(
                estimator=estimator,
                options=predictive_analysis_options,
                target_column_id=target_column_id,
                feature_column_labels=feature_column_labels
            )

    def _get_variable_importance_for_column(self, feature_importances, feature_column_id):
        """
        Args:
            feature_importances (numpy.array):
                A full list of feature importances.
            feature_column_id (int):
                A unique feature id.

        Returns:
            float: an importance of a gived feature in %.
        """
        feature_column = self.options['columns_info'][feature_column_id]
        get_feature_column_index = self.feature_column_labels.get_loc
        if hasattr(feature_column, 'virtual_columns'):
            feature_column_labels = (column.label for column in feature_column.virtual_columns)
        else:
            feature_column_labels = [feature_column.id]
        return feature_importances[[
                get_feature_column_index(feature_column_label) \
                    for feature_column_label in feature_column_labels
            ]].sum() * 100.0

    def _get_variable_importance(self):
        feature_importances = self.estimator.feature_importances_
        return {
                feature_column_id: self._get_variable_importance_for_column(
                        feature_importances,
                        feature_column_id
                    ) for feature_column_id in self.options['feature_column_ids']
            }

    def to_pfa(self):
        """
        Convert a model to PFA.

        Returns:
            PFA (OrderedDict): PFA classification tree representation.
        """
        tree = self.estimator.tree_

        def recurse(node):
            if tree.feature[node] != TREE_UNDEFINED:
                name = self.feature_column_labels[tree.feature[node]]
                threshold = tree.threshold[node]
                return OrderedDict([
                        ('if', {'<=': ['input.var%s' % name, threshold]}),
                        ('then', recurse(tree.children_left[node])),
                        ('else', recurse(tree.children_right[node]))
                    ])
            else:
                return {'double': numpy.argmax(tree.value[node])}

        return {
                'input': {
                        'type': 'record',
                        'fields': [
                                {'name': 'var%s' % feature_column_label, 'type': 'double'} \
                                    for feature_column_label in self.feature_column_labels
                            ],
                    },
                'output': 'double',
                'action': recurse(0)
            }


class DecisionTreeClassifier(BaseDecisionTree):
    ESTIMATOR = decision_tree.DecisionTreeClassifier

    def get_info(self, input_dataset):
        """
        Args:
            input_dataset (dask.dataframe):
                Dataset containing features and target values.

        Returns:
            info (dict): Various values that measure model peformance.
            Currently, only ROC score.
        """
        # TODO: Implement distributed versions of getting performance stats.
        dataset_df = input_dataset.compute()

        features_values = dataset_df[self.feature_column_labels].values
        target_values = dataset_df[self.target_column_id].values

        return {
                'ROC': self.estimator.score(features_values, target_values),
                'variable_importance': self._get_variable_importance(),
            }


class DecisionTreeRegressor(BaseDecisionTree):
    ESTIMATOR = decision_tree.DecisionTreeRegressor

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
                'R_squared': self.estimator.score(features_values, target_values),
                'variable_importance': self._get_variable_importance(),
            }
