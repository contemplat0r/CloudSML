# encoding: utf-8

from collections import OrderedDict
import heapq
import math
import operator

import sklearn
from sklearn.tree._tree import TREE_UNDEFINED
from sklearn import ensemble
from sklearn.ensemble.partial_dependence import partial_dependence

from . import base


class BaseGBM(base.BaseEstimator):
    """
    Attributes:
        estimator (sklearn.ensemble.GradientBoostingClassifier):
            Underalying object that train model using CART algorithm.
        options (dict):
            Saved predictive_analysis_options.
        target_column_id (string):
            Unique ID of the target column.
        feature_column_labels (list):
            Names of features columns.
        init_prediction (float):
            The initial value of a decision function.
    """
    ESTIMATOR = lambda *args, **kwargs: None
    PARTIAL_DEPENDENCE_FEATURES_COUNT = 5

    def __init__(
            self,
            estimator,
            options,
            target_column_id,
            feature_column_labels,
            init_prediction
        ):
        """
        NOTE: Use ``.build`` class method instead of manual class instantiation.
        """
        self.estimator = estimator
        self.options = options
        self.target_column_id = target_column_id
        self.feature_column_labels = feature_column_labels
        self.init_prediction = init_prediction

    @classmethod
    def build(cls, dask_learn_dataset_df, **predictive_analysis_options):
        """
        Args:
            learn_dataset_df (dask.dataframe):
                Dataset.
            predictive_analysis_options (dict):
                kwargs to predictive analysis method.

        Returns:
            model (object): new model instance.
        """
        learn_dataset_df = dask_learn_dataset_df.compute()
        target_column_id = predictive_analysis_options['target_column_id']
        feature_column_labels = learn_dataset_df.columns.drop([target_column_id])
        features_values = learn_dataset_df[feature_column_labels].values
        target_values = learn_dataset_df[target_column_id].values

        method_parameters = {
                'n_estimators': 100,
                'max_leaf_nodes': 32,
                'min_samples_leaf': 1,
                'learning_rate': 0.01,
                'random_state': 0,
            }
        method_parameters.update(
                predictive_analysis_options.get('method_parameters', {})
            )
        if method_parameters['max_leaf_nodes'] and 'max_depth' not in method_parameters:
            method_parameters['max_depth'] = math.ceil(
                    math.log2(method_parameters['max_leaf_nodes'])
            )
        estimator = cls.ESTIMATOR(**method_parameters)
        estimator = estimator.fit(features_values, target_values)

        return cls(
                estimator=estimator,
                options=predictive_analysis_options,
                target_column_id=target_column_id,
                feature_column_labels=feature_column_labels,
                # pylint: disable=protected-access
                init_prediction=estimator._init_decision_function(features_values)[0]
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

    def _get_partial_dependence_for_column(
            self,
            features_values,
            target_feature_column_id,
            grid_resolution
        ):
        """
        Args:
            features_values (numpy.ndarray): features matrix.
            target_feature_column_id (int): a unique id of a feature
                column which is targeted for partial dependence.
            grid_resolution (int): the number of equally spaced points on the grid.

        Returns:
            (feature_axes, outputs): Where ``feature_axes`` is a list of
            floats or strings, and ``outputs`` is a dictionary of ``feature_id``,
            ``output`` and other extra fields.
        """
        get_feature_index = self.feature_column_labels.get_loc
        target_feature_column = self.options['columns_info'][target_feature_column_id]
        number_of_trees_per_stage = self.estimator.estimators_.shape[1]
        if target_feature_column_id in self.options['categorical_column_ids']:
            feature_axes = []
            partial_dependeces_per_column = []
            for feature_column in target_feature_column.virtual_columns:
                partial_dependence_matrix, virtual_feature_axes = partial_dependence(
                        self.estimator,
                        (get_feature_index(feature_column.label),),
                        X=features_values,
                        percentiles=(0, 1),
                        grid_resolution=2
                    )
                if len(virtual_feature_axes[0]) == 2:
                    assert all(virtual_feature_axes[0] == (0.0, 1.0))
                    partial_dependence_for_virtual_column = partial_dependence_matrix[:, 1]
                elif len(virtual_feature_axes[0]) == 1:
                    if all(virtual_feature_axes[0] == 1.0):
                        partial_dependence_for_virtual_column = partial_dependence_matrix[:, 0]
                    else:
                        assert all(virtual_feature_axes[0] == 0.0)
                        partial_dependence_for_virtual_column = [0] * number_of_trees_per_stage
                else:
                    assert False, (
                            "Partial dependence matrix must contain one or two columns, but "
                            "%d columns found for '%s' (#%d) feature: %r" % (
                                    partial_dependence_matrix.shape[1],
                                    feature_column.label,
                                    target_feature_column_id,
                                    partial_dependence_matrix
                                )
                        )

                feature_axes.append(feature_column.category_value)
                assert len(partial_dependence_for_virtual_column) == number_of_trees_per_stage
                partial_dependeces_per_column.append(partial_dependence_for_virtual_column)
            return (
                    feature_axes,
                    [
                        {
                            'feature_id': target_feature_column_id,
                            'output': [
                                partial_dependence[i]
                                    for partial_dependence in partial_dependeces_per_column
                            ]
                        } for i in range(len(partial_dependeces_per_column[0]))
                    ]
                )
        else:
            partial_dependence_matrix, feature_axes = partial_dependence(
                    self.estimator,
                    (get_feature_index(target_feature_column_id),),
                    X=features_values,
                    grid_resolution=grid_resolution
                )
            return (
                    feature_axes[0],
                    [
                        {
                            'feature_id': target_feature_column_id,
                            'output': partial_dependence
                        } for partial_dependence in partial_dependence_matrix
                    ]
                )

    def _get_partial_dependence(
            self,
            features_values,
            target_feature_column_ids,
            grid_resolution=50
        ):
        restructured_output = {'feature_axes': {}, 'outputs': []}
        for target_feature_column_id in target_feature_column_ids:
            feature_axes, outputs = self._get_partial_dependence_for_column(
                    features_values,
                    target_feature_column_id,
                    grid_resolution
                )
            restructured_output['feature_axes'][target_feature_column_id] = feature_axes
            restructured_output['outputs'] += outputs
        return restructured_output

    def _tree_to_pfa(self, tree):
        """
        Args:
            tree (sklearn.tree._tree.Tree):
                scikit-learn object that contains all information for export
                decision tree.

        Returns (OrderedDict):
            PFA regression tree representation.
        """
        tree_ = tree.tree_

        def recurse(node):
            if tree_.feature[node] != TREE_UNDEFINED:
                name = self.feature_column_labels[tree_.feature[node]]
                threshold = tree_.threshold[node]
                return OrderedDict([
                        ('if', {'<=': ['input.var%s' % name, threshold]}),
                        ('then', recurse(tree_.children_left[node])),
                        ('else', recurse(tree_.children_right[node]))
                    ])
            else:
                return {'double': tree_.value[node][0][0]}

        return recurse(0)


class GBMClassifier(BaseGBM):
    ESTIMATOR = ensemble.GradientBoostingClassifier

    def to_pfa(self):
        """
        Convert to PFA.

        Returns:
            dict: PFA gradient boosting trees ensemble model representation.
        """
        estimators = self.estimator.estimators_

        pfa_estimators = []
        for i in range(len(estimators[0])):
            pfa_estimators.append(
                {
                    'a.sum': {
                        'type': {'type': 'array', 'items': 'double'},
                        'new': []
                    }
                }
            )
            for tree_array in estimators:
                pfa_estimators[i]['a.sum']['new'].append(self._tree_to_pfa(tree_array[i]))
            pfa_estimators[i]['a.sum']['new'].insert(0, self.init_prediction[i])

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
                    'type': {'type': 'array', 'items': 'double'},
                    'new': pfa_estimators,
                }
            }
        }

    def _compute_staged_performance_stats(self, features_values, target_values):
        """
        Args:
            features_values (numpy.ndarray):
                Features values for computing stats.
            target_values (numpy.ndarray):
                Target values for computing stats.

        Returns:
            performance_stats (dict):
            Various values that measure model peformance. Now MSE and R^2 score,
            and staged MSE and R^2 scores.
        """
        performance_stats = []

        prediction = self.estimator.staged_predict(features_values)
        for i, prediction in enumerate(
                self.estimator.staged_predict(features_values),
                start=1
            ):
            performance_stats.append(
                {
                    'estimators_count': i,
                    'ROC': sklearn.metrics.accuracy_score(
                        target_values,
                        prediction
                    )
                }
            )
        return {
            'estimators_count': self.estimator.estimators_.shape[0],
            'performance_stats': performance_stats
        }

    def _get_partial_dependence_for_column(
            self,
            features_values,
            target_feature_column_id,
            grid_resolution
        ):
        predictor_values, outputs = super(GBMClassifier, self)._get_partial_dependence_for_column(
                features_values,
                target_feature_column_id,
                grid_resolution
            )
        assert len(outputs) == self.estimator.estimators_.shape[1]
        uniques_stats = (
                self.options['columns_info'][self.target_column_id].statistics['uniques_stats']
            )
        if self.estimator.estimators_.shape[1] == 1:
            assert len(uniques_stats) == 2, (
                    "It seems that the binary classification was built, but we have more than 2 "
                    "unique values in this columns"
                )
            outputs[0]['target_class'] = uniques_stats[1][0]
        else:
            for unique_value, output in zip(uniques_stats, outputs):
                output['target_class'] = unique_value[0]

        return predictor_values, outputs


    def get_info(self, input_dataset):
        """
        Args:
            input_dataset (dask.dataframe):
                Dataset containing features and target values.

        Returns:
            info (dict): Various values that measure model peformance.
        """
        input_dataset_df = input_dataset.compute()
        features_values = input_dataset_df[self.feature_column_labels].values
        target_values = input_dataset_df[self.target_column_id].values
        staged_performance_stats = self._compute_staged_performance_stats(
                features_values,
                target_values
            )
        last_staged_performance_stats = staged_performance_stats['performance_stats'][-1]
        variable_importance = self._get_variable_importance()
        return {
                'ROC': last_staged_performance_stats['ROC'],
                'staged_performance_stats': staged_performance_stats,
                'variable_importance': variable_importance,
                'partial_dependence': self._get_partial_dependence(
                        features_values,
                        [
                            column_id for column_id, _ in heapq.nlargest(
                                    self.PARTIAL_DEPENDENCE_FEATURES_COUNT,
                                    variable_importance.items(),
                                    key=operator.itemgetter(1)
                                )
                        ]
                    )
            }


class GBMRegressor(BaseGBM):
    ESTIMATOR = ensemble.GradientBoostingRegressor

    def to_pfa(self):
        """
        Convert to PFA.

        Returns:
            PFA (dict): PFA gradient boosting trees ensemble representation.
        """
        learning_rate = self.estimator.learning_rate
        estimators = self.estimator.estimators_
        return {
            'input': {
                'type': 'record',
                'fields': [
                    {'name': 'var%s' % feature_column_label, 'type': 'double'} \
                        for feature_column_label in self.feature_column_labels
                ]
            },
            'output': 'double',
            'action': {
                '+': [
                    self.init_prediction[0],
                    {
                        '*': [
                            learning_rate,
                            {
                                'a.sum': {
                                    'type': {'type': 'array', 'items': 'double'},
                                    'new': [self._tree_to_pfa(tree_array[0]) \
                                        for tree_array in estimators]
                                }
                            }
                        ]
                    }
                ]
            }
        }

    def _compute_staged_performance_stats(self, features_values, target_values):
        """
        Args:
            features_values (numpy.ndarray):
                Features values for computing stats.
            target_values (numpy.ndarray):
                Target values for computing stats.

        Returns:
            dict: staged performance stats
        """
        performance_stats = []

        residual_sum_of_squares = ((target_values - target_values.mean()) ** 2).sum()

        for i, prediction in enumerate(
                self.estimator.staged_predict(features_values),
                start=1
            ):
            performance_stats.append(
                {
                    'estimators_count': i,
                    'MSE': sklearn.metrics.mean_squared_error(
                            target_values,
                            prediction
                        ),
                    'R_squared': (
                            1 - ((target_values - prediction) ** 2).sum() / residual_sum_of_squares
                        )
                }
            )
        return {
            'estimators_count': self.estimator.estimators_.shape[0],
            'performance_stats': performance_stats
        }

    def get_info(self, input_dataset):
        """
        Args:
            input_dataset (dask.dataframe):
                Dataset containing features and target values.

        Returns:
            info (dict): Various values that measure model peformance.
        """
        input_dataset_df = input_dataset.compute()
        features_values = input_dataset_df[self.feature_column_labels].values
        target_values = input_dataset_df[self.target_column_id].values
        staged_performance_stats = self._compute_staged_performance_stats(
                features_values,
                target_values
            )
        last_staged_performance_stats = staged_performance_stats['performance_stats'][-1]
        variable_importance = self._get_variable_importance()
        return {
                'MSE': last_staged_performance_stats['MSE'],
                'R_squared': last_staged_performance_stats['R_squared'],
                'staged_performance_stats': staged_performance_stats,
                'variable_importance': variable_importance,
                'partial_dependence': self._get_partial_dependence(
                        features_values,
                        [
                            column_id for column_id, _ in heapq.nlargest(
                                    self.PARTIAL_DEPENDENCE_FEATURES_COUNT,
                                    variable_importance.items(),
                                    key=operator.itemgetter(1)
                                )
                        ]
                    )
            }
