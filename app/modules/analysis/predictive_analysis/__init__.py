# encoding: utf-8

from cloudsml_computational_backend_common.analysis.consts import PredictiveAnalysisMethods
from . import (
        decision_tree,
        gbm,
        linear_regression,
        logistic_regression,
    )

PREDICTIVE_ANALYSIS_METHODS = {
        PredictiveAnalysisMethods.stepwise_linear_regression.name: \
            linear_regression.LinearRegressor.build,

        PredictiveAnalysisMethods.logit.name: \
            logistic_regression.LogisticRegressionClassifier.build,

        PredictiveAnalysisMethods.CART_regression.name: \
            decision_tree.DecisionTreeRegressor.build,
        PredictiveAnalysisMethods.CART_classification.name: \
            decision_tree.DecisionTreeClassifier.build,

        PredictiveAnalysisMethods.TN_regression.name: \
            gbm.GBMRegressor.build,
        PredictiveAnalysisMethods.TN_classification.name: \
            gbm.GBMClassifier.build,
    }
