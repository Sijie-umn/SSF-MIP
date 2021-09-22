
"""
Multitask Lasso Model

Details: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html


"""
from sklearn.linear_model import MultiTaskLasso, Lasso
from joblib import Parallel, delayed
import numpy as np


class LassoMultitask_wo_subx():
    """ XGBoost models for k locations
    """
    def __init__(self, alpha=1.0, num_models=197, n_jobs=16, fit_intercept=False, normalize=False, copy_X=True,
                 max_iter=1000, tol=0.0001, warm_start=False, random_state=None,
                 selection='cyclic'):

        """ Initilize a list of Lasso
        """
        super().__init__()

        self.models = [Lasso(alpha=alpha, fit_intercept=fit_intercept,
                             normalize=normalize, copy_X=copy_X,
                             max_iter=max_iter, tol=tol,
                             warm_start=warm_start, random_state=random_state,
                             selection=selection) for i in range(num_models)]
        self.n_jobs = n_jobs
        self.num_models = num_models

    def fit(self, X, y):

        """ fit XGBoost model at each location
        """
        self.models = Parallel(n_jobs=self.n_jobs)(delayed(self.models[loc].fit)(X[0], y[:, loc])
                                                   for loc in range(self.num_models))  # 197x2x num_estimators

        return self.models

    # def fit_cv(self, train_x, train_y, val_x, val_y):
    #
    #     """ For hyper-parameter tuning: fit XGBoost model at each location
    #     """
    #     history = np.asarray(Parallel(n_jobs=self.n_jobs)(delayed(self.fit_single_output)
    #                                                       (self.models[loc], np.concatenate((train_x[0], np.expand_dims(train_x[1][:, loc], axis=1)), axis=1), train_y[:, loc],
    #                                                       np.concatenate((val_x[0], np.expand_dims(val_x[1][:, loc], axis=1)), axis=1), val_y[:, loc])
    #                                                       for loc in range(self.num_models)))  # 197x2x num_estimators
    #     return history

    def predict(self, X):
        """ For XGBoost model at each location, make prediction
        """
        pred_y = np.asarray((Parallel(n_jobs=self.n_jobs)(delayed(self.models[loc].predict)(X[0])
                                                          for loc in range(self.num_models))))  # 197x2x num_estimators

        return pred_y.T

    # def fit_single_output(self, mdl, train_x, train_y, test_x, test_y):
    #     """ For hyper-parameter tuning: fit XGBoost model at each location
    #     """
    #     eval_set = [(train_x, train_y), (test_x, test_y)]
    #     eval_metric = ['rmse']
    #
    #     mdl.fit(train_x, train_y, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
    #
    #     train_eval = mdl.evals_result()['validation_0']['rmse']
    #     test_eval = mdl.evals_result()['validation_1']['rmse']
    #
    #     return train_eval, test_eval

#
# class LassoMultitask_subx(MultiTaskLasso):
#     """Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
#
#
#     """
#     def __init__(self, alpha=1.0, fit_intercept=False, normalize=False, copy_X=True,
#                  max_iter=1000, tol=0.0001, warm_start=False, random_state=None,
#                  selection='cyclic'):
#
#         """Initalize a Multitask Lasso Model.
#
#         """
#
#         super(LassoMultitask, self).__init__(alpha=alpha, fit_intercept=fit_intercept,
#                                              normalize=normalize, copy_X=copy_X,
#                                              max_iter=max_iter, tol=tol,
#                                              warm_start=warm_start, random_state=random_state,
#                                              selection=selection)
#
#     def fit(self, X, y):
#         """Fit MultiTask Lasso Model.
#         """
#
#         super(LassoMultitask, self).fit(X, y)
#         return self
#
#     def fit_cv(self, train_x, train_y, val_x, val_y):
#         """For Hyper-parameter Tuning: Fit MultiTask Lasso Model.
#         """
#
#         self.fit(train_x, train_y)
#         # pred_y = self.predict(val_x)
#
#         return self.predict(train_x), self.predict(val_x)
#
#     def predict(self, X):
#         """Predict using trained Multitask Model.
#         """
#
#         return super(LassoMultitask, self).predict(X)
