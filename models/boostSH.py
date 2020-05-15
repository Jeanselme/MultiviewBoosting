import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict

class BoostSH(BaseEstimator, ClassifierMixin):
    
    def __init__(self, basemodel, views, num_estimators = 10):
        """
            Boost SH : Build a adaboost classification for multiview with shared weights
            Greedy approach in which each view is tested to evaluate the one with larger
            edge
            
            Arguments:
                basemodel {sklearn model} -- Base model to use on each views
                views {Dict of pd Dataframe} -- Views to use for the task 
                    (index much match with train **and** test)
                num_estimators {int} -- Number of models to train
                learning_rate {float} --  Learning rate for the adaboost (default: .01)
        """
        super(BoostSH, self).__init__()
        self.basemodel = basemodel
        self.views = views

        self.models = []
        self.alphas = []
        self.views_selected = []

        self.num_estimators = num_estimators

    def __compute_edge__(self, data, labels, weights, edge_estimation_cv):
        """
            Train a model and compute its edge

            Arguments:
                data {pd Dataframe} -- Features - Index has to be contained in views
                labels {pd Dataframe}-- Labels - Index has to be contained in views
                weights {int} -- Number of fold used to estimate the edge 
                edge_estimation_cv {[type]} -- [description]

            Returns
                model {sklearn model} -- Basemodel trained on all data
                edge {float} -- Edge of the model (estimated on train or cv)
                forecast {pd.DataFrame} -- Forecast for each data points (train or cv)
        """
        model = clone(self.basemodel).fit(data, labels, sample_weight = weights)
        if edge_estimation_cv is None:
            forecast = model.predict(data)
        else:
            forecast = cross_val_predict(clone(self.basemodel), data, labels, \
                cv = edge_estimation_cv, fit_params = {'sample_weight': weights})
        edge = (weights * forecast * 2 * ((forecast == labels) - .5)).sum()

        return model, edge, forecast

    def fit(self, X, Y, edge_estimation_cv = None):
        """
            Fit the model by adding models in a adaboost fashion
        
            Arguments:
                X {pd Dataframe} -- Features - Index has to be contained in views
                Y {pd Dataframe} -- Labels - Index has to be contained in views
                edge_estimation_cv {int} -- Number of fold used to estimate the edge 
                    (default: None - Performance are computed on training set)
        """
        # Add training in the pool
        self.views['original'] = X
        weights = pd.Series(1, index= X.index)
        
        for i in range(self.num_estimators):
            # Normalize weights
            weights /= weights.sum()

            # For each view compute the edge
            models, edges, forecast = {}, {}, {}
            for v in self.views:
                models[v], edges[v], forecast[v] = self.__compute_edge__(self.views[v].loc[X.index], Y, weights, edge_estimation_cv)
            best_model = max(edges, key = lambda k: edges[k])
            alpha = .5 *  np.log((1 + edges[best_model]) / (1 - edges[best_model]))

            # Update weights
            weights *= np.exp(- alpha * 2 * ((forecast[v] == Y) - .5))
            
            self.models.append(models[best_model])
            self.alphas.append(alpha)
            self.views_selected.append(best_model)

        return self

    def predict_proba(self, X):
        assert len(self.models) > 0, 'Model not trained'
        predictions = None
        for m, a, v in zip(self.models, self.alphas, self.views_selected):
            if v == 'original':
                data = X
            else:
                data = self.views[v].loc[X.index]

            if predictions is None:
                predictions = pd.DataFrame(m.predict_proba(data)*a, index = X.index)
            else:
                predictions += m.predict_proba(data)*a

        return predictions.values / np.sum(self.alphas)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)