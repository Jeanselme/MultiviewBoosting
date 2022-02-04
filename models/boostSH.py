import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict

eps = 10**(-6)

class BoostSH(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_estimator, views, n_estimators = 10, learning_rate = 1.):
        """
            Boost SH : Build a adaboost classification for multiview with shared weights
            Greedy approach in which each view is tested to evaluate the one with larger
            edge
            
            Arguments:
                base_estimator {sklearn model} -- Base model to use on each views
                views {Dict of pd Dataframe} -- Views to use for the task 
                    (index much match with train **and** test)
                n_estimators {int} -- Number of models to train
                learning_rate {float} --  Learning rate for the adaboost (default: 1)
        """
        super(BoostSH, self).__init__()
        self.base_estimator = base_estimator
        self.views = views

        self.models = []
        self.used_classes = []
        self.alphas = []
        self.views_selected = []

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def __compute_edge__(self, data, labels, weights, edge_estimation_cv):
        """
            Train a model and compute its edge

            Arguments:
                data {pd Dataframe} -- Features - Index has to be contained in views
                labels {pd Dataframe}-- Labels - Index has to be contained in views
                weights {int} -- Number of fold used to estimate the edge 
                edge_estimation_cv {[type]} -- [description]

            Returns
                model {sklearn model} -- base_estimator trained on all data
                edge {float} -- Edge of the model (estimated on train or cv)
                forecast {pd.DataFrame} -- Forecast for each data points (train or cv)
        """
        model = clone(self.base_estimator).fit(data.values, labels.values, sample_weight = weights.values)
        if edge_estimation_cv is None:
            forecast = model.predict(data.values)
        else:
            forecast = cross_val_predict(clone(self.base_estimator), data.values, labels.values, \
                cv = edge_estimation_cv, fit_params = {'sample_weight': weights.values})
        edge = (weights * 2 * ((forecast == labels) - .5)).sum()

        return model, edge, forecast, sorted(np.unique(labels))

    def view_weights(self):
        """
            Return relative importance of the different views in the final decision
        """
        assert len(self.models) > 0, 'Model not trained'
        view_weights = pd.DataFrame({"view": self.views_selected, "alpha": np.abs(self.alphas)})
        return (view_weights.groupby('view').sum() / np.sum(np.abs(self.alphas))).sort_values('alpha')

    def fit(self, X, Y, edge_estimation_cv = None):
        """
            Fit the model by adding models in a adaboost fashion
        
            Arguments:
                X {pd Dataframe} -- Features - Index has to be contained in views
                Y {pd Dataframe} -- Labels - Index has to be contained in views
                edge_estimation_cv {int} -- Number of fold used to estimate the edge 
                    (default: None - Performance are computed on training set)
        """
        self.check_impute(X, Y)

        # Add training in the pool
        self.views['original'] = X
        self.classes = np.unique(Y)
        weights = pd.Series(1, index= X.index)
        
        for i in range(self.n_estimators):
            # Normalize weights
            weights /= weights.sum()
            if weights.sum() == 0:
                break

            # For each view compute the edge
            models, edges, forecast, classes = {}, {}, {}, {}
            for v in self.views:
                models[v], edges[v], forecast[v], classes[v] = self.__compute_edge__(self.views[v].loc[X.index], Y, weights, edge_estimation_cv)
            best_model = max(edges, key = lambda k: edges[k])
            if (1 - edges[best_model]) < eps:
                alpha = self.learning_rate * .5 * 10.
            else:
                alpha = self.learning_rate * .5 *  np.log((1 + edges[best_model]) / (1 - edges[best_model]))

            # Update weights
            weights *= np.exp(- alpha * 2 * ((forecast[v] == Y) - .5))
            
            self.models.append(models[best_model])
            self.alphas.append(alpha)
            self.views_selected.append(best_model)
            self.used_classes.append(classes[best_model])

        return self

    def predict_proba(self, X):
        self.check_impute(X)

        assert len(self.models) > 0, 'Model not trained'
        predictions = pd.DataFrame(np.zeros((len(X), len(self.classes))), index = X.index, columns = self.classes)
        for m, a, v, c in zip(self.models, self.alphas, self.views_selected, self.used_classes):
            if v == 'original':
                data = X
            else:
                data = self.views[v].loc[X.index]

            predictions += pd.DataFrame(m.predict_proba(data.values), index = data.index, columns = c)*a

        return (predictions / predictions.values.sum(axis = 1)[:, None]).fillna(-1)

    def predict(self, X):
        self.check_impute(X)

        return self.predict_proba(X).idxmax(axis = 1)

    def check_impute(self, X, Y = None):
        assert isinstance(X, pd.DataFrame), "Not right format for x"

        if Y is not None:
            assert isinstance(Y, pd.Series), "Not right format for y"
            assert len(Y.unique()) > 1, "One class in data"