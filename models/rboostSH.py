from models.boostSH import BoostSH

import pandas as pd
import numpy as np

class RBoostSH(BoostSH):
    
    def __init__(self, basemodel, views, num_estimators = 10, sigma = 0.15, gamma = 0.3):
        """
            rBoost SH : Build a adaboost classification for multiview with shared weights
            Multi Arm Bandit approach in which a view is selected 
            
            Arguments:
                model {sklearn model} -- Base model to use on each views
                views {Dict of pd Dataframe} -- Views to use for the task 
                    (index much match with train **and** test)
                num_estimators {int} -- Number of models to train
                sigma {float} -- Used for theoretical guarantee
                gamma {float} -- Same
        """
        super(RBoostSH, self).__init__(basemodel, views, num_estimators)
        self.sigma = sigma
        self.gamma = gamma

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
        views = list(self.views.keys())
        weights = pd.Series(1, index= X.index)
        M = len(views)

        p_views = pd.Series(np.exp(self.sigma * self.gamma / 3 * \
            np.sqrt(self.num_estimators / M)),
            index = views)
        
        for i in range(self.num_estimators):
            # Normalize weights
            weights /= weights.sum()

            # Bandit selection of best view
            q_views = (1 - self.gamma) * p_views / p_views.sum() + self.gamma / M
            selected_view = np.random.choice(views, p = q_views)

            # Training model
            model, edge, forecast = self.__compute_edge__(self.views[selected_view].loc[X.index], Y, weights, edge_estimation_cv)
            alpha = .5 *  np.log((1 + edge) / (1 - edge))

            # Update weights
            weights *= np.exp(- alpha * 2 * ((forecast == Y) - .5))

            # Update arm probability
            r_views = pd.Series(0, index = views)
            r_views[selected_view] = (1 - np.sqrt(1 - edge ** 2)) / q_views[selected_view]
            p_views *= np.exp(self.gamma / (3*M) * r_views + \
                self.sigma / (q_views * np.sqrt(self.num_estimators / M)))
            
            self.models.append(model)
            self.alphas.append(alpha)
            self.views_selected.append(selected_view)

        return self