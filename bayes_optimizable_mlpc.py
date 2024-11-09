import pandas as pd
from sklearn.neural_network import MLPClassifier


class BayesOptimizableMLPC(MLPClassifier):
    def __init__(self, hidden_layer_sizes="50", activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                 n_iter_no_change=10, max_fun=15000):
        # Convert string of hidden layer sizes to a tuple of integers
        if isinstance(hidden_layer_sizes, str):
            hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                         batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                         power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose,
                         warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
                         max_fun=max_fun)

    def set_params(self, **params):
        if 'hidden_layer_sizes' in params:
            hidden_layers = params['hidden_layer_sizes']
            if isinstance(hidden_layers, str):
                params['hidden_layer_sizes'] = tuple(map(int, hidden_layers.split(',')))
        return super().set_params(**params)


if __name__ == '__main__':
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Real, Integer

    df = pd.read_csv("csv/outlier_filtered.csv")

    response_var = 'Diabetes_012'
    features = list(df.columns)
    features.remove(response_var)

    print(features, response_var)
    X, y = df[features], df[response_var]

    rng = 1
    model = BayesSearchCV(
        BayesOptimizableMLPC(max_iter=1000),
        search_spaces={
            'hidden_layer_sizes': Categorical(["50", "100", "50, 50", "100, 50", "100, 100", "50, 100, 50"]),
            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
            'solver': Categorical(['lbfgs', 'sgd', 'adam']),
            'alpha': Real(1e-4, 1e-2, prior='log-uniform'),  # L2 regularization term
            'batch_size': Integer(50, 200),  # Size of minibatches for stochastic optimizers
            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),  # Initial learning rate
            'power_t': Real(0.1, 1.0),  # Only used with 'invscaling' learning rate
            'shuffle': Categorical([True, False]),  # Shuffle samples each iteration
            'tol': Real(1e-5, 1e-3, prior='log-uniform'),  # Tolerance for stopping criteria
            'momentum': Real(0.5, 0.99),  # Momentum for 'sgd'
            'nesterovs_momentum': Categorical([True, False]),  # Only used when 'sgd' and momentum > 0
            'early_stopping': Categorical([True, False]),  # Early stopping with validation split
            'validation_fraction': Real(0.1, 0.3),  # Validation fraction if early stopping is True
            'beta_1': Real(0.8, 0.99),  # First moment decay rate in adam
            'beta_2': Real(0.9, 0.999),  # Second moment decay rate in adam
            'epsilon': Real(1e-8, 1e-7, prior='log-uniform'),  # Stability in adam
        },
        random_state=rng,
        scoring='f1_macro',
        n_jobs=-1,
        n_points=2,
    )
    model.fit(X, y)
