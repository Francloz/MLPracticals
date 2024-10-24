import itertools
from xml.sax.handler import all_features

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import wittgenstein as lw


# Evaluator using bias-corrected 10-fold cross-validation
def evaluate_model(model, X, y):
    # Create a KFold cross-validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # shuffle=True helps with bias correction
    # Perform cross-validation and return the mean accuracy (or any metric)
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_weighted', 'f1_weighted': 'f1_weighted',
               'roc_auc_ovr': 'roc_auc_ovr', "cohen": make_scorer(cohen_kappa_score)}
    # make_scorer(cohen_kappa_score)
    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring,
                            return_train_score=True)  # you can change 'accuracy' to other metrics

    for (key, value) in scores.items():
        scores[key] = (np.mean(value), np.std(value))
    return scores


def evaluate(classifiers):
    # validator = sklearn.model_selection.cross_validate()
    for (key, value) in classifiers.items():
        for args in value['params']:
            model = value['model'](**args)
            X, y = df[features], df[response_var]
            scores = evaluate_model(model, X=X, y=y)

            print(f"{key}, {args}, ", end="")
            for (score, (mean, std)) in scores.items():
                print(f"{score}={mean:.3f} ± {std:.3f},", end="")
            print()


if __name__ == "__main__":
    df = pd.read_csv("csv/processed_dataset_quantile.csv")

    # df = df.sample(n=10000)  # For debugging purposes

    # Print column labels
    print("Column labels:", df.columns)
    variable_descriptions = {
        "Diabetes_012": "Diabetes status: 0 = no diabetes, 1 = prediabetes, 2 = diabetes",
        "HighBP": "High Blood Pressure: 0 = no high BP, 1 = high BP",
        "HighChol": "High Cholesterol: 0 = no high cholesterol, 1 = high cholesterol",
        "CholCheck": "Cholesterol check in past 5 years: 0 = no, 1 = yes",
        "BMI": "Body Mass Index",
        "Smoker": "Have smoked at least 100 cigarettes in lifetime: 0 = no, 1 = yes",
        "Stroke": "Ever had a stroke: 0 = no, 1 = yes",
        "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction: 0 = no, 1 = yes",
        "PhysActivity": "Physical activity in past 30 days (excluding job): 0 = no, 1 = yes",
        "Fruits": "Consume fruit 1 or more times per day: 0 = no, 1 = yes",
        "Veggies": "Consume vegetables 1 or more times per day: 0 = no, 1 = yes",
        "HvyAlcoholConsump": "Heavy drinker (men >14 drinks/week, women >7 drinks/week): 0 = no, 1 = yes",
        "AnyHealthcare": "Have any kind of health care coverage: 0 = no, 1 = yes",
        "NoDocbcCost": "Could not see doctor due to cost in past 12 months: 0 = no, 1 = yes",
        "GenHlth": "General health status: 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor",
        "MentHlth": "Number of days mental health was not good in past 30 days (1-30)",
        "PhysHlth": "Number of days physical health was not good in past 30 days (1-30)",
        "DiffWalk": "Serious difficulty walking or climbing stairs: 0 = no, 1 = yes",
        "Sex": "Sex: 0 = female, 1 = male",
        "Age": "Age category: 1 = 18-24, 9 = 60-64, 13 = 80 or older",
        "Education": "Education level: 1 = Never attended school or only kindergarten, 2 = Grades 1-8, 3 = Grades 9-11, 4 = Grade 12 or GED, 5 = Some college, 6 = College graduate",
        "Income": "Income level: 1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more"
    }

    # Print the first rows (default is 5 rows, you can specify how many with df.head(n))
    # print("\nFirst rows of the DataFrame:")
    # print(df.loc[0])

    # Categorical Variables
    nominal_vars = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
        'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
        'DiffWalk', 'Sex'
    ]

    ordinal_vars = [
        'Education', 'Income'
    ]

    # Numerical Variables
    numerical_vars = [
        'BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age'
    ]

    categorical_vars = ordinal_vars + nominal_vars

    response_var = 'Diabetes_012'

    df_nominal = df[nominal_vars]
    df_ordinal = df[ordinal_vars]
    df_numerical = df[numerical_vars]
    df_categorical = df[categorical_vars]

    features = categorical_vars + numerical_vars

    classifiers = {
        'knn': {
            'model': KNeighborsClassifier,
            'params': [{'n_neighbors': n, 'weights': weights} for (n, weights) in
                       itertools.product(range(3, 20), ['uniform', 'distance'])]
        },
        'MLP': {
            'model': MLPClassifier,
            'params': [
                {'hidden_layer_sizes': hidden_layers, 'activation': activation, 'max_iter': 5000, 'random_state': 0}
                for (hidden_layers, activation) in
                itertools.product([(15,), (32, 16), (64, 32, 16), (256, 128, 64)], ['logistic', 'relu'])]
        },
        'svm': {
            'model': SVC,
            'params': [{'C': 1.0, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'random_state': 0, 'coef0': coef0}
                       for
                       (kernel, gamma, degree, coef0) in
                       itertools.product(['sigmoid', 'linear', 'poly', 'rbf'], ['scale'],
                                         [2, 3, 4, 5, 6, 8, 10], [0.0, 0.01, 0.05, 0.1, 0.2, 0.5])]
        },
        'tree': {
            'model': DecisionTreeClassifier,
            'params': [{'criterion': criterion, 'random_state': 0, 'ccp_alpha': ccp_alpha} for (criterion, ccp_alpha) in
                       itertools.product(['gini', 'entropy', 'log_loss'], np.arange(0, 0.05, 0.0025))]
        },
        """
    Parameters
        ----------
        k : int, default=2
            Number of RIPPERk optimization iterations.
        prune_size : float, default=.33
            Proportion of training set to be used for pruning.
        dl_allowance : int, default=64
            Terminate Ruleset grow phase early if a Ruleset description length is encountered
            that is more than this amount above the lowest description length so far encountered.
        n_discretize_bins : int, default=10
            Fit apparent numeric attributes into a maximum of n_discretize_bins discrete bins, inclusive on upper part of range. Pass None to disable auto-discretization.
        random_state : int, default=None
            Random seed for repeatable results.
        verbosity : int, default=0
            Output progress, model development, and/or computation. Each level includes the information belonging to lower-value levels.
               1: Show results of each major phase
               2: Show Ruleset grow/optimization steps
               3: Show Ruleset grow/optimization calculations
               4: Show Rule grow/prune steps
               5: Show Rule grow/prune calculations
    """
        'rule': {
            'model': lw.RIPPER,
            'params': [{'k': ripperk, 'prune_size': prune_size, 'random_state': 0, 'n_discretize_bins': None, } for
                       (ripperk, prune_size) in
                       itertools.product(range(2, 10), np.arange(0.1, 0.33, 0.05), )]
        }
    }

    evaluate(classifiers)
