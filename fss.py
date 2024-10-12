import pandas as pd
# import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector

# Find this dataset at https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

df = df.sample(n=10000)  # For debugging purposes

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

# Categorical Variables
nominal_vars = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'DiffWalk', 'Sex'
]

ordinal_vars = [
    'GenHlth', 'Age', 'Education', 'Income'
]

# Numerical Variables
numerical_vars = [
    'BMI', 'MentHlth', 'PhysHlth'
]

categorical_vars = ordinal_vars + nominal_vars
all_vars = categorical_vars + numerical_vars
response_var = 'Diabetes_012'

df_nominal = df[nominal_vars]
df_ordinal = df[ordinal_vars]
df_numerical = df[numerical_vars]
df_categorical = df[categorical_vars]




"""
A FSS process depends on:
1. Starting point
2. Search organization
3. Evaluation strategy
4. Stopping criterion
"""

starting_point = [
    'none', 'random', 'all'
]
search_organization = [
    'exhaustive', 'forward', 'backward', 'stepwise', 'metaheuristic'
]
evaluation_strategy = [
    'filter', 'wrapper'
]
""""
Univariate:
    Parametric methods:
        Discrete predictors:
            Mutual information Blanco et al. (2005)
            Gain ratio Hall and Smith (1998)
            Symmetrical uncertainty Hall (1999)
            Chi-squared Forman (2003)
            Odds ratio Mladenic and Grobelnik (1999)
            Bi-normal separation Forman (2003
        Continuous predictors:
            t-test family Jafari and Azuaje (2006)
            ANOVA Jafari and Azuaje (2006)
    Model-free methods:
        Threshold number of misclassification (TNoM) Ben-dor et al. (2000)
        P-metric Slonim et al. (2000)
        Mann-Whitney test Thomas et al. (2001)
        Kruskal-Wallis test Lan and Vucetic (2011)
        Between-groups to within-groups sum of squares Dudoit et al. (2002)
        Scores based on estimating density functions Inza et al. (2004)
Multivariate:
    RELIEF Kira and Rendell (1992)
    Correlation-based feature selection Hall (1999)
    Conditional mutual information Fleuret (2004)
"""
filter_options = [
    'symmetrical_uncertainty', 't-test', 'kruskal-wallis', 'conditional_mutual_information'
]
"""
Deterministic heuristics:
    Sequential feature selection Fu (1968)
    Sequential forward feature selection Fu (1968)
    Sequential backward elimination Marill and Green (1963)
    Greedy hill climbing John et al. (1994)
    Best first Xu et al. (1988)
    Plus-L-Minus-r algorithm Stearns (1976)
    Floating search selection Pudil et al. (1994)
    Tabu search Zhang and Sun (2002)
    Branch and bound Lawler and Wood (1966)
Non-deterministic heuristics:
    Single-solution metaheuristics:
        Simulated annealing Doak (1992)
        Las Vegas algorithm Liu and Motoda (1998)
        Greedy randomized adaptive search procedure Bermejo et al. (2011)
        Variable neighborhood search Garcia-Torres et al. (2005)
    Population-based metaheuristics:
        Scatter search Garcia-Lopez et al. (2006)
        Ant colony optimization Al-An (2005)
        Particle swarm optimization Lin et al. (2008)
        Evolutionary algorithms:
            Genetic algorithms Siedlecki and Sklansky (1989)
            Estimation of distribution algorithms Inza et al. (2000)
            Differential evolution Khushaba et al. (2008)
            Genetic programming Muni et al. (2004)
            Evolution strategies Vatolkin et al. (2009)
"""
wrapper_options = [
    'sequential_feature_selection', 'grasp'
]
stopping_criterion = [
    'timeout', 'performance_plateau',
    'limit_fea_8', 'limit_fea_10', 'limit_fea_12', 'limit_fea_14', 'limit_fea_16', 'limit_fea_18'
]
wrapper_models = [
    'knn', 'logistic_reg', 'ann-in-k-k-1', 'ann-in-k-k-k-1'
]

def random_selection():
    return np.random.choice([True, False], size=len(all_vars))

def get_SFS(start, end, direction, model) -> np.ndarray:
    """
    sklearn SequentialFeatureSelector(estimator, *, n_features_to_select='auto', tol=None, direction='forward', scoring=None, cv=5, n_jobs=None)
    """

    if direction not in ['forward', 'backward']:
        return np.empty(0)

    tol = None
    n_fea = 'auto'

    if end == 'performance_plateau':
        tol = 0.025
    elif end[:len('limit_features')] == 'limit_features':
        n_fea = int(end[len('limit_features')+1:])

    sfs = SequentialFeatureSelector(estimator=model, direction=direction, tol=tol, n_features_to_select=n_fea, n_jobs=-1)

    if start == 'random':
        start_subset = random_selection()
        curr_vars = np.array(all_vars)[start_subset]
        X = df[curr_vars]
        y = df[response_var]
        sfs.fit(X, y)
        selected = curr_vars[sfs.get_support()]
        result = np.array((var in selected) for var in all_vars)
    else:
        X = df[all_vars]
        y = df[response_var]
        sfs.fit(X, y)
        result = sfs.get_support()
    return result

def fss(start_fs, end_condition, dir, eval_strategy, model=None) -> np.ndarray:
    if eval_strategy == 'sequential_feature_selection':
        return get_SFS(start_fs, end_condition, dir, model)
    elif eval_strategy == 'grasp':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'symmetrical_uncertainty':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 't-test':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'kruskal-wallis':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'conditional_mutual_information':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    else:
        raise Exception("Invalid evaluator strategy found: " + eval_strategy)


for start in starting_point:
    for direction in search_organization:
        if (start == 'none' and direction == 'backward') or (start == 'all' and direction == 'forward'):
            continue

        for end in stopping_criterion:

            for evaluator in evaluation_strategy:
                if evaluator == 'filter':
                    for subevaluator in filter_options:
                        fs = fss(start, end, direction, subevaluator)
                elif evaluator == 'wrapper':
                    for subevaluator in wrapper_options:
                        for model in wrapper_models:
                            fs = fss(start, end, direction, subevaluator, model=model)
                else:
                    raise Exception("Invalid evaluation strategy found: " + evaluator)
