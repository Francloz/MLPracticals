from multiprocessing.managers import Value

import pandas as pd
# import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def random_selection(n):
    return np.random.choice([True, False], size=n)


def get_SFS(start, end, direction, model, df, features, target) -> np.ndarray:
    """
    sklearn SequentialFeatureSelector(estimator, *, n_features_to_select='auto', tol=None, direction='forward', scoring=None, cv=5, n_jobs=None)
    """

    if direction not in ['forward', 'backward']:
        return np.empty(0)

    tol = None
    n_fea = 'auto'

    if end == 'performance_plateau':
        tol = 0.025
    elif end[:len('limit_fea')] == 'limit_fea':
        n_fea = int(end[len('limit_fea') + 1:])

    result = None
    if start == 'random':
        return np.zeros(1)
        # start_subset = np.ones(len(features))
        # while (1 if direction == 'forward' else -1)*(n_fea-np.sum(start_subset)) >= 0:
        #     start_subset = random_selection(len(features))
        #
        # curr_vars = np.array(features)[start_subset]
        # X = df[curr_vars]
        # y = df[target]
        # sfs = SequentialFeatureSelector(estimator=model, direction=direction, tol=tol, n_features_to_select=n_fea-np.sum(start_subset),
        #                                 n_jobs=-1)
        # sfs.fit(X, y)
        # selected = curr_vars[sfs.get_support()]
        # mask = list((var in selected) for var in features)
        # result = np.array(mask)
    else:
        sfs = SequentialFeatureSelector(estimator=model, direction=direction, tol=tol, n_features_to_select=n_fea,
                                        n_jobs=-1)
        X = df[features]
        y = df[target]
        sfs.fit(X, y)
        result = sfs.get_support()

    assert (np.sum(result) == n_fea)
    return result


def get_fss_filter(start, end, direction, df, features, target, metric='mutual_information', mode='univariate'):
    if end == 'timeout' or direction == 'exhaustive':
        return np.zeros(1)

    if end == 'performance_plateau':
        return np.zeros(1)

    if end == 'performance_plateau':
        return np.zeros(1)

    elif end[:len('limit_fea')] == 'limit_fea':
        n_fea = int(end[len('limit_fea') + 1:])
        assert (n_fea < len(features))

        if start == 'random':
            subset = random_selection(len(features))
            while (1 if direction == 'forward' else -1) * (n_fea - np.sum(subset)) <= 0:
                subset = random_selection(len(features))
        elif start == 'none':
            subset = np.zeros(len(features), dtype=bool)
        else:  # start == 'all':
            subset = np.ones(len(features), dtype=bool)

        if metric == 'mutual_information':
            scores = mutual_info_classif(df[features], df[target])
        elif metric == 'f_val':
            scores = -f_classif(df[features], df[target])[1]
        elif metric == 'chi2':
            scores = -chi2(df[features], df[target])[1]
        else:
            raise ValueError("Invalid metric used for filter FSS: " + metric)

        initial_subset = np.copy(subset)

        if direction == 'forward':
            to_select = n_fea - np.sum(subset)
            masked_scores = np.ma.masked_array(scores,  subset)
            indices = np.ma.masked_array.argsort(masked_scores, fill_value=-np.infty)
            # sorted_scores = scores[indices]
            kbest = indices[-to_select:]
            subset[kbest] = True

            if np.sum(subset) != n_fea:
                x = 0
        elif direction == 'backward':
            to_select = np.sum(subset) - n_fea
            masked_scores = np.ma.masked_array(scores, np.logical_not(subset))
            indices = np.ma.masked_array.argsort(masked_scores, fill_value=np.infty)
            # sorted_scores = scores[indices]

            kworst = indices[:to_select]

            if np.sum(subset) - len(indices[:to_select]) != n_fea:
                x = 0

            subset[kworst] = False

            if np.sum(subset) != n_fea:
                x = 0
        else:
            return np.zeros(1)

        assert (np.sum(subset) == n_fea)
    return subset


def prob_equal(arrs, values):
    eqs = [arr == val for arr, val in zip(arrs, values)]
    ret = np.ones(len(arrs[0]), dtype=np.bool_)

    for eq in eqs:
        ret = np.logical_and(ret, eq)

    count = np.sum(ret)
    return np.float64(count) / len(arrs[0])


def cmi(X_i, Y, X_j):
    Xis = np.unique(X_i)
    Ys = np.unique(Y)
    Xjs = np.unique(X_j)

    cmi_val = 0
    for xj in Xjs:
        p_z = prob_equal((X_j,), (xj,))
        for xi in Xis:
            p_xz = prob_equal((X_j, X_i), (xj, xi))
            if p_xz == 0:
                continue

            for y in Ys:
                p_yz = prob_equal((Y, X_j), (y, xj))
                p_xyz = prob_equal((X_i, Y, X_j), (xi, y, xj))

                if p_yz * p_xyz == 0:
                    continue

                log_bdy = (p_z * p_xyz) / (p_xz * p_yz)
                cmi_val += p_xyz * np.log(log_bdy)
    return cmi_val


def multivariate_cmi(start, end, direction, df, features, target):
    if end[:len('limit_fea')] == 'limit_fea':
        n_fea = int(end[len('limit_fea') + 1:])
    elif end == 'perf_plateau':
        raise ValueError(f"Invalid end {end}")
    else:
        raise ValueError(f"Invalid end {end}")

    if start == 'random':
        subset = random_selection(len(features))
        while (1 if direction == 'forward' else -1) * (n_fea - np.sum(subset)) < 0:
            subset = random_selection(len(features))
    elif start == 'none':
        subset = np.zeros(len(features), dtype=bool)
    else:  # start == 'all':
        subset = np.ones(len(features), dtype=bool)

    if direction == "forward":
        k_fea = n_fea - np.sum(subset)
        for _ in range(k_fea):
            best_i = -1
            best = 0

            for i in range(len(subset)):
                if subset[i]:
                    continue

                min_cmi = np.inf

                if not np.any(subset):
                    mi = cmi(df[features[i]].values, df[target].values, np.ones(df[target].values.shape))
                    if best < mi:
                        best = mi
                        best_i = i
                else:
                    for j in range(len(subset)):
                        if not subset[j]:
                            continue

                        min_cmi = min(min_cmi, cmi(df[features[i]].values, df[target].values, df[features[j]].values))

                    if best < min_cmi:
                        best = min_cmi
                        best_i = i

            subset[best_i] = True

    elif direction == "backward":
        k_fea = - n_fea + np.sum(subset)
        for _ in range(k_fea):
            best_i = -1
            best = np.inf

            for i in range(len(subset)):
                if not subset[i]:
                    continue

                min_cmi = -np.inf

                if subset.all():
                    mi = cmi(df[features[i]].values, df[target].values, np.ones(df[target].values.shape))
                    if best > mi:
                        best = mi
                        best_i = i
                else:
                    for j in range(len(subset)):
                        if subset[j]:
                            continue

                        min_cmi = max(min_cmi, cmi(df[features[i]].values, df[target].values, df[features[j]].values))

                    if best > min_cmi:
                        best = min_cmi
                        best_i = i

            subset[best_i] = False
    else:
        raise ValueError(f"Invalid direction {direction}")
    assert(n_fea == np.sum(subset))
    return subset


def fss(start_fs, end_condition, dir, eval_strategy, df, features, target, model=None) -> np.ndarray:
    if eval_strategy == 'sequential_feature_selection':
        return get_SFS(start_fs, end_condition, dir, model, df=df, features=features, target=target)
    elif eval_strategy == 'grasp':
        return np.empty(1)  # raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'symmetrical_uncertainty':
        return np.empty(1)  # raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 't-test':
        return np.empty(1)  # raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'kruskal-wallis':
        return np.empty(1)  # raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'mutual_information':
        return get_fss_filter(start_fs, end_condition, dir, metric='mutual_information', df=df, features=features,
                              target=target)
    elif eval_strategy == 'chi2':
        return get_fss_filter(start_fs, end_condition, dir, metric='chi2', df=df, features=features, target=target)
    elif eval_strategy == 'f_val':
        return get_fss_filter(start_fs, end_condition, dir, metric='f_val', df=df, features=features, target=target)
    elif eval_strategy == 'multivariate_cmi':
        return multivariate_cmi(start_fs, end_condition, dir, df=df, features=features, target=target)
    else:
        raise Exception("Invalid evaluator strategy found: " + eval_strategy)


if __name__ == "__main__":
    # Find this dataset at https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
    df = pd.read_csv("csv/diabetes_012_health_indicators_BRFSS2015.csv")

    df = df.sample(n=50000)  # For debugging purposes

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
        "Education": "Education level: 1 = Never attended school or only kindergarten, 2 = Grades 1-8, 3 = Grades "
                     "9-11, 4 = Grade 12 or GED, 5 = Some college, 6 = College graduate",
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
        'all', 'none', # 'random',
    ]
    search_organization = [
        'exhaustive', 'forward', 'backward'  # , 'stepwise', 'metaheuristic'
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
        KNeighborsClassifier(10, weights='distance'),
        LogisticRegression(random_state=0),
        MLPClassifier(solver='adam', alpha=1e-5,
                      hidden_layer_sizes=(32), random_state=1),
        MLPClassifier(solver='adam', alpha=1e-5,
                      hidden_layer_sizes=(32, 32), random_state=1),
        MLPClassifier(solver='adam', alpha=1e-5,
                      hidden_layer_sizes=(32, 64, 32), random_state=1),
        MLPClassifier(solver='adam', alpha=1e-5,
                      hidden_layer_sizes=(32, 64, 64, 32), random_state=1)
    ]

    # evaluation_strategy = ['filter']
    # filter_options = ['chi2', 'mutual_information']

    print(
        f"start,end,direction,subevaluator,model,{str(all_vars)[1:-1]}")
    for start in starting_point:
        for direction in search_organization:
            if (start == 'none' and direction == 'backward') or (start == 'all' and direction == 'forward'):
                continue

            for end in stopping_criterion:

                for evaluator in evaluation_strategy:
                    if evaluator == 'filter':
                        for subevaluator in filter_options:
                            fs = fss(start, end, direction, subevaluator, df=df, features=all_vars, target=response_var)
                            if len(fs) > 1:
                                # print(
                                #     f"Selection starting with {start} till {end}, moving {direction}, with evaluation {subevaluator} is {np.sort(np.array(all_vars)[fs])}")
                                print(
                                    f"{start},{end},{direction},{evaluator},{subevaluator},{np.array2string(fs.astype(int), separator=',')[1:-1]}")
                    elif evaluator == 'wrapper':
                        for subevaluator in wrapper_options:
                            for model in wrapper_models:
                                fs = fss(start, end, direction, subevaluator, model=model, df=df, features=all_vars,
                                         target=response_var)
                                if len(fs) > 1:
                                    # print(
                                    #     f"Selection starting with {start} till {end}, moving {direction}, with evaluation {subevaluator} of {model.__class__} is {np.sort(np.array(all_vars)[fs])}")
                                    model_srt = str(model).replace("alpha=1e-05, early_stopping=True, ", "").replace(
                                        ", random_state=1", "")
                                    print(
                                        f"{start},{end},{direction},{subevaluator},{model_srt},{np.array2string(fs.astype(int), separator=',')[1:-1]}")
                    else:
                        raise Exception("Invalid evaluation strategy found: " + evaluator)
