
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SequentialFeatureSelector, f_classif, chi2, mutual_info_classif, SelectKBest, \
    f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, cohen_kappa_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import discrete_random_variable as drv  # from pyitlib

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
    # scorer = make_scorer(roc_auc_score, needs_proba=True, average='macro')
    scorer = 'roc_auc_ovr_weighted'
    scorer = 'f1_macro'

    if end == 'performance_plateau':
        tol = 0.025
        if start == 'random':
            return np.zeros(1)
        else:
            sfs = SequentialFeatureSelector(estimator=model, direction=direction, tol=tol, n_features_to_select="auto",
                                            n_jobs=-1, scoring=scorer, cv=3)

    elif end[:len('limit_fea')] == 'limit_fea':
        n_fea = int(end[len('limit_fea') + 1:])
        if start == 'random':
            return np.zeros(1)
        else:
            sfs = SequentialFeatureSelector(estimator=model, direction=direction, n_features_to_select=n_fea,
                                            n_jobs=-1, scoring=scorer, cv=3)
    else:
        raise ValueError("Incorrect end value")

    X = df[features]
    y = df[target]
    sfs.fit(X, y)
    result = sfs.get_support()
    assert (np.sum(result) == n_fea)
    return result


def get_fss_filter(start, end, direction, df, features, target, metric='mutual_information'):
    if end == 'timeout' or direction == 'exhaustive':
        return np.zeros(1)
    if end == 'performance_plateau':
        return np.zeros(1)
    if direction not in ['forward']:
        return np.empty(0)

    if metric == 'mutual_information':
        score_func = mutual_info_classif
    elif metric == 'f_classif':
        score_func = f_classif
    elif metric == 'chi2':
        score_func = chi2
    else:
        raise ValueError('Metric must be either "mutual_information", "f_classif" or "chi2"')

    subset = np.zeros(1)
    if end[:len('limit_fea')] == 'limit_fea':
        n_fea = int(end[len('limit_fea') + 1:])
        sfs = SelectKBest(score_func=score_func, k=n_fea)
        sfs.fit(df[features], df[target])
        subset = sfs.get_support()
    else:
        raise ValueError('Only limit of features accepted for filter')
    return subset

    # elif end[:len('limit_fea')] == 'limit_fea':
    #     n_fea = int(end[len('limit_fea') + 1:])
    #     assert (n_fea < len(features))
    #
    #     if start == 'random':
    #         subset = random_selection(len(features))
    #         while (1 if direction == 'forward' else -1) * (n_fea - np.sum(subset)) <= 0:
    #             subset = random_selection(len(features))
    #     elif start == 'none':
    #         subset = np.zeros(len(features), dtype=bool)
    #     else:  # start == 'all':
    #         subset = np.ones(len(features), dtype=bool)
    #
    #     if metric == 'mutual_information':
    #         scores = mutual_info_classif(df[features], df[target])
    #     elif metric == 'f_val':
    #         scores = -f_classif(df[features], df[target])[1]
    #     elif metric == 'chi2':
    #         scores = -chi2(df[features], df[target])[1]
    #     else:
    #         raise ValueError("Invalid metric used for filter FSS: " + metric)
    #
    #     if direction == 'forward':
    #         to_select = n_fea - np.sum(subset)
    #         masked_scores = np.ma.masked_array(scores,  subset)
    #         indices = np.ma.masked_array.argsort(masked_scores, fill_value=-np.infty)
    #         kbest = indices[-to_select:]
    #         subset[kbest] = True
    #
    #         if np.sum(subset) != n_fea:
    #             x = 0
    #     elif direction == 'backward':
    #         to_select = np.sum(subset) - n_fea
    #         masked_scores = np.ma.masked_array(scores, np.logical_not(subset))
    #         indices = np.ma.masked_array.argsort(masked_scores, fill_value=np.infty)
    #         kworst = indices[:to_select]
    #
    #         if np.sum(subset) - len(indices[:to_select]) != n_fea:
    #             x = 0
    #
    #         subset[kworst] = False
    #
    #         if np.sum(subset) != n_fea:
    #             x = 0
    #     else:
    #         return np.zeros(1)
    #
    #     assert (np.sum(subset) == n_fea)
    # return subset


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

    cmi_fun = cmi # drv.information_mutual

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
                    mi = drv.information_mutual(df[features[i]].values, df[target].values)
                    if best < mi:
                        best = mi
                        best_i = i
                else:
                    for j in range(len(subset)):
                        if not subset[j]:
                            continue

                        min_cmi = min(min_cmi, cmi_fun(df[features[i]].values, df[target].values, df[features[j]].values))

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
                    mi = drv.information_mutual(df[features[i]].values, df[target].values)
                    if best > mi:
                        best = mi
                        best_i = i
                else:
                    for j in range(len(subset)):
                        if subset[j]:
                            continue

                        min_cmi = max(min_cmi,  cmi_fun(df[features[i]].values, df[target].values, df[features[j]].values))

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
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'symmetrical_uncertainty':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 't-test':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'kruskal-wallis':
        raise Exception("Evaluator unimplemented: " + eval_strategy)
    elif eval_strategy == 'multivariate_cmi':
        return multivariate_cmi(start_fs, end_condition, dir, df, features, target)
    elif eval_strategy in ["f_classif", "mutual_information", "chi2"]:
        return get_fss_filter(start_fs, end_condition, dir, metric=eval_strategy, df=df, features=features, target=target)
    else:
        raise Exception("Invalid evaluation strategy found: " + eval_strategy)

if __name__ == "__main__":
    df = pd.read_csv("csv/outlier_filtered.csv")

    response_var = 'Diabetes_012'
    features = list(df.columns)
    features.remove(response_var)
    starting_point = [
        'all', 'none'
    ]
    search_organization = [
        'backward', 'forward'  # , 'stepwise', 'metaheuristic'
    ]
    evaluation_strategy = [
         'wrapper'
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
    uni_filter_options = [
        "f_classif", "mutual_information", "chi2"
    ]
    multi_filter_options = [
        'multivariate_cmi'
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
        'sequential_feature_selection',
    ]
    stopping_criterion = [
        # 'performance_plateau',
        'limit_fea_8', 'limit_fea_10', 'limit_fea_12', 'limit_fea_14', 'limit_fea_16'
    ]
    wrapper_models = [
        LogisticRegression(random_state=1, max_iter=1000),
        KNeighborsClassifier(5, weights='distance'),
        MLPClassifier(solver='adam', alpha=1e-5,
                      hidden_layer_sizes=(32, 32), random_state=1, max_iter=1000, early_stopping=True),
    ]

    print(
        f"start,end,direction,subevaluator,model,{str(features)[1:-1]}".replace("'", ""))
    for evaluator in evaluation_strategy:
        for start in starting_point:
            for direction in search_organization:
                if (start == 'none' and direction == 'backward') or (start == 'all' and direction == 'forward'):
                    continue

                for end in stopping_criterion:
                    if evaluator == 'uni_filter':
                        for subevaluator in uni_filter_options:
                            fs = fss(start, end, direction, subevaluator, df=df, features=features, target=response_var)
                            if len(fs) > 1:
                                print(
                                    f"{start},{end},{direction},{evaluator},{subevaluator},{np.array2string(fs.astype(int), separator=',')[1:-1]}")
                    elif evaluator == 'multi_filter':
                        for subevaluator in multi_filter_options:
                            fs = fss(start, end, direction, subevaluator, df=df, features=features, target=response_var)
                            if len(fs) > 1:
                                print(
                                    f"{start},{end},{direction},{evaluator},{subevaluator},{np.array2string(fs.astype(int), separator=',')[1:-1]}")
                    elif evaluator == 'wrapper':
                        for subevaluator in wrapper_options:
                            for model in wrapper_models:
                                fs = fss(start, end, direction, subevaluator, model=model, df=df, features=features,
                                         target=response_var)
                                if len(fs) > 1:
                                    model_srt = str(model).replace("alpha=1e-05, early_stopping=True, ", "").replace(
                                        ", random_state=1", "").replace("\n              ", "")
                                    print(
                                        f"{start},{end},{direction},{subevaluator},{model_srt.replace(',', '|')},{np.array2string(fs.astype(int), separator=',')[1:-1]}")
                    else:
                        raise Exception("Invalid evaluation strategy found: " + evaluator)
