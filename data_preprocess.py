import pandas as pd
# import sklearn as skl
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler
import warnings

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Find this dataset at https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
df = pd.read_csv("csv/diabetes_012_health_indicators_BRFSS2015.csv")

df_test = df.sample(n=1000, random_state=0)  # For debugging purposes

# Print column labels
print("Column labels:", df.columns)
variable_descriptions = {
    "Diabetes_012": "Diabetes status: 0 = no diabetes, 1 = prediabetes, 2 = diabetes",
    "HighBP": "High Blood Pressure: 0 = no high BP, 1 = high BP",
    "HighChol": "High Cholesterol: 0 = no high cholesterol, 1 = high cholesterol",
    "CholCheck": "Cholesterol test in past 5 years: 0 = no, 1 = yes",
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


def normalization_test(vec):
    # statistic, pvalue = scipy.stats.jarque_bera(vec.values.reshape(-1, 1))
    statistic, pvalue = scipy.stats.shapiro(vec)
    return pvalue


def log_normal_test(vec):
    tvec = np.log(vec)
    if np.isnan(tvec).any() or np.isinf(tvec).any():
        return 0
    return normalization_test(tvec)


def inv_normal_test(vec: np.ndarray):
    tvec = 1 / vec
    if np.isnan(tvec).any() or np.isinf(tvec).any():
        return 0
    return normalization_test(tvec)


def sqrt_normal_test(vec):
    tvec = np.sqrt(vec)
    if np.isnan(tvec).any() or np.isinf(tvec).any():
        return 0
    return normalization_test(tvec)


def cube_root_normal_test(vec):
    tvec = np.power(vec, 1 / 3)
    if np.isnan(tvec).any() or np.isinf(tvec).any():
        return 0
    return normalization_test(tvec)


def inv_sqrt_normal_test(vec):
    tvec = 1 / np.sqrt(vec)
    if np.isnan(tvec).any() or np.isinf(tvec).any():
        return 0
    return normalization_test(tvec)


def symmetric_test(vec):
    stat, pvalue = scipy.stats.wilcoxon(vec - np.median(vec))
    return pvalue[0]


min_pvalue = 0  # np.power(10.0, -30)

scalers_options = {
    'normal': {'test': normalization_test, 'transform': scipy.stats.zscore},
    'log-normal': {'test': log_normal_test, 'transform': lambda x: scipy.stats.zscore(np.log(x))},
    'inv-normal': {'test': inv_normal_test, 'transform': lambda x: scipy.stats.zscore(1 / x)},
    'sqrt-normal': {'test': sqrt_normal_test, 'transform': lambda x: scipy.stats.zscore(np.sqrt(x))},
    'cube-root-normal': {'test': cube_root_normal_test,
                         'transform': lambda x: scipy.stats.zscore(1 / np.power(x, 1 / 3))},
    'inv-sqrt-normal': {'test': inv_sqrt_normal_test, 'transform': lambda x: scipy.stats.zscore(1 / np.sqrt(x))},
    # 'symmetric': {'test': symmetric_test, 'transform': lambda x: MaxAbsScaler().fit_transform(x - np.median(x))},
    # 'default': {'test': lambda x: min_pvalue, 'transform': lambda x: QuantileTransformer().fit_transform(x - np.median(x))},
    'default': {'test': lambda x: min_pvalue, 'transform': lambda x: RobustScaler().fit_transform(x - np.median(x))}
}



epsilon = 1

for fea in ordinal_vars + numerical_vars:
    df[fea] = df[fea].values.reshape(-1, 1) - np.max([np.min(df[fea].values.reshape(-1, 1)), 0]) + epsilon

for fea in df[ordinal_vars]:
    best_pvalue = 0.0
    transform = 'default'

    for option in scalers_options.keys():
        scaler = scalers_options[option]
        pvalue = scaler['test'](df_test[fea].values.reshape(-1, 1))
        print(f"{fea},{pvalue},{option}")
        if best_pvalue < pvalue:
            best_pvalue = pvalue
            transform = option

    print(f"{fea},{transform}")
    new_fea = scalers_options[transform]['transform'](df[fea].values.reshape(-1, 1))

    # # Plotting the histogram
    # plt.figure(figsize=(8, 5))
    # plt.hist(new_fea, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Data Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()

    df[fea] = new_fea

for fea in numerical_vars:
    best_pvalue = 0.0
    transform = 'default'

    for option in scalers_options.keys():
        scaler = scalers_options[option]
        pvalue = scaler['test'](df[fea].values.reshape(-1, 1))
        # print(f"{fea},{pvalue},{option}")
        if best_pvalue < pvalue:
            best_pvalue = pvalue
            transform = option

    print(f"{fea},{transform}")
    new_fea = scalers_options[transform]['transform'](df[fea].values.reshape(-1, 1))

    # # Plotting the histogram
    # plt.figure(figsize=(8, 5))
    # plt.hist(new_fea, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Data Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()

    df[fea] = new_fea

for fea in df[nominal_vars]:
    unique = np.unique(df[fea].values)

    best_pvalue = 0
    transform = 'default'

    if len(unique) == 2:
        print(f"{fea} retained as is due to binary options")
    else:
        raise ValueError(f"Feature {fea} is nominal with more than two options. Consider vector embedding.")

df.to_csv('csv/test.csv')
