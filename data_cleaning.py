import pandas as pd
# import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    'GenHlth', 'Age', 'Education', 'Income'
]

# Numerical Variables
numerical_vars = [
    'BMI', 'MentHlth', 'PhysHlth'
]

categorical_vars = ordinal_vars + nominal_vars

response_var = 'Diabetes_012'

df_nominal = df[nominal_vars]
df_ordinal = df[ordinal_vars]
df_numerical = df[numerical_vars]
df_categorical = df[categorical_vars]


null_counts = df.isnull().sum()



