import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("weatherAUS.csv")

df = df.sample(n=1000)

# Print column labels
print("Column labels:", df.columns)
""" 
Date: Days
Location
MinTemp
MaxTemp
Rainfall
Evaporation
Sunshine
WindGustDir
WindGustSpeed
WindDir9am
WindDir3pm
WindSpeed9am 
WindSpeed3pm
Cloud9am
Cloud3pm
Temp9am
Temp3pm
RainToday
RainTomorrow
"""

# Print the first rows (default is 5 rows, you can specify how many with df.head(n))
# print("\nFirst rows of the DataFrame:")
# print(df.loc[0])

nominal_vars = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
ordinal_vars = ['Date']
categorical_vars = nominal_vars + ordinal_vars

response_var = 'RainTomorrow'

numerical_vars = list(filter(lambda x: x not in categorical_vars and x != response_var, df.columns))
df_nominal = df[nominal_vars]
df_ordinal = df[ordinal_vars]
df_numerical = df[numerical_vars]

cycled_days = pd.to_datetime(df_ordinal['Date']).dt.dayofyear

# Plotting each column as a scatterplot against its index and as a histogram
for column in df_numerical.columns:
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    # Histogram of the column values
    sns.jointplot(x=column, y=cycled_days, data=df_numerical, kind='kde')

    # sns.histplot(df_numerical[column], kde=True, ax=axes[1])
    axes.set_title(f'KDE plot of {column} vs day of the year')
    axes.set_xlabel(column)
    axes.set_ylabel('Day of year')

    plt.tight_layout()

    # Save the plot to the 'plots' directory
    plot_filename = f'plots/{column}_vs_days_kde.png'
    plt.savefig(plot_filename)

    # Close the plot to free memory
    plt.close(fig)


for column in df_numerical.columns:
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    # Histogram of the column values
    sns.kdeplot(x=column, data=df_numerical)

    # sns.histplot(df_numerical[column], kde=True, ax=axes[1])
    axes.set_title(f'KDE plot of {column}')
    axes.set_xlabel(column)
    axes.set_ylabel('Density')

    plt.tight_layout()

    # Save the plot to the 'plots' directory
    plot_filename = f'plots/{column}_scatter_hist.png'
    plt.savefig(plot_filename)

    # Close the plot to free memory
    plt.close(fig)