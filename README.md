# Titanic Data Analysis

This repository contains a Python script for exploratory data analysis (EDA) of the Titanic dataset. The dataset includes information about the passengers aboard the Titanic, such as their demographic details and whether they survived the disaster.

## Dataset Description

The dataset contains the following columns:

- **Survived**: Indicates whether the passenger survived (1) or not (0).
- **Pclass**: The passenger's class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Name**: The passenger's name.
- **Sex**: The passenger's gender.
- **Age**: The passenger's age.
- **SibSp**: The number of siblings/spouses aboard.
- **Parch**: The number of parents/children aboard.
- **Ticket**: The ticket number.
- **Fare**: The fare paid for the ticket.
- **Cabin**: The cabin number.
- **Embarked**: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Analysis Steps

1. **Loading the Data**: The dataset is loaded into a Pandas DataFrame.
2. **Data Exploration**: Basic statistics and information about the dataset are displayed.
3. **Survival Analysis**: The number of survivors and non-survivors is calculated.
4. **Class and Gender Analysis**: The survival rates based on passenger class and gender are analyzed.
5. **Age Analysis**: Passengers are categorized into age groups, and survival rates are analyzed.
6. **Fare Analysis**: The relationship between fare and passenger class is explored.
7. **Visualizations**: Various visualizations are created to understand the data better, including bar plots, scatter plots, and heatmaps.

## Code Overview

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load the dataset
data = pd.read_csv('./titanic.csv', index_col='PassengerId')

# Basic data exploration
print(data.shape)
print(data.head())
print(data.describe())

# Survival analysis
print(data.Survived.value_counts())

# Class and gender analysis
print(data.groupby('Pclass').mean(numeric_only=True).get('Survived'))
print(data.groupby('Sex').mean(numeric_only=True).get('Survived'))

# Age analysis
def age_category(age):
    if age < 30:
        return 1
    elif 30 <= age < 55:
        return 2
    else:
        return 3

data['AgeCategory'] = data['Age'].apply(age_category)

# Fare analysis
sns.boxplot(x='Pclass', y='Fare', data=data, palette='viridis')
plt.show()

# Visualizations
sns.countplot(x='Sex', hue='Survived', data=data, palette='viridis')
plt.show()

sns.histplot(data=data, x='Age', hue='Survived', kde=True, palette='viridis', bins=30)
plt.show()

# Correlation matrix
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
numeric_data = data[numeric_columns]
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

## Installation
Install the necessary libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn missingno
```

## Usage
Clone the repository:
```bash
git clone <your repository>
```

Перейдите в директорию репозитория:
```bash
cd <repository directory>
```

Run the script:
```bash
python titanic_analysis.py
```

## Results
Data analysis allows us to draw the following conclusions:

- Class and survivability: Upper class passengers had a higher chance of survival.

- Gender and survival: Women survived more often than men.

- Age and survival rate: Younger passengers survived more often than older ones.

## Visualizations

- Survival rate by gender: A graph showing the number of survivors and deaths depending on gender.

- Survival Rate by class: A graph showing the number of survivors and dead depending on the cabin class.

- Age distribution: A histogram showing the age distribution of survivors and dead.

- Correlation matrix: A heat map showing the correlation between numerical features.

## Conclusion

This data analysis provides a detailed overview of the Titanic dataset, highlighting the key factors that influenced passenger survival. Visualizations and statistical analysis help to understand the relationships between different variables and their impact on survival rates.

## Author

[Kirill Doroshev (DKMFzF)](https://vk.com/dkmfzf )

## License

This project is licensed under the MIT license
