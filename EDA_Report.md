# Exploratory Data Analysis Report

## Dataset Overview

The Titanic dataset contains passenger information such as passenger class, gender, age, fare, ticket details, and survival status.

---

## Summary Statistics

Descriptive statistics were generated for all numerical features.

Key observations:

* Average passenger age is approximately 29 years.
* Fare values show high variability.
* Significant differences exist between passenger classes.

---

## Missing Value Analysis

### Missing Columns

* Age
* Cabin
* Embarked

### Observation

Cabin contains a large number of missing values, while Age contains a moderate number of missing values.

---

## Age Analysis

### Findings

* Most passengers were between 20 and 40 years old.
* Age distribution shows moderate positive skewness.
* Few elderly passengers are present.

---

## Fare Analysis

### Findings

* Fare distribution is highly right-skewed.
* Several extreme outliers exist.
* First-class passengers generally paid higher fares.

---

## Survival Analysis

### Findings

* More passengers did not survive than survived.
* Female passengers had significantly higher survival rates.
* First-class passengers showed better survival probabilities.

---

## Correlation Analysis

### Findings

* Fare and Passenger Class show a meaningful relationship.
* Survival is influenced by Fare and Passenger Class.
* Some features exhibit moderate correlations.

---

## Outlier Analysis

### Findings

* Fare contains several extreme outliers.
* Boxplots clearly identify anomalous fare values.

---

## Key Insights

1. Female passengers had a higher chance of survival.
2. First-class passengers survived more often.
3. Higher fare passengers generally had better survival rates.
4. Age had a moderate impact on survival.
5. Fare distribution contains significant outliers.

---

## Conclusion

Exploratory Data Analysis revealed important patterns and relationships within the Titanic dataset. Features such as Gender, Passenger Class, Fare, and Age play a significant role in passenger survival and can be used as strong predictors in future machine learning models.
