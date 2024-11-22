# Data Preprocessing
- **Importance**: Essential for converting raw data into a format suitable for analysis.

- **Goals**: Enhance data quality, improve analysis efficiency, and prepare data for machine learning.

## Data Preprocessing Workflow
- **Cleaning Data**: Remove duplicates, correct errors.
- **Handling Missing Values**: Impute missing values or remove them.
- **Normalization**: Scale data using methods like Min-Max scaling or Z-score normalization.
- **Feature Engineering**: Create new features from existing data.

## Understanding Data Types and Scales
- #### Data Types
    - **Numeric (Quantitative)**: Numbers representing continuous or discrete data.
    - **Categorical (Qualitative)**: Data grouped into categories.
- #### Scales
    - **Nominal**: Categories without order (e.g., blood types).
    - **Ordinal**: Ordered categories (e.g., class levels).
    - **Interval**: Numeric scales without true zero (e.g., temperature in Celsius).
    - **Ratio**: Numeric scales with true zero (e.g., height).


## Basic summary in python 
We have a dataset related to Covid. We download the file to operate on it. <br>
https://github.com/100daysofml/100daysofml.github.io/blob/main/content/Week_03/covid_data.csv <br>

We operate on this dataset to perform different operations.

```
#import relevant libraries
import numpy as np
import pandas as pd
from scipy import stats

covid_data = pd.read_csv("covid_data.csv")
covid_data.head()
covid_data.tail()
covid_data.info()

# mode

stats.mode(covid_datanew['new_cases']): Returns a ModeResult object with the mode and its count.

stats.mode(covid_datanew['new_cases'])[0]: Accesses the array containing the mode value(s).

stats.mode(covid_datanew['new_cases'])[0][0]: Accesses the first element of the array, providing the actual mode value.

```

We can calculate variance and standard deviation using both Numpy and Pandas.

### Why would there be a difference in the variance and standard deviation between NumPy and Pandas?
Numpy var(``numpy.var()``) and Pandas(``pandas.series.var()``) var give different value. 

The difference between the numpy var and pandas var methods are not dependent on the range of the data but on ``the degrees of freedom (ddof)`` set by package. pandas sets ddof=1 (unbiased estimator) while numpy sets ddof = 0 (mle). 


### Why are Quartiles and Interquartile Range Important?
Quartiles and the Interquartile Range (IQR) are essential in data analysis for several key reasons:

- Measure of Spread
    Quartiles divide a dataset into four equal parts, providing insight into the distribution and variability of the data.

- Outlier Detection
    The IQR is a robust measure of statistical dispersion and is commonly used for identifying outliers. Values that fall below ``Q1 - 1.5*IQR ``or above ``Q3 + 1.5*IQR`` are often considered outliers.

- Non-parametric
    Quartiles do not assume a normal distribution of data, making them non-parametric and robust measures for skewed distributions or data with outliers.

- Data Segmentation and Comparison
    Quartiles allow for easy segmentation of data into groups, which is useful in various applications like finance and sales.

- Informative for Further Statistical Analysis
    Understanding quartile positions helps in making informed decisions for further statistical analyses, especially with skewed data.

- Basis for Other Statistical Measures
    Quartiles are foundational for other statistical visualizations like box plots, which depict quartiles and outliers graphically.

```
# Calculate quartiles
Q1 = np.quantile(covid_data["new_cases"],0.25)
Q3 = np.quantile(covid_data["new_cases"],0.75)

# Calculate the Interquartile Range
IQR = Q3 - Q1

```

## Data Splitting Techniques
In data science or machine learning, data splitting comes into the picture when the given data is divided into two or more subsets so that a model can get trained, tested and evaluated.
- If two splits are there, it means one will be utilised for training and another one will be used for testing, or,
- If three splits are there will mean there are training, testing and validation sets.

### Simple Random Split
```python
from sklearn.model_selection import train_test_split
# Assuming df is your DataFrame
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
```
The ``random_state`` is a pseudo-random number parameter that allows us to reproduce the same train test split each time you run the code.

### Stratified Random Split
```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["stratum_column"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
```
This code demonstrates how to use ``StratifiedShuffleSplit`` to split a dataset into training and testing sets while preserving the proportions of a specific column's classes across both sets.

This code ensures that both strat_train_set and strat_test_set have the same class proportions for "stratum_column" as the original dataset, making it especially useful for imbalanced data.

## Advanced Splitting Techniques
### K-Fold Cross-Validation
K-Fold Cross-Validation is a technique in machine learning used to evaluate the performance of a model by dividing the dataset into ``K`` equal-sized subsets, or "folds." The model is trained and evaluated ``K`` times, each time using a different fold as the test set and the remaining ``K-1`` folds as the training set repeated ``K`` times.