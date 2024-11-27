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

>Each data point is used for both training and validation.

#### Steps in K-Fold Cross-Validation
1. Divide the Dataset: Split the data into K equal parts, or "folds."
2. Train and Test on Each Fold:
    - For each fold (from 1 to K):
        - Use the current fold as the test set.
        - Use the remaining K-1 folds as the training set.
         - Train the model on the training set and evaluate it on the test set.
3. Average the Results: After running K experiments, calculate the average performance across all folds to get a more accurate measure of the model’s performance.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(model, df_features, df_target, cv=kf)

print("Cross-validation scores:", scores)
```
Here ``n_splits`` is the value of K, increasing the value of K increases the accuracy of the model.

### Stratified Cross-Validation
Stratified Cross-Validation is a variation of K-Fold Cross-Validation that ensures each fold has approximately the same percentage of samples for each class as in the full dataset. This is particularly useful for datasets with imbalanced classes, where some classes are underrepresented.<br>
In standard K-Fold Cross-Validation, each fold is selected randomly, so there’s no guarantee that each class’s proportion is consistent across folds. Stratified Cross-Validation solves this by maintaining the class distribution in each fold.

>Essential for datasets with imbalanced classes

#### How Stratified Cross-Validation Works

1. **Divide Dataset into Folds with Proportionate Class Distribution**: The data is split into K folds, with each fold keeping the same class distribution as the entire dataset.
2. **Train and Test on Each Fold**: For each fold, train the model on K-1 folds and test it on the remaining fold. Repeat this K times.
3. **Average Results Across Folds**: Compute the average performance across all K folds, resulting in a more reliable performance estimate, especially with imbalanced classes.

```python
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
```
### Blocked Cross-Validation
Blocked Cross-Validation is a cross-validation technique designed for datasets where the order of data points matters, such as time-series data. In time-series data, observations are not independent of each other (e.g., stock prices, weather data), so the data needs to be split in a way that respects the temporal order to avoid data leakage.

>Ideal for grouped data, such as repeated measurements from the same subject.
#### How Blocked Cross-Validation Works

1. **Define Blocks**: The data is divided into separate, non-overlapping blocks based on the inherent structure of the data (e.g., by time or location).
2. **Train and Test on Blocks**:
        - For each block, train the model on the data outside the current block and use the block as the test set.
3. **Repeat for Each Block**: Continue this process, treating each block as the test set once while training on the other blocks.
4. **Average the Results**: After testing on all blocks, average the performance metrics across all test blocks for an overall evaluation.

### Rolling Cross-Validation
Rolling Cross-Validation, also known as Time-Series Cross-Validation or Rolling Window Cross-Validation, is a technique designed specifically for time-series data or any data where observations are sequentially dependent. In rolling cross-validation, training and test sets are created by sequentially moving a "window" through time, training on one period and testing on the next, then shifting the window forward.

>**Purpose**: Rolling cross-validation is designed to assess the performance of machine learning models on sequential data. It helps in understanding how well a model can predict future values based on past observations.

>**Methodology**: The process involves iteratively expanding the training set while moving the validation set forward in time. This allows the model to be trained on all available data up to a certain point before testing it on the next observation(s).

#### Example of Rolling Cross-Validation

1. Iteration 1:
    - Training set: Data from Year 1
    - Validation set: Data from Year 2
2. Iteration 2:
    - Training set: Data from Years 1 and 2
    - Validation set: Data from Year 3
3. Iteration 3:
    - Training set: Data from Years 1, 2, and 3
    - Validation set: Data from Year 4

This pattern continues until the last observation is reached.

```python
# Parameters for the rolling window
train_size = 50     # Initial training window size
test_size = 10      # Size of each test set
n_splits = (len(data) - train_size) // test_size  # Number of rolling iterations

# List to store the results
mse_scores = []

# Perform rolling cross-validation
for i in range(n_splits):
    # Define training and test sets for the current window
    train_end = train_size + i * test_size
    test_end = train_end + test_size
    X_train, X_test = data.iloc[:train_end], data.iloc[train_end:test_end]
    
    # Train a simple model
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train.index.values.reshape(-1, 1), X_train['value'])
    
    # Make predictions and evaluate
    predictions = model.predict(X_test.index.values.reshape(-1, 1))
    mse = mean_squared_error(X_test['value'], predictions)
    mse_scores.append(mse)
```

# Advanced Techniques for Handling Missing Data in Python
### Types of Missing Data
- **Missing Completely at Random (MCAR)**: No relationship with any other data.
- **Missing at Random (MAR)**: Missingness related to observed data.
- **Missing Not at Random (MNAR)**: Missingness related to unobserved data.

## Basic Imputation Techniques

1.  ### **Mean/Median Imputation**:
    - Use when: Data is MCAR and normally distributed.
    - Best Practice: Use median for skewed data.

```python
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
```
Filling missing data is important step for preparing dataset for training and analysis. Choice of imputation technique depends on the nature of the data and the specific column with missing values.

#### **Why Use Median for Age:**

- **Robust to Outliers**: The median is less sensitive to outliers compared to the mean. In datasets where age might have extreme values or a skewed distribution, using the median ensures that the imputed values are not unduly influenced by these extremes.

- **Dealing with MCAR or MAR**: If the missingness in the ‘Age’ column is MCAR (Missing Completely at Random) or MAR (Missing at Random), median imputation provides a reasonable estimate that preserves the central tendency of the data.

2. ### Mode Imputation:
- Use when: Best for categorical data..
```python
embarked_mode = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'].fillna(embarked_mode, inplace=True)
```
#### Why Use Mode for Cabin:

- Categorical Data: ‘Embarked’ is a categorical variable (representing different embarkation points). For categorical data, mean and median are not applicable. The mode, which is the most frequently occurring category, is a natural choice for imputation.

- Preserving the Most Common Category: By using the mode, you maintain the original distribution of the data as closely as possible. It ensures that the statistical and structural integrity of the ‘Embarked’ column is maintained post-imputation.

- Handling MCAR/MAR: If the data is MCAR or MAR, replacing missing values with the most

3. ### KNN Imputation:
KNN (K-Nearest Neighbors) Imputation is a method to fill in missing data by looking at the closest neighbors. Think of it as finding people similar to you (neighbors) and using their information to guess what’s missing in your data.
- Use when: Data has identifiable patterns.
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)
numeric_columns = titanic_data.select_dtypes(include=['float64', 'int64'])
titanic_data[numeric_columns.columns] = imputer.fit_transform(numeric_columns)
```

#### How and Why to Use KNN Imputation
- **Mechanism**: 
    - **Find Neighbors**: It looks at other data points (rows) in your dataset that are the most similar to the one with missing values.
    - **Use Neighbors to Fill the Gaps**: It takes the values from these similar data points to estimate and fill in the missing value.

    For example: If you're missing your favorite fruit in a survey, KNN might look at what people with similar preferences to you like, and assume you’d like the same.

- **Applicability**: This method is particularly useful when the data exhibits a pattern or relationship, such as in cases where similar observations have similar types of values.

- **Choosing ‘k’**: The choice of ‘k’ (number of neighbors) is crucial. 
    - **Small ‘k’ (fewer neighbors)**: Captures more detail but may overfit (too specific to a small group).
    - **Large ‘k’ (more neighbors)**: Smooths things out but might miss some subtle differences.

- **Distance Metric**: KNN uses a distance metric (like Euclidean distance) to find the closest neighbors. The choice of the distance metric can significantly impact the imputation. The closer two points are, the more similar they are considered.
- **Why Use KNN Imputation?**: It keeps the relationships and patterns in your data intact.It's flexible and works for many types of data.
> Keep in Mind:<br>
>It can be slow for very large datasets (because it compares each point with all others).<br>
>Choosing the right ‘k’ and distance metric is key to good results.

4. ### Regression Imputation:
Regression imputation is a method for filling in missing data by predicting the missing value using a regression model. 
- Use when: The missingness is MAR (Missing at Random) and can be predicted using other variables in the dataset.

- How Does It Work?
    1. Build a Regression Model:
        Use the rows that have all the data (no missing values) to create a predictive model. For instance:
            - Predict weight based on height, age, and gender.
            - The formula might look like:
            `` Weight = 2.5 * Height + 0.8 * Age - 5 * Gender + Error``

    2. Predict the Missing Values:
    Once the model is built, it predicts the missing values using the information from the other columns.

    Example:
    Imagine a dataset with three columns: Age, Height, and Weight, but some rows are missing Weight.

    1. Use rows where you know Weight to build a regression model:
    ``Weight = f(Age, Height)``

    2. For rows where Weight is missing:
        Plug in the Age and Height to predict Weight using the regression equation.

- When to Use Regression Imputation?

    - When there’s a strong relationship between the variable with missing data and other variables in your dataset.
    Example: If you know height and age are good predictors of weight.

## Handling MNAR Data (Missing Not at Random)
When dealing with Missing Not at Random (MNAR) data, the missingness is related to the unobserved data itself. This type of missing data requires more sophisticated approaches, such as data augmentation or model-based methods.
1. ### Data Augmentation Methods
Data augmentation involves creating additional data samples from your existing dataset. It’s commonly used in machine learning, especially for images, text, and audio, to increase the diversity and size of the training data, helping models generalize better.

Choosing the right augmentation methods depends on your data type and problem. For example, image tasks often rely on geometric transformations, while text tasks use synonym replacement and back translation.

2. ### Model-Based Methods
These methods involve building statistical models that directly account for the mechanism of missingness. Commonly used in scenarios where the reasons for missingness are complex and intertwined with the unobserved values.
Some of the model-based methods are : 
- Multiple imputation:
    Instead of filling in missing values once, multiple imputation creates several datasets with different imputed values.
- Expectation-Maximization(EM) Algorithm:
    EM is an iterative technique that estimates missing data and model parameters simultaneously.
    - Expectation Step: Estimate missing values based on observed data and current model parameters.
    - Maximization Step: Update model parameters using the completed dataset.
    - Repeat until the model converges.
- Matrix Factorization:
    Common in recommender systems, this approach decomposes a matrix (e.g., user-item interactions) into latent factors and uses them to predict missing values.<br>
    **Example**: In a user-movie rating matrix, predict missing ratings by learning patterns in existing ratings.
- K-Nearest Neighbors(KNN) Imputation:
    This method uses the closest data points (neighbors) to fill in missing values based on a similarity metric (e.g., Euclidean distance).

- Regression Imputation:
    Use regression models to predict missing values based on other variables in the dataset.

- Neural Network-Based Methods:
    Neural networks can model complex, nonlinear relationships to estimate missing values. Autoencoders, for example, can reconstruct incomplete data by learning a compressed representation.

- Bayesian Methods:
    Use Bayesian inference to model the distribution of missing data and estimate its values probabilistically.
    **Example**: Use Gibbs Sampling to estimate missing values iteratively.