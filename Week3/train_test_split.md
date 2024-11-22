# Train_test Split
The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions **_on data not used to train the model._**

# Train-Test Split Evaluation
It can be used for **classification** or **regression problems** and can be used for **any supervised learning algorithm**.<br>

The procedure involves taking a dataset and dividing it into two subsets. The first subset is used to fit the model and is referred to as the training dataset. The second subset is not used to train the model; instead, the input element of the dataset is provided to the model, then predictions are made and compared to the expected values. This second dataset is referred to as the test dataset.
    - Train Dataset: Used to fit the machine learning model.
    - Test Dataset: Used to evaluate the fit machine learning model.

## When to Use the Train-Test Split
The dataset must be sufficiently large that it can be split into train and test dataset and each of those dataset are suitable representation of problem domain.(So we should have suffiecient data, sufficient enough to effectively evaluate the model performance.)

A suitable representation of the problem domain means that there are enough records to cover all common cases and most uncommon cases in the domain. This might mean combinations of input variables observed in practice. It might require thousands, hundreds of thousands, or millions of examples.

>If you have insufficient data, then a suitable alternate model evaluation procedure would be the ``k-fold cross-validation`` procedure.

In addition to dataset size, another reason to use the train-test split evaluation procedure is computational efficiency. Some models are very costly to train, and in that case, repeated evaluation used in other procedures is intractable. An example might be deep neural network models.

## How to Configure the Train-Test Split
The procedure has one main configuration parameter, which is the **_size of the train and test sets_**. This is most commonly expressed as a percentage between 0 and 1 for either the train or test datasets. For example, a training set with the size of 0.67 (67 percent) means that the remainder percentage 0.33 (33 percent) is assigned to the test set. (there is no optimal split percentage)


We must choose a split percentage that meets our project’s objectives with considerations that include:

    - Computational cost in training the model.
    - Computational cost in evaluating the model.
    - Training set representativeness.
    - Test set representativeness.

Nevertheless, common split percentages include:

    - Train: 80%, Test: 20%
    - Train: 67%, Test: 33%
    - Train: 50%, Test: 50%

# Train-Test Split Procedure in Scikit-Learn
The scikit-learn Python machine learning library provides an implementation of the train-test split evaluation procedure via the ``train_test_split()`` function.

The function takes a loaded dataset as input and returns the dataset split into two subsets.
```python
# split into train test sets
train, test = train_test_split(dataset, ...)
```
Ideally, we can split your original dataset into input (X) and output (y) columns, then call the function passing both arrays and have them split appropriately into train and test subsets.
```python
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
```

The size of the split can be specified via the ``“test_size”`` argument that takes a number of rows (integer) or a percentage (float) of the size of the dataset between 0 and 1.

The latter is the most common, with values used such as 0.33 where 33 percent of the dataset will be allocated to the test set and 67 percent will be allocated to the training set.
```python
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

The complete example is listed below.
```python
# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_blobs(n_samples=1000)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

Running the example splits the dataset into train and test sets, then prints the size of the new dataset.

We can see that 670 examples (67 percent) were allocated to the training set and 330 examples (33 percent) were allocated to the test set, as we specified.
```python
(670, 2) (330, 2) (670,) (330,)
```

Alternatively, the dataset can be split by specifying the ``“train_size”`` argument that can be either a number of rows (integer) or a percentage of the original dataset between 0 and 1, such as 0.67 for 67 percent.
```python
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
```

## Repeatable Train-Test Splits
>Another important consideration is that rows are assigned to the train and test sets randomly. This is done to ensure that datasets are a representative sample (e.g. random sample) of the original dataset, which in turn, should be a representative sample of observations from the problem domain.

When comparing machine learning algorithms, it is desirable (perhaps required) that they are fit and evaluated on the same subsets of the dataset. This can be achieved by fixing the seed for the pseudo-random number generator used when splitting the dataset.

This can be achieved by setting the ``“random_state”`` to an integer value. Any value will do; it is not a tunable hyperparameter.

```python
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```
The example below demonstrates this and shows that two separate splits of the data result in the same result.
```python
# demonstrate that the train-test split procedure is repeatable
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_blobs(n_samples=100)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
```

Running the example splits the dataset and prints the first five rows of the training dataset.

The dataset is split again and the first five rows of the training dataset are printed showing identical values, confirming that when we fix the seed for the pseudorandom number generator, we get an identical split of the original dataset.
```python
[[-2.54341511  4.98947608]
 [ 5.65996724 -8.50997751]
 [-2.5072835  10.06155749]
 [ 6.92679558 -5.91095498]
 [ 6.01313957 -7.7749444 ]]
 
[[-2.54341511  4.98947608]
 [ 5.65996724 -8.50997751]
 [-2.5072835  10.06155749]
 [ 6.92679558 -5.91095498]
 [ 6.01313957 -7.7749444 ]]
```
In this code, ``random_state=1`` is effectively setting the seed for the pseudorandom number generator used by ``train_test_split``. By specifying ``random_state=1``, we ensure that every time we run this code, the split of the dataset into training and testing sets will be identical.

## Stratified Train-Test Splits
>(for classification problems only)

A stratified train-test split is a method of dividing data into training and testing sets such that each set maintains the same proportion of each class as in the original dataset. This is especially useful for datasets with imbalanced classes, ensuring that the training and testing sets reflect the original distribution of classes. This helps prevent any class from being underrepresented in either the training or test set, which could lead to biased model training and inaccurate performance evaluation.
### When to Use Stratified Splits
    - Class Imbalance: When the dataset has significantly imbalanced classes (e.g., 90% of one class and 10% of another), a regular random split may lead to a test set that doesn’t adequately represent minority classes.
    - Ensuring Reproducibility: With stratified splitting, we can combine it with setting a random_state to ensure that each class remains proportionally represented, while the split is also reproducible.

### How to Implement a Stratified Split in Scikit-Learn
In Scikit-Learn, we can use the ``train_test_split`` function with the ``stratify`` parameter to create a stratified split.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a synthetic dataset with an imbalanced class distribution
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=1)

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
```
If original data has 90% of class 0 and 10% of class 1:

    - After a stratified split, both the training and test sets will also have about 90% of class 0 and 10% of class 1.

# Train-Test Split to Evaluate Machine Learning Models
## Train-Test Split for Classification
We will demonstrate how to use the train-test split to evaluate a random forest algorithm on the sonar dataset.

The sonar dataset is a standard machine learning dataset composed of 208 rows of data with 60 numerical input variables and a target variable with two class values, e.g. binary classification.

```python
# summarize the sonar dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```
Output:
```python
(208, 60) (208,)
# As expected, we can see that there are 208 rows of data with 60 input variables.
```
Next, we can split the dataset so that 67 percent is used to train the model and 33 percent is used to evaluate it. This split was chosen arbitrarily.
```python
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```
Output:
```python
(139, 60) (69, 60) (139,) (69,)
#The dataset is split into train and test sets and we can see that there are 139 rows for training and 69 rows for the test set.
```
We can then define and fit the model on the training dataset.
```python
# fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
```
Then use the fit model to make predictions and evaluate the predictions using the classification accuracy performance metric.
```python
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)
```
Output:
```python
Accuracy: 0.783
#Finally, the model is evaluated on the test set and the performance of the model when making predictions on new data has an accuracy of about 78.3 percent.
```
## Train-Test Split for Regression
We will demonstrate how to use the train-test split to evaluate a random forest algorithm on the housing dataset.

The housing dataset is a standard machine learning dataset composed of 506 rows of data with 13 numerical input variables and a numerical target variable.

The dataset involves predicting the house price given details of the house’s suburb in the American city of Boston.
```python
# load and summarize the housing dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
```
Output:
```python
(506, 14)
#506 rows of data with 13 numerical input variables and single numeric target variables (14 in total).
```
We can now evaluate a model using a train-test split.

First, the loaded dataset must be split into input and output components.
```python
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
#output: (506, 13) (506,)
```
Next, we can split the dataset so that 67 percent is used to train the model and 33 percent is used to evaluate it. This split was chosen arbitrarily.

```python
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Output: (339, 13) (167, 13) (339,) (167,)
# The dataset is split into train and test sets and we can see that there are 339 rows for training and 167 rows for the test set.
```
We can then define and fit the model on the training dataset.
```python
# fit the model
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
```
Then use the fit model to make predictions and evaluate the predictions using the mean absolute error (MAE) performance metric.
```python
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
#MAE: 2.171
# the model is evaluated on the test set and the performance of the model when making predictions on new data is a mean absolute error of about 2.211 (thousands of dollars).
```