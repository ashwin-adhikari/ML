Preprocessing the data implies using the data which is easily readable by the machine learning model.

## Data that needs data preprocessing
Since data comes in various formats, there can be certain errors that need to be corrected. 
Let us learn about these dataset errors:

### 1. Missing values in dataset
The values can be missed because of various reasons such as human errors, mechanical errors, etc.
There are three techniques to solve the missing values’ problem to find out the most accurate features, and they are:

- **Dropping**
    Those rows in the dataset or the entire columns with missed values are dropped to avoid errors from occurring in data analysis. (Dropping rows with missing values might lead to a decrease in performance of model)
    A simple solution for the problem of a decreased training size due to the dropping of values is to use **_imputation_**. In case of dropping, you can define a threshold to the machine.
    Let us take 60% threshold in our example, which means that 60% of data with missing values will be accepted by the model/algorithm as the training dataset, but the features with more than 60% missing values will be dropped.
    ```python
    # Dropping columns in the data higher than 60% threshold
    data = data[data.columns[data.isnull().mean() < threshold]]

    # Dropping rows in the data higher than 60% threshold
    data = data.loc[data.isnull().mean(axis=1) < threshold]
    ```

- **Numerical imputation**
    Means replacing the missing values with such a value that makes sense usually done with numbers.

    > For instance, if there is a tabular dataset with the number of stocks, commodities and derivatives traded in a month as the columns, it is better to replace the missed value with a “0” than leave them as it is.
    Data size is preserved hence predictive models can predict accurately.
    ```python
    # For filling all the missed values as 0
    data = data.fillna(0)

    # For replacing missed values with median of columns
    data = data.fillna(data.median())
    ```

- **Categorical imputation**
    Replacing the missing values with the one which occurs maximum number of times in column. But, in case there is no such value that occurs frequently or dominates the other values, then it is best to fill the same as “NAN”.
    ```python
    # Categorical imputation
    data['column_name'].fillna(data['column_name'].value_counts().idxmax(), inplace=True)
    ```

### 2. Outliers in dataset
Outliers are anomalies or extreme values in a dataset that can skew analysis or modelling results often addressed through identification and removal techniques.
An outlier differs significantly from other values and is too distanced from the mean of the values. Such values that are considered outliers are usually due to some systematic errors or flaws.
```python
# For identifying the outliers with the standard deviation method
outliers = [x for x in data if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

# Remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))
```

### 3. Overfitting in dataset
Overfitting occurs when the model fits the data too well or simply put when the model is too complex.
The overfitting model learns the detail and noise in the training data to such an extent that it negatively impacts the performance of the model on new data/test data.
>The overfitting problem can be solved by decreasing the number of features/inputs or by increasing the number of training examples to make the machine learning algorithms more generalised.
The most common solution is **_regularisation_** in an overfitting case. **_Binning_** is the technique that helps with the regularisation of the data which also makes you lose some data every time you regularise it. Binning simply means grouping of data in finite range of intervals.
The Python code for binning:
```python
data['bin'] = pd.cut(data['value'], bins=[100,250,400,500], labels=["Lowest", "Mid", "High"])
```
Output of code looks like this:
|     | **Value** | **Bin**  |
|-----|-----------|----------|
| 0   | 102       | Low      |
| 1   | 300       | Mid      |
| 2   | 107       | Low      |
| 3   | 470       | High     |

### 4. Data with no numerical values in a dataset
In the case of the dataset with no numerical values, it becomes impossible for the machine learning model to learn the information.
The machine learning model can only handle numerical values and thus, it is best to spread the values in the columns with assigned binary numbers “0” or “1”. This technique is known as **_one-hot encoding_**.
In this type of technique, the grouped columns already exist. For instance, below I have mentioned a grouped column:

| **Infected** | **Covid variants** |
|--------------|--------------------|
| 2            | Delta              |
| 4            | Lambda             |
| 5            | Omicron            |
| 6            | Lambda             |
| 4            | Delta              |
| 3            | Omicron            |
| 5            | Omicron            |
| 4            | Lambda             |
| 2            | Delta              |

Now, the above-grouped data can be encoded with the binary numbers ”0” and “1” with one hot encoding technique. This technique subtly converts the categorical data into a numerical format in the following manner:

| **Infected** | **Delta** | **Lambda** | **Omicron** |
|--------------|-----------|------------|-------------|
| 2            | 1         | 0          | 0           |
| 4            | 0         | 1          | 0           |
| 5            | 0         | 0          | 1           |
| 6            | 0         | 1          | 0           |
| 4            | 1         | 0          | 0           |
| 3            | 0         | 0          | 1           |
| 5            | 0         | 0          | 1           |
| 4            | 0         | 1          | 0           |
| 2            | 1         | 0          | 0           |

Hence, it results in better handling of grouped data by converting the same into encoded data for the machine learning model to grasp the encoded (which is numerical) information quickly.

### 5. Different date formats  in a dataset
With the different date formats such as “25-12-2021”, “25th December 2021” etc. the machine learning model needs to be equipped with each of them. Or else, it is difficult for the machine learning model to understand all the formats.

With such a dataset, we can preprocess or decompose the data by mentioning three different columns for the parts of the date, such as Year, Month and Day.
```python
#Convert to datetime object
df['Date'] = pd.to_datetime(df['Date'])

#Decomposition
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df[['Year','Month','Day']].head()
```
Output:

| **Year** | **Month** | **Day** |
|------|-------|-----|
| 2019 | 1     | 5   |
| 2019 | 3     | 8   |
| 2019 | 3     | 3   |
| 2019 | 1     | 27  |
| 2019 | 2     | 8   |

In the output above, the dataset is in date format which is numerical. And because of decomposing the date into different parts such as Year, Month and Day, the machine learning model will be able to learn the date format.

The entire process mentioned above where data cleaning takes place can also be termed as **_data wrangling_**.

## Data cleaning vs data preprocessing
In the context of trading, data cleaning may involve handling errors in historical stock prices or addressing inconsistencies in trading volumes.

However, data preprocessing is then applied to prepare the data for technical analysis or machine learning models, including tasks such as scaling prices or encoding categorical variables like stock symbols.
| Aspect         | Data Cleaning                                           | Data Preprocessing                                          |
|----------------|---------------------------------------------------------|-------------------------------------------------------------|
| **Objective**  | Identify and rectify errors or inaccuracies in stock prices. | Transform and enhance raw stock market data for analysis.   |
| **Focus**      | Eliminating inconsistencies and errors in historical price data. | Addressing missing values in daily trading volumes and handling outliers. |
| **Tasks**      | Removing duplicate entries.                             | Scaling stock prices for analysis.                          |
| **Importance** | Essential for ensuring accurate historical price data.  | Necessary for preparing data for technical analysis and modeling. |
| **Example Tasks** | Removing days with missing closing prices. Correcting anomalies in historical data. | Scaling stock prices for comparability. Encoding stock symbols. |
| **Dependencies** | Often performed before technical analysis.            | Typically follows data cleaning in the trading data workflow. |
| **Outcome**    | A cleaned dataset with accurate historical stock prices. | A preprocessed dataset ready for technical analysis or algorithmic trading. |

## Data preparation vs data preprocessing
| Aspect         | Data Preparation                                                                                   | Data Preprocessing                                                                       |
|----------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Objective**  | Prepare raw data for analysis or modeling.                                                         | Transform and enhance data for improved analysis or modeling.                            |
| **Example Tasks** | Collecting data from various sources, combining data from multiple datasets, aggregating data at different levels, and splitting data into training and testing sets. | Imputing missing values in a specific column, scaling numerical features for machine learning models, and encoding categorical variables for analysis. |
| **Scope**      | Broader term encompassing various activities.                                                      | A subset of data preparation, focusing on specific transformations.                      |
| **Tasks**      | Data collection, data cleaning, data integration, data transformation, data reduction, and data splitting. | Handling missing data, scaling features, encoding categorical variables, handling outliers, and feature engineering. |
| **Importance** | Essential for ensuring data availability and organization.                                         | Necessary for preparing data to improve analysis or model performance.                   |
| **Dependencies** | Often precedes data preprocessing in the overall data workflow.                                   | Follows data collection and is closely related to data cleaning.                         |
| **Outcome**    | Well-organized dataset ready for analysis or modeling.                                             | Preprocessed dataset optimized for specific analytical or modeling tasks.                |


## Data preprocessing vs feature engineering
Data preprocessing involves tasks such as handling missing data and scaling, while feature engineering focuses on creating new features or modifying existing ones to improve the predictive power of machine learning models.
| Aspect         | Data Preprocessing                                                                        | Feature Engineering                                                                    |
|----------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Objective**  | Transform and enhance raw data for analysis or modeling.                                  | Create new features or modify existing ones for improved model performance.            |
| **Example Tasks** | Imputing missing values and scaling numerical features.                                | Creating a feature for the ratio of two existing features and adding polynomial features. |
| **Scope**      | Subset of data preparation, focusing on data transformations.                             | Specialized tasks within data preparation, focusing on feature creation or modification. |
| **Tasks**      | Handling missing data, scaling and normalization, encoding categorical variables, handling outliers, and data imputation. Data preprocessing is a broader term which includes the tasks of data cleaning and data preparation as well. | Creating new features based on existing ones, polynomial features, interaction terms, and dimensionality reduction. |
| **Importance** | Necessary for preparing data for analysis or modeling.                                    | Enhances predictive power by introducing relevant features.                            |
| **Dependencies** | Typically follows data cleaning and precedes model training.                           | Often follows data preprocessing and precedes model training.                          |
| **Outcome**    | A preprocessed dataset ready for analysis or modeling.                                    | A dataset with engineered features optimized for model performance.                    |
