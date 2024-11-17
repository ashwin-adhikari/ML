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