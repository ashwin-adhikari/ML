# Vector in python ###

> ``x = np.array([1,2,3,4])``
> here x is a vector.

### Addition and Subtraction
> ``x1 = np.array([3,4])``
> ``x2 = np.array([1,2])``
> sum = x1+x2 
> or sum = np.add(x1,x2)
> diff = x1-x2

### Multiplication
Scalar multiplication plays a pivotal role in adjusting the learning rate. For instance, when training a neural network, scaling the learning rate dynamically based on the progress of training helps avoid overshooting or slow convergence, ensuring effective model training.

> scaler_mul = x1 * 2
> Output : [6,8]

#### Calculating magnitude and direction
>``magnitude = np.linalg.norm(x1)``
> ``linalg`` is linear algebra module and ``norm`` is normal is used to calculate magnitude of x1 i.e $\sqrt{(3^2 + 4^2)}$
>``direction = np.arctan2(x1[1], x1[0])``<br>
> ``arctan`` is inverse of tan and above expression is ${tan^-1(x1[1]/x1[0])}$ 


> Magnitude of x1: 5.0
> Direction of x1: 0.9272952180016122 radians

### Dot product
Dot products are fundamental for text classification in natural language processing (NLP). By calculating dot products between word vectors and predefined sentiment vectors, you can determine the sentiment of a text. This enables automated sentiment analysis for large volumes of text data.

>dot_product = np.dot(x1, x2)
> x1_3D = np.append(x1, 0)
> x2_3D = np.append(x2, 0)
>``append`` is used to add element at end of the array or vector. We can add **only one element** or a variable 
> ``x1_3D = np.append(x1,x2)``
>Output = [3,4,1,2]

### Cross Product
Cross products are crucial in 3D graphics and physics, particularly for calculating normals to surfaces, which are essential for lighting and rendering. They are also used in physics for finding torques and rotational vectors, helping to understand the dynamics of rotating bodies.

>cross_product = np.cross(x1_3D, x2_3D)<br>

> **TO PERFORM CROSS PRODUCT VECTOR LENGTH MUST BE EQUAL**



**Vector norms** are used in machine learning for regularization and similarity calculations.


To calculate the L2 norm of a vector in Python, you can use the numpy library:

>vector_v = np.array([3, 4])
>l2_norm = np.linalg.norm(vector_v)

## Real-world Application: Vector in Computer Graphics

- Understanding vector representation and manipulation in digital space is fundamental in computer graphics and game development.
- Predicting object movements under various conditions is crucial in mechanical and aerospace engineering.

# Matrix

>Created using: `` x1 = np.array([1,2,3],[4,5,6])``
>``y1 = np.array([[7, 8, 9], [10, 11, 12]])``
> print(x1.shape,'\n') 
The *.shape after the variable, allows you to verify the shape of the matrix.

#### output:

```
[[1 2 3]
 [4 5 6]]
(2, 3) 
```

> Addition is same as of vector
```
addition = np.add(x1,y1) or x1+y1
sub = np.subtract(x1,y1) or x1-y1
```
### Multiplication
```
m_sca = 5 * x1 #Scalar multiplication
m_mul = x1@z1
@ is used to for matrix multiplication
```

### Transpose
```
X = np.array([[1, 3, 5], [7, 9, 11], [13, 15, 17]])
X_T = X.transpose()

Output:
[[ 1  7 13]
 [ 3  9 15]
 [ 5 11 17]]
```

### Inverse
```
X = np.array([[1, 3, 5], [7, 9, 11], [13, 15, 17]])
X_inv = np.linalg.inv(X)

Output:
[[-2.81474977e+14  5.62949953e+14 -2.81474977e+14]
 [ 5.62949953e+14 -1.12589991e+15  5.62949953e+14]
 [-2.81474977e+14  5.62949953e+14 -2.81474977e+14]]

```

# Derivatives

Derivatives are the fundamental tools of Calculus. It is very useful for optimizing a loss function with **_gradient descent_** in Machine Learning is possible only because of derivatives.

Python ``SymPy`` library is created for symbolic mathematics. The SymPy project aims to become a full-featured computer algebra system (CAS) while keeping the code simple to understand. Let’s see how to calculate derivatives in Python using SymPy.

```
import sympy as sp

# create a "symbol" called x
x = Symbol('x')
 
#Define function
f = x**2
 
#Calculating Derivative
derivative_f = f.diff(x)
```

### Applications of Derivatives

In Machine Learning and AI:
- Optimization: <br>
 Derivatives are fundamental in optimizing cost functions in algorithms like Gradient Descent.
- Training Neural Networks: <br>
 The backpropagation algorithm, used in training neural networks, utilizes derivatives to compute gradients for weight updates.

- Sensitivity Analysis: <br>Derivatives help in understanding how changes in input affect the output, aiding in feature selection and model understanding.

- Model Behavior Analysis:<br> In algorithms like SVM, derivatives help formulate the optimization problem, and in decision trees, they can assist in node splitting and feature importance analysis.


# Integrals

Integrals find extensive applications in ML and AI for:

- Optimization Problems: Integrals are used in calculating areas under the curve in ROC analysis, integral representations in loss functions, and in regularization terms to prevent overfitting.

- Understanding Data Distributions: Integrals help in defining continuous probability distributions, essential for calculating probabilities and expectations in statistical models.

- Deep Learning: In neural networks, integrals appear in backpropagation algorithms for computing gradients and in defining activation functions.


``sympy`` is used for integrals as well.
```
from sympy import *
import sympy as sp

x = sp.symbols('x')
function = x**2

integral = function.integrate(x)
```

>``lambdify`` converts a symbolic SymPy expression into a numerical function that can take specific values as inputs.


>``linspace`` generates a specified number of equally spaced values between two endpoints.



# Probability and statistics

In python they are implemented via these libraries
```
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
```

>``scipy.stats (stats)``: Provides functions for statistical analysis, including measures of central tendency, dispersion, and hypothesis testing.

> ``pandas (pd)``: Used for data manipulation and analysis.

## 1. Sample Data and Central Tendency Calculations
```
# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Central Tendency
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0]
```
>**Mean** (``np.mean(data)``): Calculates the average of data.

>**Median** (``np.median(data)``): Finds the middle value of data when sorted.

>**Mode** (``stats.mode(data)[0]``): Calculates the mode (most frequently occurring value) in data. The [0] extracts the mode value itself from the output tuple returned by stats.mode.

## 2. Dispersion Calculations
```
range_ = np.ptp(data)
variance = np.var(data)
std_dev = np.std(data)
```
>**Range** (``np.ptp(data)``): Computes the range (peak-to-peak) as the difference between the maximum and minimum values.


> **Variance** (``np.var(data)``): Measures the average squared deviation from the mean, providing an idea of data spread.


>**Standard Deviation** ``(np.std(data))``: Measures the spread of data from the mean, in the same units as data, making it easier to interpret.


> Dispersion measures show how spread out data is. For example, high variance or standard deviation indicates data points that vary widely from the mean, while low values indicate data close to the mean.

## 3. Normal Distribution Example
```
# Parameters
mu, sigma = 0, 0.1

# Generating random values
s = np.random.normal(mu, sigma, 1000)
```

>**_mu (mean)_** and **_sigma (standard deviation)_** are set to 0 and 0.1, respectively.


> **Random Sampling**: ``np.random.normal(mu, sigma, 1000) ``generates 1000 values from a normal (Gaussian) distribution centered at mu with spread determined by sigma.

>The normal distribution is fundamental in statistics as many real-world phenomena follow it. This function allows us to simulate such data for analysis or testing.

## 4.  Binomial Distribution Example
```
# Parameters
n, p = 10, 0.5

# Generating random values
s = np.random.binomial(n, p, 1000)
```
>**_n (number of trials)_** and **_p (probability of success)_** define a binomial distribution with 10 trials and a success probability of 0.5.


>**Random Sampling**: ``np.random.binomial(n, p, 1000)`` generates 1000 values from this binomial distribution.

>The binomial distribution is useful for modeling scenarios with binary outcomes (e.g., success/failure) in repeated trials. It helps in analyzing discrete random events.


## 5. Poisson Distribution Example
```
# Parameters
lambda_ = 5

# Generating random values
s = np.random.poisson(lambda_, 1000)
```
>``lambda_`` represents the expected number of occurrences in a fixed interval (mean rate of events).


>**Random Sampling**: ``np.random.poisson(lambda_, 1000)`` generates 1000 values from a Poisson distribution with mean ``lambda_``

>The Poisson distribution is valuable for modeling rare events, like customer arrivals in queues or call frequencies. It applies to counts or events in fixed intervals.

## 6. Inferential Statistics - Hypothesis Testing Example
```
# Parameters
mu_0 = 5
alpha = 0.05  # Significance level

# T-test
t_statistic, p_value = stats.ttest_1samp(data, mu_0)
```
>**Null Hypothesis (H0)**: The sample data has a population mean of ``mu_0`` (5 in this case).


>**Alternative Hypothesis (H1)**: The sample data does not have a mean of mu_0.


>**Significance Level** (``alpha = 0.05``): Sets a threshold (5%) for rejecting the null hypothesis.


>**T-test (``stats.ttest_1samp(data, mu_0)``)**: Performs a one-sample t-test to test if the mean of data differs from mu_0.


> A **t-test** is an inferential statistic used to determine if there is a significant difference between the means of two groups and how they are related.

>Hypothesis testing is essential for making inferences about populations based on sample data. The t-test checks if data's mean significantly differs from mu_0, helping determine if observed differences are likely due to random variation or indicate a meaningful effect.
