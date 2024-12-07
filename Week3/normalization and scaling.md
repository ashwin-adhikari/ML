# Background
Data normalization and scaling are essential preprocessing techniques. They ensure consistent data representation across features, improving the performance and interpretability of machine learning algorithms. By modifying feature scales, these techniques help in stabilizing model training and enhancing algorithm convergence speed.

## Scaling types
### Z-Score Normalization (Standardization)
Z-Score Normalization, also known as standardization, transforms data to have a mean of zero and a standard deviation of one. This technique is useful when the features have different units or when the data follows a Gaussian distribution (e.g., Linear Regression, Logistic Regression, Linear Discriminant Analysis).
- **Pros**:
    - Less affected by outliers.
    - Keeps useful information about outliers.
- **Cons**:
    - Not bound to a specific range.
- **Use Case**: Algorithms that assume data is Gaussian (e.g., Linear Regression, Gaussian Naive Bayes).
- **Formula**: 
   - ${Z = \frac{(X-\mu)}{\sigma}}$
    - By subtracting the mean, the data is centered around zero.
    - Dividing by the standard deviation scales the data so that the variance is normalized.
```python
def z_score_normalize(x, mean, std_dev):
    return (x - mean) / std_dev
```

### Min-Max Scaling
Min-Max Scaling rescales the data to a fixed range, typically [0, 1]. This method is useful when the data does not follow a Gaussian distribution and when the algorithm does not assume any distribution of the data.
- **Pros**:
    - Bound to a specific range.
    - Preserves relationships in data.

- **Cons**:
    - Sensitive to outliers.

- **Use Case**: Algorithms sensitive to input scale (e.g., Neural Networks, KNN).
- **Formula**: 
    - ${Xnorm = \frac{(X-Xmin)}{(Xmax-Xmin)}}$
    - For custom range ${[a,b]}$ :
    ${Xnorm = a + \frac{(X-Xmin)(b-a)}{Xmax-Xmin}}$

```python
def min_max_scale(x, x_min, x_max, a=0, b=1):
    return a + (x - x_min) * (b - a) / (x_max - x_min)
```