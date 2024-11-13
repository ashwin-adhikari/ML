import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Central Tendency
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0]

# Dispersion
range_ = np.ptp(data)
variance = np.var(data)
std_dev = np.std(data)

print(f"Mean: {mean},\n Median: {median},\n Mode: {mode},\n Range: {range_},\n Variance: {variance},\n Standard Deviation: {std_dev}\n") 


# Normal Distribution Example
# Parameters
mu, sigma = 0, 0.1

# Generating random values
random_n = np.random.normal(mu, sigma, 1000)

# Plotting
count, bins, ignored = plt.hist(random_n, 30, density=True)
    #plt.hist(s, 30, density=True) plots a histogram of the normal data. density=True normalizes the histogram, so its area is 1.
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.title('Normal Distribution')
# plt.show()

# Binomial Distribution Example
# Parameters
n, p = 10, 0.5

# Generating random values
random_b = np.random.binomial(n, p, 1000)

# Plotting
plt.hist(random_b, bins=30, density=True)
plt.title('Binomial Distribution')
# plt.show()


# Poisson Distribution Example
# Parameters
lambda_ = 5

# Generating random values
random_p = np.random.poisson(lambda_, 1000)

# Plotting
plt.hist(random_p, bins=30, density=True)
plt.title('Poisson Distribution')
plt.show()

# Inferential Statistics
# Hypothesis Testing Example
# Null Hypothesis: The sample comes from a population with a mean of μ_0
# Alternative Hypothesis: The sample does not come from a population with a mean of μ_0

# Parameters
mu_0 = 5
alpha = 0.05  # Significance level

# T-test
t_statistic, p_value = stats.ttest_1samp(data, mu_0)

print(f"T-statistic: {t_statistic}, P-value: {p_value}")
print("Reject Null Hypothesis" if p_value < alpha else "Fail to Reject Null Hypothesis")