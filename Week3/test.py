#import relevant libraries
import numpy as np
import pandas as pd
from scipy import stats

covid_data = pd.read_csv(r"C:\Users\Ripple\Desktop\ML\Week3\covid_data.csv")
print(covid_data.head())