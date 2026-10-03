import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Fake data: two features on very different scales
ages = np.array([22, 38, 26, 35, 60]).reshape(-1, 1)
fares = np.array([7.25, 71.28, 7.92, 53.1, 263.0]).reshape(-1, 1)

print("Original ages:", ages.flatten())
print("Original fares:", fares.flatten())

# Min-Max normalization: squish to 0-1
minmax = MinMaxScaler()
ages_normalized = minmax.fit_transform(ages)
fares_normalized = minmax.fit_transform(fares)

print("\nNormalized ages:", ages_normalized.flatten())
print("Normalized fares:", fares_normalized.flatten())
standard = StandardScaler()
ages_standardized = standard.fit_transform(ages)
print("\nStandardized ages:", ages_standardized.flatten())
print("Mean of standardized ages:", ages_standardized.mean())
import pandas as pd

embarked_sample = pd.DataFrame({"embarked": ["S", "C", "S", "Q", "C"]})
print(embarked_sample)

one_hot = pd.get_dummies(embarked_sample["embarked"])
print("\nOne-hot encoded:\n", one_hot)
