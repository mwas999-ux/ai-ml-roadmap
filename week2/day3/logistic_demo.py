import numpy as np
from sklearn.linear_model import LogisticRegression

# Pretend: hours studied vs pass/fail (1 = pass, 0 = fail)
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(hours_studied, passed)

# Predict probability of passing for 3.5 hours studied
prob = model.predict_proba([[3.5]])
print("Probability [fail, pass] for 3.5 hours:", prob)

# Predict the actual class (0 or 1)
prediction = model.predict([[3.5]])
print("Predicted class for 3.5 hours:", prediction)

# Try a few more values
for h in [1, 3, 4.5, 5, 8]:
    p = model.predict_proba([[h]])[0][1]  # probability of "pass"
    print(f"{h} hours -> {p:.3f} probability of passing")
