import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Features: [hours_studied, hours_slept]
# Label: 1 = pass, 0 = fail
X = np.array([[1, 4], [2, 5], [3, 4], [4, 6], [5, 7], [6, 6], [7, 8], [8, 7]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

prediction = model.predict([[4.5, 6]])
print("Prediction for 4.5hrs studied, 6hrs slept:", prediction)

plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=["hours_studied", "hours_slept"],
    class_names=["fail", "pass"],
    filled=True,
)
plt.savefig("week2/day4/tree_plot.png")
print("Tree saved as image")
