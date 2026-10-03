import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X = np.random.rand(200, 1) * 10
y = (X.flatten() > 5).astype(int)

print("Before noise:", y[:5])

flip_indices = np.random.choice(200, 30, replace=False)
y[flip_indices] = 1 - y[flip_indices]

print("After noise:", y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# "Barely studied" - extremely shallow tree, max_depth=1
barely_studied = DecisionTreeClassifier(max_depth=1, random_state=1)
barely_studied.fit(X_train, y_train)
print(
    "\nBarely studied - Train acc:",
    accuracy_score(y_train, barely_studied.predict(X_train)),
)
print(
    "Barely studied - Test acc:", accuracy_score(y_test, barely_studied.predict(X_test))
)
# "Memorized the exact answers" - unlimited tree depth
memorized = DecisionTreeClassifier(max_depth=None, random_state=1)
memorized.fit(X_train, y_train)
print("\nMemorized - Train acc:", accuracy_score(y_train, memorized.predict(X_train)))
print("Memorized - Test acc:", accuracy_score(y_test, memorized.predict(X_test)))
# "Actually learned the rule" - reasonable depth, not too shallow, not unlimited
learned_properly = DecisionTreeClassifier(max_depth=3, random_state=1)
learned_properly.fit(X_train, y_train)
print(
    "\nLearned properly - Train acc:",
    accuracy_score(y_train, learned_properly.predict(X_train)),
)
print(
    "Learned properly - Test acc:",
    accuracy_score(y_test, learned_properly.predict(X_test)),
)
