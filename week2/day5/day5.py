import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Same study data as yesterday, but with a trickier new point mixed in
X = np.array(
    [
        [1, 4],
        [2, 5],
        [3, 4],
        [4, 6],
        [5, 7],
        [6, 6],
        [7, 8],
        [8, 7],
        [4, 8],
        [5, 3],  # two "unusual" cases - lots of sleep but little study, vice versa
    ]
)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X, y)

# Predict the same tricky point from yesterday
prediction = forest.predict([[4.5, 6]])
probability = forest.predict_proba([[4.5, 6]])
print("Prediction:", prediction)
print("Probability [fail, pass]:", probability)

# Feature importance - which feature did the forest rely on more?
print("Feature importance [hours_studied, hours_slept]:", forest.feature_importances_)
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss")
xgb_model.fit(X, y)

prediction = xgb_model.predict([[4.5, 6]])
probability = xgb_model.predict_proba([[4.5, 6]])
print("\nXGBoost Prediction:", prediction)
print("XGBoost Probability [fail, pass]:", probability)
print("XGBoost feature importance:", xgb_model.feature_importances_)
