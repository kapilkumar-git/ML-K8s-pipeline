"""
Trains a simple RandomForest classifier on the classic Iris dataset
and saves it as model.pkl. This runs once at image build time so the
container ships with a ready-to-serve model (no training at runtime).
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

joblib.dump(clf, "model.pkl")
print("Model trained and saved to model.pkl")
print("Target names:", iris.target_names.tolist())
