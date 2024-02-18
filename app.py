import sys
import json
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train the Gradient Boosting model
gradient_boosting = GradientBoostingClassifier(random_state=42)
gradient_boosting.fit(X, y)

sepal_length, sepal_width, petal_length, petal_width = map(float, sys.argv[1:])

# Make a probability prediction
probabilities = gradient_boosting.predict_proba([[sepal_length, sepal_width, petal_length, petal_width]])[0]

# Create a dictionary of class names and their corresponding probabilities
class_probabilities = {iris.target_names[i]: probabilities[i] for i in range(len(iris.target_names))}

# Output the prediction as a JSON-formatted string
print(json.dumps({'class_probabilities': class_probabilities}, indent=4))