import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Load your data from the ChatCSV file
data = pd.read_csv("ChatCSV.csv")

# Select relevant features and target variable
features = data[['common_words', 'sequence_similarity', 'length_difference', 'contains_response_words']]
target = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize a Logistic Regression model
model = LogisticRegression()

# Train the model for different iterations and plot the learning curve
iterations = [10, 50, 100, 200, 500, 1000]
train_losses = []
test_losses = []

for iteration in iterations:
    # Train the model
    model.fit(X_train, y_train)

    # Predictions on training and testing sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Calculate accuracy and F1 score
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    # Append accuracy to the lists for plotting
    train_losses.append(1 - train_acc)  # error rate
    test_losses.append(1 - test_acc)  # error rate

    # Print iteration, model coefficients, and error
    print(f"Iteration: {iteration}, Coefficients: {model.coef_}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Plot the learning curve
plt.plot(iterations, train_losses, label='Training Error')
plt.plot(iterations, test_losses, label='Testing Error')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.title('Learning Curve for Logistic Regression')
plt.legend()
plt.show()
