import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Feature Scaling
def scale_features(features):
    # Standardization (subtract mean and divide by standard deviation)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    scaled_features = (features - mean) / std
    return scaled_features, mean, std

# Feature Extraction
def extract_features(messages):
    # Add a column of ones for the bias term
    bias = np.ones((messages.shape[0], 1))
    features = np.hstack((bias, messages))
    return features

# Labeling
def predict(features, weights):
    # Linear prediction: h(x) = w0*x0 + w1*x1 + ... + wn*xn
    return np.dot(features, weights)

def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def calculate_loss(predictions, target):
    # Binary cross-entropy loss
    epsilon = 1e-15  # for numerical stability
    loss = -target * np.log(predictions + epsilon) - (1 - target) * np.log(1 - predictions + epsilon)
    return np.mean(loss)

def calculate_accuracy(predictions, target):
    # Convert predicted probabilities to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    # Calculate accuracy
    accuracy = np.mean(binary_predictions == target)
    return accuracy

def gradient_descent(features, target, weights, learning_rate, iterations):
    losses = []
    accuracies = []

    for iteration in range(iterations):
        # Make predictions
        predictions = sigmoid(predict(features, weights))

        # Calculate the gradient
        gradient = np.dot(features.T, (predictions - target)) / len(target)

        # Update weights
        weights -= learning_rate * gradient

        # Calculate loss and accuracy for monitoring convergence
        loss = calculate_loss(predictions, target)
        accuracy = calculate_accuracy(predictions, target)
        losses.append(loss)
        accuracies.append(accuracy)

        print(f"Iteration: {iteration}, Coefficients: {weights}, Loss: {loss}, Accuracy: {accuracy}")

    return losses, accuracies

# Load your data from the ChatCSV file
data = pd.read_csv("ChatCSV.csv")
features = data[['common_words', 'sequence_similarity', 'length_difference', 'contains_response_words']]
target = data['y'].values  # convert to NumPy array

# Feature scaling
scaled_features, mean, std = scale_features(features.values)

# Feature extraction
X = extract_features(scaled_features)

# Initialize weights
weights = np.zeros(X.shape[1])

# Experiment with Different Step Sizes
learning_rates = [0.01, 0.1, 0.5, 1.0]
iterations = 1000

plt.figure(figsize=(10, 6))

for learning_rate in learning_rates:
    # Clone weights to keep the initial weights for each learning rate
    weights_copy = np.copy(weights)

    # Run gradient descent
    losses, accuracies = gradient_descent(X, target, weights_copy, learning_rate, iterations)

    # Plot learning curve and accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(iterations), losses, label=f'Learning Rate: {learning_rate}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Learning Curves for Different Step Sizes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), accuracies, label=f'Learning Rate: {learning_rate}')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Step Sizes')
    plt.legend()

plt.tight_layout()
plt.show()
