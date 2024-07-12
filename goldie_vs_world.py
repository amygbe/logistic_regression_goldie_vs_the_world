import numpy as np
from proj_fcns import load_images, sigmoid, initialize_parameters, plot_costs
import os

base_path = os.getcwd()

# Construct the relative path to the images folder
cat_folder = os.path.join(base_path, "images", "goldie_pics")
not_cat_folder = os.path.join(base_path, "images", "random_pics")

# Load images
cat_images, cat_labels = load_images(cat_folder, 1)
not_cat_images, not_cat_labels = load_images(not_cat_folder, 0)

# Combine the datasets
all_images = np.array(cat_images + not_cat_images)
all_labels = np.array(cat_labels + not_cat_labels)

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(all_labels))
all_images = all_images[shuffle_indices]
all_labels = all_labels[shuffle_indices]

# Split into training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(all_labels))
train_images = all_images[:split_index]
train_labels = all_labels[:split_index]
test_images = all_images[split_index:]
test_labels = all_labels[split_index:]

# Reshape
train_images_flatten = train_images.reshape(train_images.shape[0], -1).T
test_images_flatten = test_images.reshape(test_images.shape[0], -1).T

# Standardize 
train_set = train_images_flatten / 255.
test_set = test_images_flatten / 255.


def propagate(w, b, X, Y, lambda_reg):
    """
    Implements forward and backward propagation for the model.

    Parameters:
        w (ndarray): Weights, shape (dim, 1).
        b (float): Bias.
        X (ndarray): Input data, shape (dim, number of examples).
        Y (ndarray): True labels, shape (1, number of examples).
        lambda_reg (float): Regularization parameter.

    Returns:
        tuple: Gradients (dict with "dw" and "db") and cost.
    """
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Add epsilon to avoid log(0) issues
    epsilon = 1e-10
    A = np.clip(A, epsilon, 1 - epsilon)
    
    # Cost function
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost += (lambda_reg / m) * np.sum(np.abs(w))  # L1 Regularization term
    
    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T) + (lambda_reg / m) * np.sign(w)  # L1 Regularization term
    db = (1/m) * np.sum(A - Y)
    
    # Clip gradients to avoid massive gradients
    dw = np.clip(dw, -1, 1)
    db = np.clip(db, -1, 1)
    
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, lambda_reg, print_cost=False):
    """
    Optimizes parameters using gradient descent.

    Parameters:
        w (ndarray): Weights, shape (dim, 1).
        b (float): Bias.
        X (ndarray): Input data, shape (dim, number of examples).
        Y (ndarray): True labels, shape (1, number of examples).
        num_iterations (int): Number of iterations for optimization.
        learning_rate (float): Learning rate for gradient descent.
        lambda_reg (float): Regularization parameter.
        print_cost (bool): If True, prints cost every 100 iterations. Default is False.

    Returns:
        tuple: Optimized parameters (dict with "w" and "b"), gradients (dict with "dw" and "db"), and list of costs.
    """
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y, lambda_reg)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0 or i == num_iterations - 1:
            costs.append(cost)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {cost}")
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs
    


def predict(w, b, X):
    """
    Predicts labels for input data using learned parameters.

    Parameters:
        w (ndarray): Weights, shape (dim, 1).
        b (float): Bias.
        X (ndarray): Input data, shape (dim, number of examples).

    Returns:
        ndarray: Predicted labels, shape (1, number of examples).
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, lambda_reg=0.1, print_cost=False):
    """
    Builds and trains a logistic regression model, then evaluates it on the test set.

    Parameters:
        X_train (ndarray): Training data, shape (dim, number of training examples).
        Y_train (ndarray): Training labels, shape (1, number of training examples).
        X_test (ndarray): Test data, shape (dim, number of test examples).
        Y_test (ndarray): Test labels, shape (1, number of test examples).
        num_iterations (int): Number of iterations for optimization. Default is 2000.
        learning_rate (float): Learning rate for gradient descent. Default is 0.5.
        lambda_reg (float): Regularization parameter. Default is 0.1.
        print_cost (bool): If True, prints cost every 100 iterations. Default is False.

    Returns:
        dict: Dictionary containing model information including costs, predictions, parameters, learning rate, number of iterations, and regularization parameter.
    """
    
    w, b = initialize_parameters(X_train.shape[0]) 
    
    parameters, _, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, lambda_reg, print_cost) # gradient dosn't need to be assigned
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_train": Y_prediction_train, 
         "Y_prediction_test": Y_prediction_test, 
         "w": w, 
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "lambda_reg": lambda_reg}
    
    return d

# Reshape labels 
train_labels = train_labels.reshape(1, -1)
test_labels = test_labels.reshape(1, -1)

# Train the model
d = model(train_set, train_labels, test_set, test_labels, num_iterations=5000, learning_rate=0.001, lambda_reg=0.01, print_cost=True)

# Plot the costs
plot_costs(d["costs"])
