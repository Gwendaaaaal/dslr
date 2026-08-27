from lib import dot_product, sigmoid, transpose_matrix
from math import log

learning_rate = 0.001
iterations = 10000

def compute_gradient(
    X_transposed: list[list[float]],
    error: list[float],
    m: int
) -> list[float]:
    """
    Compute the gradient:
        gradient = (1/m) * X_scaled_with_bias.T . error
        (m = number of training examples)
    """

    res = dot_product(X_transposed, error)
    for idx, elem in enumerate(res):
        res[idx] /= m

    return res
    
def update_weights(
    W: list[float],
    gradient: list[float]
) -> list[float]:
    """
    Update weights:
        Weight_i = Weight_i - learning_rate * gradient_i
    Returns:
        Updated weights
    """
    new_W = []
    for grad, old_W in zip(gradient, W):
        new_W.append(old_W - learning_rate * grad)
    return new_W
    

def gradient_descent(
    X: list[list[float]],
    Y: list[float],
    W: list[float]
) -> list[float]:
    """
    Repeat for a fixed number of iterations (or until convergence):
        a. Compute z = X_biased . weights  (dot product)
        b. Compute predictions: h = sigmoid(z)
        c. Compute the error: error = h - y_house
        d. Compute the gradient:
            gradient = (1/m) * X_scaled_with_bias.T . error
            (m = number of training examples)
        e. Update weights:
            weights = weights - learning_rate * gradient
        f. (Optional but recommended) Compute and store the cost
            (log loss) at each iteration so you can plot it later
            and check that it's decreasing (sanity check for debugging)
    """

    accs = [0.0 for i in range(10)]
    for i in range(0, iterations, 1):
        # a. Compute z = X_biased . weights   (dot product)
        Z: list[float] = dot_product(X, W)

        # b. Compute predictions: h = sigmoid(z)
        h: list[float] = [sigmoid(elem) for elem in Z]

        # c. Compute the error: error = h - y_house
        error: list[float] = [
            prediction - is_house
            for prediction, is_house in zip(h, Y)
        ]

        # d. Compute the gradient:
        #   gradient = (1/m) * X_scaled_with_bias.T . error
        #   (m = number of training examples)
        X_transposed = transpose_matrix(X)
        m = len(X)
        gradient: list[float] = compute_gradient(X_transposed, error, m)

        # e. Update weights:
        #   Weight_i = Weight_i - learning_rate * gradient_i
        previous_W = W.copy()
        W = update_weights(W, gradient)

        if i > 0:
            max_change = max(
                abs(W[idx] - previous_W[idx])
                for idx in range(len(W))
            )

            if max_change < 1e-4:
                return W
        
        # f. see
        loss = 0.0

#        for prediction, actual in zip(h, Y):
#            prediction = max(1e-15, min(1 - 1e-15, prediction))

#            loss -= (
#                actual * log(prediction)
#                + (1 - actual) * log(1 - prediction)
#            )

#        loss /= len(Y)

        if i % 1000 == 0:
            print(i, loss)
        correct = 0

        for prediction, actual in zip(h, Y):
            predicted = 1 if prediction >= 0.5 else 0

            if predicted == actual:
                correct += 1

        accuracy = correct / len(Y)
        print(f"iterations = {i}, accuracy = {accuracy}, accs = {accs}")
        
    return W
