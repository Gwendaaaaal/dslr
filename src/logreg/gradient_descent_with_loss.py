from lib import dot_product, sigmoid, transpose_matrix
from math import log

learning_rate = 0.001
iterations = 10000
loss_interval = 1000


def compute_loss(predictions: list[float], Y: list[float]) -> float:
    """Return the average binary cross-entropy loss."""

    epsilon = 1e-15
    loss = 0.0
    for prediction, actual in zip(predictions, Y):
        prediction = max(epsilon, min(1 - epsilon, prediction))
        loss -= (
            actual * log(prediction)
            + (1 - actual) * log(1 - prediction)
        )
    return loss / len(Y)


def print_loss_plot(loss_iterations: list[int], losses: list[float]) -> None:
    """Print a horizontal loss chart to stdout."""

    if not losses:
        return

    chart_width = 50
    max_loss = max(losses)
    iteration_width = len(str(max(loss_iterations)))

    print("\nLoss during gradient descent")
    for iteration, loss in zip(loss_iterations, losses):
        bar_width = round(loss / max_loss * chart_width) if max_loss else 0
        bar = "#" * bar_width
        print(f"{iteration:>{iteration_width}} | {bar:<{chart_width}} {loss:.6f}")


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
        f. Compute the log loss every 1,000 iterations and print it
            as an ASCII chart when training finishes.
    """

    loss_iterations: list[int] = []
    losses: list[float] = []

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

        should_stop = False
        if i > 0:
            max_change = max(
                abs(W[idx] - previous_W[idx])
                for idx in range(len(W))
            )

            if max_change < 1e-4:
                should_stop = True

        completed_iterations = i + 1
        if (
            completed_iterations % loss_interval == 0
            or should_stop
            or completed_iterations == iterations
        ):
            updated_predictions = [
                sigmoid(elem) for elem in dot_product(X, W)
            ]
            loss_iterations.append(completed_iterations)
            losses.append(compute_loss(updated_predictions, Y))

        if should_stop:
            break

    print_loss_plot(loss_iterations, losses)
    return W
