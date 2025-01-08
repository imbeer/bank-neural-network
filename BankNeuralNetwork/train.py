import numpy as np


# todo:
# матрица W входных весов
# вектор  H весов скрытого слоя
# sigmoid (SUM (sigmoid(W cross x) cross H)) -> R (result) вероятность -> либо да, либо нет
# error = MSE(y - R(W0, W1, W2, W3, H)) -> minimize
# SUM (W * X * H) === dot product


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def train(data, labels, learning_rate=0.2, max_epochs=10000, error_threshold=0.112):
    input_neurons = len(data[0])
    hidden_neurons = input_neurons
    output_neurons = 1

    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)

    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
    weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)

    epochs = 0
    error = float('inf')

    while error > error_threshold:
        hidden_input = np.dot(data, weights_input_hidden)  # (N, hidden_neurons)
        hidden_output = sigmoid(hidden_input)  # (N, hidden_neurons)

        final_input = np.dot(hidden_output, weights_hidden_output)  # (N, output_neurons)
        final_output = sigmoid(final_input)  # (N, output_neurons)

        # Ошибка
        errors = labels - final_output  # (N, output_neurons)
        total_error = mse(labels, final_output)

        # Градиенты
        grad_hidden_output = np.dot(hidden_output.T, errors * sigmoid_derivative(final_input)) / len(data)
        grad_input_hidden = np.dot(
            data.T,
            (np.dot(errors * sigmoid_derivative(final_input), weights_hidden_output.T) * sigmoid_derivative(hidden_input))
        ) / len(data)

        # Обновление весов
        weights_hidden_output += learning_rate * grad_hidden_output
        weights_input_hidden += learning_rate * grad_input_hidden

        epochs += 1

        if epochs % 1000 == 0:
            print(f"Epoch {epochs}, Error: {total_error}")
        error = total_error

    return weights_input_hidden, weights_hidden_output