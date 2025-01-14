import math, random

# функция активации
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# производная функции активации
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error(y_predicted, y_true):
    return (y_true - y_predicted) ** 2

# Обучает нейронную сеть. Возвращает веса для первого и второго слоя.
def train(data, learning_rate=0.01, max_epochs=10000, error_threshold=0.1):
    random.seed(42)

    w = [random.random() for _ in range(16)] # Матрица 4x4 развернутая в массив
    h = [random.random() for _ in range(4)] # Вектор 4X1

    epoch = 0
    total_error = float('inf')

    while (epoch < max_epochs) and (total_error > error_threshold):
        total_error = 0
        total_grad_w = [0] * 16
        total_grad_h = [0] * 4
        for line in data:
            x = line[:4]
            y_true = line[4]

            output_layer = output(w, h, x)
            y_predicted = sigmoid(output_layer)

            total_error += error(y_predicted, y_true)
            grad_w, grad_h = gradient(w, h, x, y_true, y_predicted, output_layer)

            for i in range(16): total_grad_w[i] += grad_w[i]
            for i in range(4): total_grad_h[i] += grad_h[i]

        for i in range(16):
            total_grad_w[i] /= len(data)
            w[i] -= learning_rate * total_grad_w[i]
        for i in range(4):
            total_grad_h[i] /= len(data)
            h[i] -= learning_rate * total_grad_h[i]

        total_error /= len(data)
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, error: {total_error}")
        epoch += 1
    return w, h

# Значение, которое получает на вход последний нейрон, до применения функции активации.
# Вынесено в отдельную функцию, потому что используется в нескольких местах.
def output(w, h, x):
    return (
            h[0]*sigmoid(w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]) +
            h[1]*sigmoid(w[4] * x[0] + w[5] * x[1] + w[6] * x[2] + w[7] * x[3]) +
            h[2]*sigmoid(w[8] * x[0] + w[9] * x[1] + w[10] * x[2] + w[11] * x[3]) +
            h[3]*sigmoid(w[12] * x[0] + w[13] * x[1] + w[14] * x[2] + w[15] * x[3])
    )

# результат на выходе из последнего слоя
def result(w, h, x):
    return sigmoid(output(w, h, x))

# Считает градиент аналитически.
# Градиент разбит на два массива для удобства:
# один для весов h (спрятанный слой -> выход), другой для весов w (слой входа -> спрятанный слой)
def gradient(w, h, x, y_true, y_predicted, output_layer):
    error_derivative =  2 * (y_predicted - y_true)
    grad_h = [error_derivative * sigmoid_derivative(output_layer) * sigmoid(
        w[0 + 4 * i] * x[0] +
        w[1 + 4 * i] * x[1] +
        w[2 + 4 * i] * x[2] +
        w[3 + 4 * i] * x[3]
    ) for i in range(4)]
    grad_w = [error_derivative * sigmoid_derivative(output_layer) * sigmoid_derivative(
        w[4 * (i // 4) + 0] * x[0] +
        w[4 * (i // 4) + 1] * x[1] +
        w[4 * (i // 4) + 2] * x[2] +
        w[4 * (i // 4) + 3] * x[3]
    ) * x[i % 4] * h[i // 4] for i in range(16)]
    return grad_w, grad_h

# Тестирует на test_data. Возвращает среднюю квадратичную ошибку результата.
def test(w, h, test_data):
    total_error = 0
    for testLine in test_data:
        test_result_true = testLine[-1]
        data = testLine[:-1]
        test_result_prediction = result(w, h, data)
        total_error += (test_result_prediction - test_result_true) ** 2
    total_error /= len(test_data)
    return total_error