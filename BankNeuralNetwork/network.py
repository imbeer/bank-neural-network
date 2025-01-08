import numpy as np
from train import sigmoid


# data - массив-строка в формате из data.csv без последнего элемента
def get_result(weights_input_hidden, weights_hidden_output, data):
    data = np.array(data)
    hidden_input = np.dot(data, weights_input_hidden)  # (N, hidden_neurons)
    hidden_output = sigmoid(hidden_input)  # (N, hidden_neurons)

    final_input = np.dot(hidden_output, weights_hidden_output)  # (N, output_neurons)
    final_output = sigmoid(final_input)
    return final_output

def test(weights_input_hidden, weights_hidden_output, test_data):
    total_error = 0
    for testLine in test_data:
        testResultTrue = testLine[-1]
        data = testLine[:-1]
        testResultPred = get_result(weights_input_hidden, weights_hidden_output, data)
        total_error += (testResultPred - testResultTrue) ** 2
    total_error /= len(test_data)
    return total_error