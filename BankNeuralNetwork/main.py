import network

data = [[]]
with open('data.csv', 'r') as dataset:
    for line in dataset:
        data.append(list(map(int, line.split(';'))))

data = data[1:]

dataset_len = len(data)
split_index = int(dataset_len * 0.7)

training_data = data[:split_index]
test_data = data[split_index:]

print(training_data)

# обучение
weights_input_hidden, weights_hidden_output = network.train(training_data, learning_rate=0.01, max_epochs=10000, error_threshold=0.1)

with open('weights.txt', 'w') as f:
    f.write(' '.join(map(str, weights_input_hidden)) + "\n")
    f.write(' '.join(map(str, weights_hidden_output)) + "\n")


# проверка
with open ('weights.txt', 'r') as f:
    line = f.readline()
    weights_input_hidden = list(map(float, line.split(' ')))
    line = f.readline()
    weights_hidden_output = list(map(float, line.split(' ')))

print(weights_input_hidden)
print(weights_hidden_output)

print(network.test(weights_input_hidden, weights_hidden_output, test_data))
