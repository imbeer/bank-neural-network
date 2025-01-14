import train
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

# labels =[row[len(row) - 1] for row in training_data] # результаты
# for row in training_data: row.pop()

weights_input_hidden, weights_hidden_output = train.train(training_data)
# обучено до 0.112 точности

with open('weights.txt', 'w') as f:
    f.write(' '.join(weights_input_hidden) + "\n")
    f.write(' '.join(weights_hidden_output) + "\n")
# weights_input_hidden = []
# weights_hidden_output = []
# with open ('weights.txt', 'r') as f:
#     for _ in range(4):
#         line = f.readline()
#         weights_input_hidden.append(list(map(float, line.split(' '))))
#     for _ in range(4):
#         line = f.readline()
#         weights_hidden_output.append(float(line))

print(weights_input_hidden)
print(weights_hidden_output)

# print(network.test(weights_input_hidden, weights_hidden_output, test_data))
