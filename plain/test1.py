from math import exp

w_h1 = [
    [0.1, 0.8],
    [0.4, 0.6],
    [-0.4, -0.6],
]

w_out = [
    [0.2, 0.3, 0.9]
]

def sigmoid(x):
    return 1 / (1 + exp(-x))

def forward(input, weights):
    result = []
    for w_neurons in weights:
        sum = 0
        for i in range(len(input)):
            sum += w_neurons[i]*input[i]
        result.append(sigmoid(sum))
    return result

def calculate_output_error(output, target):
    error = []
    for (i, t) in enumerate(target):
        o = output[i]
        error.append(o * (1 - o) * (t - o))
    return error

def calculate_hidden_layer_error(output, weights, error):
    result = [];
    for (i, o) in enumerate(output):
        error_factor = 0
        for (j, e) in enumerate(error):
            error_factor += weights[j][i] * e
        result.append(o * (1 - o) * error_factor)
    return result

def backpropagate(weights, input, output, error, learning_rate):
    new_weights = []
    for (i, w_neurons) in enumerate(weights):
        error_neuron = error[i]
        new_w_neurons = []
        for (j, w) in enumerate(w_neurons):
            new_w_neurons.append(w + error_neuron * input[j] * learning_rate)
        new_weights.append(new_w_neurons)
    return new_weights

def get_total_error_for(output, target):
    result = 0;
    for i in range(len(output)):
        result += (output[i] - target[i]) * (output[i] - target[i])
    return result


# This is specific to this network
def run_network(input):
    return forward(forward(input, w_h1), w_out)

def train(input, target, learning_rate):
    global w_out, w_h1

    hidden_layer = forward(input, w_h1)
    output = forward(hidden_layer, w_out)
    output_error = calculate_output_error(output, target)
    w_out = backpropagate(w_out, hidden_layer, output, output_error, learning_rate)
    hidden_layer_error = calculate_hidden_layer_error(hidden_layer, w_out, output_error)
    w_h1 = backpropagate(w_h1, input, hidden_layer, hidden_layer_error, learning_rate)


    return get_total_error_for(run_network(input), target)


# XOR!
inputs = [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
]

targets = [
    [1],
    [1],
    [0],
    [0]
]

#train([0.35, 0.9], [0.5], 1)

for epoch in range(10000):
    sum_error = 0
    for i in range(len(inputs)):
        sum_error += train(inputs[i], targets[i], 0.1)
    print("error", sum_error)

for i in inputs:
    print (i, run_network(i))#