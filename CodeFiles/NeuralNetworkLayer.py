class NeuralNetworkLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        layer_outputs = []
        for neuron_weights, neuron_bias in zip(self.weights, self.biases):
            neuron_output = 0
            for n_input, weight in zip(inputs, neuron_weights):
                neuron_output += n_input * weight
            neuron_output += neuron_bias
            layer_outputs.append(neuron_output)
        return layer_outputs

# Example usage:
if __name__ == "__main__":
    inputs = [1, 2, 3, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    biases = [2, 3, 0.5]
    layer = NeuralNetworkLayer(weights, biases)
    output = layer.forward(inputs)
    print(output)
