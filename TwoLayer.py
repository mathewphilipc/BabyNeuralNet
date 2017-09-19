from numpy import exp, array, random, dot

class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synapticweights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2

	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))
	def __sigmoid_derivative(self, x):
		return x * (1 - x)