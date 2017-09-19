from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs.
		random.seed(2)

		# We model a single neuron, with 3 input connections and 1 output connection.
		# We assign random weights to a 3x1 matrix, with values in the range -1 to 1
		# and mean 0

		self.synaptic_weights = 2*random.random((3,1)) - 1

	# We define the sigmoid funcion as usual
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_input, training_set_outupts, iter_number):
		for iteration in range(iter_number):
			output = self.think(training_set_inputs)
			error = training_set_outputs - output;
			adjustment = dot(training_set_input.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustment
	def think(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
	neural_network = NeuralNetwork()
	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights

	training_set_inputs = array([[0, 0, 1], [1,1,1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T

	neural_network.train(training_set_inputs, training_set_outputs, 50000)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights