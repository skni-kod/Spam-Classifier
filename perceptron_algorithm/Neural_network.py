from perceptron_algorithm.Neuron import Neuron
from feature_extraction.Vocabulary import num_of_initial_neurons, vectorized_emails

"""
Todo:
    - Każdy neuron dostaje jedną i tą samą liste na początek, potem następne warstwy dostają wyniki poprzedniej na zasadzie każdy do każdego
    - obliczanie błędu 
    - zbieranie danych warstwy poprzedniej 
"""


class NeuralNetwork:
    def __init__(self, input: list[list[int]], layer_output=None, convergence_criteria=0.1,
                 num_of_neurons=num_of_initial_neurons):
        # podajemy cały set
        self.input = input
        self.layer_output = layer_output
        self.convergence_criteria = convergence_criteria
        self.num_of_neurons = num_of_neurons

    # TODO - dodać kalkulacje straty na każdej warstwie - porównać z gotowymi klasami
    def hinge_loss(self, prediciton, layer_output):
        return max(self.convergence_criteria, 1 - prediciton * layer_output)

    def create_first_layer(self):
        # Tworzenie warstwy neuronów
        first_layer = {}
        i = 1
        for mail in self.input:
            n = Neuron(input=mail, id=i)
            first_layer[n.id] = n.weighted_sum()
            i += 1
        self.layer_output = first_layer

    def create_hidden_layers(self):
        pass

    def create_last_layer(self):
        pass

    def propagation(self):
        pass

    def train(self):
        pass


network = NeuralNetwork(input=vectorized_emails)
network.create_first_layer()
print(network.layer_output)
