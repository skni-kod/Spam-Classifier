# Taking data and classes to giv to model
from data_preparation.data_loading import data_classes
# Taking vextorized emails [[0,1,0,1],[...],...]
from feature_extraction.Vocabulary import vectorized_emails, vectorized_test_emails
# Model
from perceptron_algorithm.Neuron import Neuron
from perceptron_algorithm.Neural_network import Perceptron

"""
TODO:
- to zmienic całe
"""


def model_training(dataset_of_vectorized_emails):
    n = len(dataset_of_vectorized_emails)  # n = 4459
    emails = dataset_of_vectorized_emails
    list_of_neurons = []
    for i in range(0, n):
        new_neuron = Neuron(input_size=len(emails[i]), id=i)
        list_of_neurons.append(new_neuron)

    neural_network = Perceptron(list_of_neurons=list_of_neurons, learning_rate=0.2, convergence_criteria=0.3)
    neural_network.train(input_data=vectorized_emails, labels=data_classes, list_of_neurons=list_of_neurons)

    prediction = neural_network.prediction(feature_vector=emails, list_of_neurons=list_of_neurons)

    return prediction


trained_model_predictions = model_training(vectorized_emails)
print(trained_model_predictions)

"""
akuratne:
spam : 749
ham : 4825

"""
