from perceptron_algorithm.Neuron import Neuron
from feature_extraction.Vocabulary import num_of_initial_neurons, vectorized_emails, label_list
import random
import matplotlib.pyplot as plt

# TODO:
#  - trenowanie i aktualizacjia wag/biasów w oparciu o szukanie najmniejszego błędu (wyprowadzenie na kartce)
#  - zrobic liste aktualizowanyh neuronow z dostepem do ich wag i biasow
#  - oprzec uczenie nowe wagi o iloczyn z aktualnym errorem, jezeli error = 0 -> zostawic nie mnozyc
#  - sprawdź dlaczego błąd jaki jest w uczeniu jest zerowy, tj. czemu niby sieć jest nieomylna
class NeuralNetwork:
    def __init__(self, predictions: list[int], vectorized_mails_input: list[list[int]], layer_output=None,
                 num_of_neurons=num_of_initial_neurons):
        # podajemy cały set
        self.input = vectorized_mails_input
        self.layer_output = layer_output
        self.num_of_neurons = num_of_neurons
        self.acceptable_error = 0.1
        self.input_layer = {}
        self.hidden_layers = {}
        self.predictions = predictions
        self.network_loss = {}
        self.init_neurons = []
        self.epoch = 500

    # output = 1 prediction = 0 suma_wazona = 2.130215165708253, waga = 0,23812004459646455, teta = 0.5
    # blad = -0.25362346512925716287024027846558

    # ouput = 1 , pred = 0 , suma_wazona = 2.130215165708253, waga = 0,23812004459646455, teta = 0.5
    # blad = -0.25362346512925716287024027846558

    # bierzemy wartosc absolutna by ustandaryzowac kwestie bledu
    # pred = 0 output = 0 blad = 0.0

    def calculate_loss(self, prediction: int, layer_output: int,
                       neuron_weight: float, neuron_weighted_sum: float):
        teta = 0.5
        return teta * abs(prediction - layer_output) * neuron_weight * neuron_weighted_sum

    def save_layer_settings(self, filepath: str, list_of_neurons: list[Neuron]):
        with open(filepath, "w", encoding="utf8") as file:
            for neuron in list_of_neurons:
                file.write(f"Neuron ID: {neuron.id}\n")
                file.write(f"Neuron weight: {neuron.weight}\n")
                file.write(f"Neuron bias: {neuron.bias}\n")

    def create_input_layer(self) -> dict:  # Zwraca { id neuronu : wartość neuronu}
        # Tworzenie warstwy neuronów
        layer_loss = {}
        list_of_neurons = []

        # najlepsze : a = 0.001, b = 0.05
        range_a = 0.001
        range_b = 0.05
        # tworzenie warstwy (89 neuronów)
        for i in range(self.num_of_neurons):
            n = Neuron(id=i, weight=random.uniform(a=range_a, b=range_b))
            list_of_neurons.append(n)
        # Każdy neuron genetuje x wyników, zatem do klucza (id neuronu) dolaczamy liste wyników dla każdego maila
        self.init_neurons = list_of_neurons

        for neuron in list_of_neurons:
            list_of_outputs = []
            loss = []
            for i, input_data in enumerate(self.input):
                neuron.input = input_data
                neuron.bias = ((neuron.weight + sum(filter(lambda x: x != -1, input_data))) /
                               len(list(filter(lambda x: x != -1, input_data))))
                # zaokrąglenie do 1 liczby po przecinku w celu ułatwienia debugowania
                neuron_output = 1 if round(neuron.weighted_sum(), 1) > 0.51 else 0

                list_of_outputs.append(neuron_output)
                single_loss = self.calculate_loss(prediction=self.predictions[i], layer_output=neuron_output,
                                                  neuron_weight=neuron.weight,
                                                  neuron_weighted_sum=neuron.weighted_sum())
                loss.append(single_loss)  # Użyj indeksu i jako klucza dla layer_loss
            self.input_layer[neuron] = list_of_outputs
            layer_loss[neuron.id] = loss

        # print("loss: ", layer_loss.get(0))
        self.network_loss = layer_loss

        plt.plot(list(layer_loss.keys()),list(layer_loss.values()), marker='.', linestyle='none', color='blue')
        plt.ylabel("Błąd sieci")
        plt.xlabel("Nr neuronu")
        plt.title("Wykres błędu przypadający na neuron")
        plt.show()
        return self.input_layer

    def check_results(self, layer):
        counter = 0
        # layer values : list[list[int]]
        for label, neuron_output in zip(label_list, layer.values()):
            # neuron generuje 89 wyjść dla jednego maila, zatem sumujemy je i dzielimy przez dlugosc layer.values() by otrzymac usrednienie wyniku
            avg_neuron_output = 1 if sum(neuron_output) / len(layer.values()) > 0.51 else 0
            if label == avg_neuron_output:
                counter += 1
        return counter / len(layer.values())

    # todo - napraw bo wszystko co poniżej jest niestabilne!

    def neuron_segregation(self) -> list[Neuron]:
        num_of_neurons = 0
        list_of_neurons_hid = []
        # odsiewanie
        for neuron, neuron_output in self.input_layer.items():
            if set(neuron_output) == set(self.predictions):
                self.hidden_layers[neuron] = neuron_output
                list_of_neurons_hid.append(neuron)
                num_of_neurons += 1
        return list_of_neurons_hid

    def training_loop(self):
        # Assuming a learning rate of 0.01 for demonstration
        learning_rate = 0.75
        curr_epoch = 0
        current_error = 1 - self.check_results(self.hidden_layers)

        # Training loop
        while self.acceptable_error < current_error and curr_epoch != self.epoch:
            #print("EPOCH : ", curr_epoch, " / ", self.epoch)
            #start_time = time.time()

            list_of_neurons = self.neuron_segregation()
            neuron_loss = []
            list_of_outputs = []
            layer_loss = {}

            # Iterate through each neuron in the list
            i = 0
            for neuron in list_of_neurons:
                # Calculate the gradient of the loss function (error) with respect to the weight
                # For simplicity, assuming the derivative is the error itself
                error = self.network_loss.get(neuron.id)[-1] if neuron.id in self.network_loss else 0
                # Update the weight using the learning rate and the error
                neuron.weight += learning_rate * error

                # Update the bias similarly
                neuron.bias -= learning_rate * error

                # todo - napraw wprowadzanie inputu
                # Recalculate the weighted sum and output for the neuron
                neuron_output = 1 if neuron.weighted_sum() > 0.51 else 0

                list_of_outputs.append(neuron_output)
                single_loss = self.calculate_loss(prediction=self.predictions[i], layer_output=neuron_output,
                                                  neuron_weight=neuron.weight,
                                                  neuron_weighted_sum=neuron.weighted_sum())
                neuron_loss.append(single_loss)  # Use index i as key for layer_loss
                layer_loss[neuron.id] = neuron_loss
                self.hidden_layers[neuron] = list_of_outputs
                i+=1


            #end_time = time.time()
            current_error = 1 - self.check_results(self.hidden_layers)
            if curr_epoch % 100 == 0:
                #print("Time taken for epoch: ", end_time - start_time)
                print("Current error: ", round(current_error, 2) * 100)

            curr_epoch += 1



    def create_output_layer(self):
        pass


network = NeuralNetwork(predictions=label_list, vectorized_mails_input=vectorized_emails)
input_layer = network.create_input_layer()
network.neuron_segregation()
result = round(network.check_results(input_layer) * 100, 2)
network.save_layer_settings("layer_settings", network.init_neurons)
print("Average accuracy 1st layer: ", result, "%")
print("Average error 1st layer: ", 100 - result, "%")

"""
network.training_loop()

result = round(network.check_results(network.hidden_layers) * 100, 2)
network.save_layer_settings("layer_settings", network.init_neurons)
print("Average accuracy: ", result, "%")
print("Average error: ", 100 - result, "%")

"""
