import random


class Neuron:
    def __init__(self, input, id, weight=random.uniform(0.1, 0.3), output=None, is_activated=None):
        # Pierwsza wagi będą inicjowane losowo, jeżeli jesteśmy jużw innej warstwie niż pierwszej to będą zmieniane
        self.weight = weight
        self.input = input
        self.id = id
        # suma elementów przez długość listy - pierwszy sposób obliczania biasu
        # lambda zwraca true jeżeli x != 0
        f = lambda x: x != -1
        self.bias = sum(filter(f, self.input)) / len(list(filter(f, self.input)))

        self.output = output
        self.is_activated = is_activated

    def activation_func(self, output):
        if output > 1:
            return True
        else:
            return False

    def weighted_sum(self):
        # suma wazona ustawiamy na 0
        weighted_sum = 0.0 + self.weight

        for x in self.input:
            weighted_sum += x * self.weight
        self.output = weighted_sum - self.bias
        # kalkulujemy czy aktywujemy neuron
        self.is_activated = self.activation_func(output=self.output)
        if self.is_activated:
            return self.output
        else:
            return 0



"""#1,1,0,1,1,0,1,1,0
#1,0,0,0,1,0,1,1,0
vect = [
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1],
    [1, 1],
    [1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
]

layer = {}
for i in range(len(vect)):
    n = Neuron(input=vect[i], id=i)
    layer[n.id] = n.weighted_sum()
print(layer)
"""