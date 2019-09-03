import numpy as np
import pickle


class Neural_network:

    neurons = []  # The neurons which hold the data of the data that has been fed forward.
    neurons_post_activation = []  # Takes the neurons and passes it through an activation function
    network_layer_params = []  # Holds the activation of each layer
    weights = []  # Holds the weights for the whole network
    weights_params = []  # Holds the size of the weights for each layer
    biases = []  # Holds the biases for the whole network
    input_size = 0  # holds the input size of the data
    first_layer = True  # A bool value storing whether the first layer has been created

    def __init__(self, input_size_value):
        self.input_size = input_size_value

    def add(self, size, activation):  # Adds a layer to the network TODO add more types of layers

        self.neurons.append(np.zeros(size))
        self.neurons_post_activation.append(np.zeros(size))
        self.network_layer_params.append(activation)

        if not self.first_layer:

            self.weights.append(np.random.rand(size, self.weights_params[-1]) * 2 - 1)
            self.weights_params.append(size)

        else:

            self.weights.append(np.random.rand(size, self.input_size) * 2 - 1)
            self.weights_params.append(size)
            self.first_layer = False

        self.biases.append(np.random.rand(size)*2-1)

    def fit(self, train_x, train_y, learning_rate, batch_size, epochs, check_interval,
            validate_after_check_interval=False, test_x=None, test_y=None,
            save_weights_and_biases_after_check_interval=False):
        # Create batches
        train_x = np.split(train_x, (len(train_x)/batch_size))
        train_y = np.split(train_y, (len(train_y)/batch_size))
        correct = 0
        # Feed forward
        for count_epoch in range(epochs):

            print("Epoch:", count_epoch, "of", epochs)

            for count_batch, value_batch in enumerate(train_x):

                weights_adj = []
                for i in range(len(self.weights)):
                    weights_adj.append(self.weights[i]*0)
                biases_adj = []
                for i in range(len(self.biases)):
                    biases_adj.append(self.biases[i]*0)

                for count_example, value_example in enumerate(value_batch):

                    for count_layer, value_layer in enumerate(self.neurons):

                        if count_layer == 0:
                            self.feed_forward(value_example, count_layer)
                        else:
                            self.feed_forward(self.neurons_post_activation[count_layer - 1], count_layer)

                    # Back propagation
                    temp_weights_adj, temp_biases_adj, example_correct = \
                        self.back_propagation(value_example, train_y[count_batch][count_example])

                    for i in range(len(weights_adj)):
                        weights_adj[i] += temp_weights_adj[i]
                        biases_adj[i] += temp_biases_adj[i]

                    correct += example_correct

                for i in range(len(weights_adj)):
                    self.weights[i] -= weights_adj[i]/batch_size * learning_rate
                    self.biases[i] -= biases_adj[i]/batch_size * learning_rate

                if count_batch*batch_size % check_interval == 0 and count_batch*batch_size != 0:

                    if validate_after_check_interval:
                        self.validate(test_x, test_y)
                    else:
                        print("Percentage correct:", (correct/check_interval)*100)
                        correct = 0

                    if save_weights_and_biases_after_check_interval:
                        self.save_weights_and_biases()

    def back_propagation(self, example_x, example_y):  # Classic SGD, TODO look into adding more optimizers

        dell_cost_dell_layer = []
        for i in range(len(self.neurons)):
            dell_cost_dell_layer.append(self.neurons[i] * 0)
        weights_adj = []
        for i in range(len(self.weights)):
            weights_adj.append(self.weights[i] * 0)
        biases_adj = []
        for i in range(len(self.biases)):
            biases_adj.append(self.biases[i] * 0)

        correct = 0
        if np.argmax(self.neurons_post_activation[-1]) == np.argmax(example_y):
            correct = 1

        for count_layer in range(1, len(self.neurons) + 1):

            if count_layer == 1:

                if self.network_layer_params[-count_layer] == 'sigmoid':
                    dell_cost_dell_layer[-count_layer] = \
                        (self.neurons_post_activation[-count_layer] - example_y) * \
                        self.sigmoid(self.neurons_post_activation[-count_layer], True)

                elif self.network_layer_params[-count_layer] == 'relu':
                    dell_cost_dell_layer[-count_layer] = \
                        (self.neurons[-count_layer] - example_y) * \
                        self.relu(self.neurons[-count_layer], True)

                elif self.network_layer_params[-count_layer] == 'leaky_relu':
                    dell_cost_dell_layer[-count_layer] = \
                        (self.neurons[-count_layer] - example_y) * \
                        self.relu(self.neurons[-count_layer], True)

            else:
                if self.network_layer_params[-count_layer] == 'sigmoid':
                    dell_cost_dell_layer[-count_layer] = np.dot(dell_cost_dell_layer[-count_layer+1],
                                                                self.weights[-count_layer+1]) *\
                                                         self.sigmoid(self.neurons_post_activation
                                                                      [-count_layer], True)

                elif self.network_layer_params[-count_layer] == 'relu':
                    dell_cost_dell_layer[-count_layer] = np.dot(dell_cost_dell_layer[-count_layer+1],
                                                                self.weights[-count_layer+1]) *\
                                                         self.relu(self.neurons_post_activation[-count_layer],
                                                                   True)

                elif self.network_layer_params[-count_layer] == 'leaky_relu':
                    dell_cost_dell_layer[-count_layer] = np.dot(dell_cost_dell_layer[-count_layer+1],
                                                                self.weights[-count_layer+1]) *\
                                                         self.leaky_relu(self.neurons_post_activation
                                                                         [-count_layer], True)

            if count_layer == len(self.neurons):

                for i in range(len(weights_adj[-count_layer])):
                    weights_adj[-count_layer][i] = dell_cost_dell_layer[-count_layer][i] * example_x

                biases_adj[-count_layer] = dell_cost_dell_layer[-count_layer]

            else:

                for i in range(len(weights_adj[-count_layer])):
                    weights_adj[-count_layer][i] = dell_cost_dell_layer[-count_layer][i] * \
                                                   self.neurons_post_activation[-count_layer - 1]

                biases_adj[-count_layer] = dell_cost_dell_layer[-count_layer]

        return weights_adj, biases_adj, correct

    def feed_forward(self, layer_minus_one_value, index):

        self.neurons[index] = np.dot(self.weights[index], layer_minus_one_value) + self.biases[index]

        if self.network_layer_params[index] == 'sigmoid':
            self.neurons_post_activation[index] = self.sigmoid(self.neurons[index])
        elif self.network_layer_params[index] == 'relu':
            self.neurons_post_activation[index] = self.relu(self.neurons[index])
        elif self.network_layer_params[index] == 'leaky_relu':
            self.neurons_post_activation[index] = self.leaky_relu(self.neurons[index])

    def save_weights_and_biases(self, path=""):
        pickle.dump(self.weights, open(path + "weights.pickle", "wb"))
        pickle.dump(self.biases, open(path + "biases.pickle", "wb"))

    def load_weights_and_biases(self, path=""):
        self.weights = pickle.load(open(path + "weights.pickle", "rb"))
        self.biases = pickle.load(open(path + "biases.pickle", "rb"))

    def validate(self, test_x, test_y):
        correct = 0
        for count_test, value_test in enumerate(test_x):
            for i in range(len(self.neurons)):
                if i == 0:
                    self.feed_forward(value_test, i)
                else:
                    self.feed_forward(self.neurons_post_activation[i - 1], i)
            if np.argmax(self.neurons_post_activation[-1]) == np.argmax(test_y[count_test]):
                correct += 1
        print("Percentage correct:", (correct/len(test_y)*100))

    def sigmoid(self, x, deriv=False):
        if not deriv:
            return 1 / (1 + np.exp(-x))
        else:
            # We pass pre sigmoided value so no need for: x = self.sigmoid(x)
            return x * (1 - x)

    def relu(self, x, deriv=False):
        if not deriv:
            return np.maximum(x,0)
        else:
            x[x<=0] = 0
            x[x>0] = 1
            return x

    def leaky_relu(self, x, deriv=False):
        if not deriv:
            return np.maximum(x,0.05*x)
        else:
            x[x <= 0] = 0.05
            x[x > 0] = 1
            return x




if __name__ == '__main__':
    # Here is an example of how you would use the class
    
    def open_and_unpickle_data():
        file_train_x, file_train_y = open("train_x.pickle", "rb"), open("train_y.pickle", "rb")
        file_test_x, file_test_y = open("test_x.pickle", "rb"), open("test_y.pickle", "rb")

        train_x, train_y = pickle.load(file_train_x), pickle.load(file_train_y)
        test_x, test_y = pickle.load(file_test_x), pickle.load(file_test_y)

        return train_x, train_y, test_x, test_y


    def create_train_y_test_y(val_train_y, val_test_y):
        out_train_y = np.zeros((60000, 10))
        for i in range(len(val_train_y)):
            out_train_y[i][int(val_train_y[i])] = 1

        out_test_y = np.zeros((10000, 10))
        for i in range(len(val_test_y)):
            out_test_y[i][int(val_test_y[i])] = 1

        return out_test_y, out_train_y
    
    train_x, before_train_y, test_x, before_test_y = open_and_unpickle_data()

    # Makes the output data into x,10 size vectors
    # E.g the output 4 becomes [0,0,0,0,1,0,0,0,0,0]
    test_y, train_y = create_train_y_test_y(before_train_y, before_test_y)

    model = Neural_network(784)
    model.add(100, 'sigmoid')
    model.add(50, 'sigmoid')
    model.add(10, 'sigmoid')
    model.fit(train_x, train_y, 0.1, 40, 15000, 2500, True, test_x, test_y, True)
    #model.load_weights_and_biases()
    model.validate(test_x, test_y)

