from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree
import matplotlib.pyplot as plt

class BaseScenario:

    def __init__(self, n1=2000, n2=5000, p=20, seed=1234):
        self.n1 = n1
        self.n2 = n2
        self.p = p
        self.seed = seed

    def generate_data(self, n, p, seed):
        return NotImplementedError

    def draw_points(self, features, predictions):
        plt.plot(features[predictions == 0,0], features[predictions == 0, 1], 'bo')
        plt.plot(features[predictions == 1,0], features[predictions == 1, 1], 'ro')

    def draw_area(self):
        return NotImplementedError

    def run_model(self, model, name):
        # generate training and test data
        training_features, training_labels = self.generate_data(self.n1, self.p, self.seed)
        test_features, test_labels = self.generate_data(self.n2, self.p, self.seed+1000)
        # fit model and make predictions
        model.fit(training_features, training_labels)
        predictions = model.predict(test_features).flatten()
        # show error rate
        error_rate = numpy.mean(numpy.abs(test_labels-predictions))
        print(error_rate)
        # draw picture
        self.draw_points(test_features, predictions)
        self.draw_area()
        plt.savefig(name + ".png")



