from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree
import matplotlib.pyplot as plt

class ParityScenario(BaseScenario):
    def draw_area(self):
        pass

    def generate_data(self, n, p, seed):
        def f(a):
            if sum(a) % 2 == 0:
                return numpy.random.binomial(1, 0.9)
            else:
                return numpy.random.binomial(1, 0.1)

        numpy.random.seed(seed)
        features = numpy.random.randint(low=0, high=2, size=n*p).reshape(n,p)
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = ParityScenario(200, 5000, 8, 1236)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=4, weight_decay=0.001),
        Layer("Sigmoid", units=4, weight_decay=0.001),
        Layer("Sigmoid", units=4, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

scenario.run_model(nn, "and_or_nn")

scenario.run_model(tree.DecisionTreeClassifier(), "and_or_tree_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")