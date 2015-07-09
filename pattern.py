from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree
import matplotlib.pyplot as plt

class PatternScenario(BaseScenario):
    def draw_area(self):
        pass

    def generate_data(self, n, p, seed):

        numpy.random.seed(seed)

        def f(width):
            r = [0 for i in range(p)]
            pos = numpy.random.randint(width, p-width, size=1)
            r[(pos-width):(pos+width+1)] = [1 for i in range(width*2+1)]
            return r

        def g(t):
            if t == 1:
                return f(25)
            else:
                return f(20)

        def flip(v):
            if numpy.random.binomial(1, 0.9) == 1:
                return v
            else:
                return 1 - v

        truth = numpy.random.randint(low=0, high=2, size=n)
        features = numpy.array([g(i) for i in truth])
        labels = numpy.array([flip(v) for v in truth])
        return features, labels


scenario = PatternScenario(800, 5000, 500, 388472)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=500, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

scenario.run_model(nn, "and_or_nn")

scenario.run_model(tree.DecisionTreeClassifier(max_leaf_nodes=10000), "pattern_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")