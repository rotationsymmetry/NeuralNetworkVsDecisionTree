from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree
import matplotlib.pyplot as plt

class AndOrScenario(BaseScenario):
    def draw_area(self):
        pass

    def generate_data(self, n, p, seed):
        def f(a):
            c = 0.2887
            if ((a[0] > c and a[1] > c) or (a[2] > c and a[3] > c) or (a[4] > c and a[5] > c) or
                    (a[6] > c and a[7] > c) or (a[8] > c and a[9] > c) or (a[10] > c and a[11] > c)):
                return numpy.random.binomial(1, 0.8)
            else:
                return numpy.random.binomial(1, 0.2)

        numpy.random.seed(seed)
        features = numpy.random.rand(n, p)
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = AndOrScenario(200, 5000, 40, 1236)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=20, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

#scenario.run_model(nn, "and_or_nn")

scenario.run_model(tree.DecisionTreeClassifier(max_leaf_nodes=10), "and_or_tree_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")