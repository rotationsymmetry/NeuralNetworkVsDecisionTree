from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy, math
from sklearn import tree
import matplotlib.pyplot as plt


class FortyFiveScenario(BaseScenario):
    def draw_area(self):
        pass

    def generate_data(self, n, p, seed):
        def line(x):
            q = 3
            a = (q / 0.5) - 1
            return math.pow(x, a)

        def f(a):
            if (a[0] < line(a[1]) or a[2] < line(a[3]) or a[4] < line(a[5])):
                return numpy.random.binomial(1, 0.8)
            else:
                return numpy.random.binomial(1, 0.2)

        numpy.random.seed(seed)
        features = numpy.random.rand(n, p)
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = FortyFiveScenario(500, 5000, 20, 1236)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=20, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

scenario.run_model(nn, "45_nn")

scenario.run_model(tree.DecisionTreeClassifier(max_leaf_nodes=20), "45_tree_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")