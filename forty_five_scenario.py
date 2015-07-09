from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree
import matplotlib.pyplot as plt

class FortyFiveScenario(BaseScenario):
    def draw_area(self):
        plt.plot([0,1], [0,1], 'k')

    def generate_data(self, n, p, seed):
        def f(a):
            if a[0] > a[1]:
                return numpy.random.binomial(1, 0.9)
            else:
                return numpy.random.binomial(1, 0.1)

        numpy.random.seed(seed)
        features = numpy.random.rand(n, p)
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = FortyFiveScenario(2000, 5000, 20, 1236)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=10, weight_decay=0.000),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=250)

#scenario.run_model(nn, "45_nn")

scenario.run_model(tree.DecisionTreeClassifier(max_leaf_nodes=1000), "45_tree_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")