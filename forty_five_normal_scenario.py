from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy, math
from sklearn import tree
import matplotlib.pyplot as plt

class FortyFiveNormalScenario(BaseScenario):
    def draw_area(self):
        plt.plot([0,1], [0,1], 'k')

    def generate_data(self, n, p, seed):
        def f(a):
            if a[0] > a[1]:
                return numpy.random.binomial(1, 0.8)
            else:
                return numpy.random.binomial(1, 0.2)

        def sample_value(index):
            x0 = 99
            x1 = 99
            while not (0 <= x0 <= 1 and 0 <= x1 <= 1):
                s0 = numpy.random.uniform(low=0, high=math.sqrt(2))
                s1 = numpy.random.normal(loc=0, scale=0.5)
                ff = math.pi/4
                x0 = math.cos(ff)*s0 + math.sin(-ff)*s1
                x1 = math.sin(ff)*s0 + math.cos(ff)*s1
            return [x0, x1]

        numpy.random.seed(seed)
        noises = numpy.random.rand(n, p-2)
        info = numpy.array([sample_value(i) for i in range(n)])
        features = numpy.c_[info, noises]
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = FortyFiveNormalScenario(500, 5000, 20, 144)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=5, weight_decay=0.01),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

scenario.run_model(nn, "45n_nn")

scenario.run_model(tree.DecisionTreeClassifier(max_leaf_nodes=10), "45n_tree_gini")

#scenario.run_model(tree.DecisionTreeClassifier(criterion="entropy"), "45_tree_entropy")