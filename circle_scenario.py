from BaseScenario import BaseScenario
from sknn.mlp import Classifier, Layer
import numpy, math
from sklearn import tree
import matplotlib.pyplot as plt

class CircleScenario(BaseScenario):
    def draw_area(self):
        plt.axis('equal')
        n = 100
        x = [math.cos(i/n*2*3.14)*math.sqrt(2/3.14) for i in range(0,n)]
        y = [math.sin(i/n*2*3.14)*math.sqrt(2/3.14) for i in range(0,n)]
        plt.plot(x, y, 'k')

    def generate_data(self, n, p, seed):
        def f(a):
            if numpy.square(a[0]) + numpy.square(a[1]) < 2/3.14:
                return numpy.random.binomial(1, 0.8)
            else:
                return numpy.random.binomial(1, 0.2)

        numpy.random.seed(seed)
        features = numpy.random.rand(n, p)*2-1
        labels = numpy.apply_along_axis(f, 1, features)
        return features, labels


scenario = CircleScenario(2000, 5000, 20, 123)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=50, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=250)

scenario.run_model(nn, "circle_nn")

scenario.run_model(tree.DecisionTreeClassifier(), "circle_tree")