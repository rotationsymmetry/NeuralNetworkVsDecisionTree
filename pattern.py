from sknn.mlp import Classifier, Layer
import numpy
from sklearn import tree


class BaseScenario:
    def __init__(self, n1, n2, p):
        self.n1 = n1
        self.n2 = n2
        self.p = p

    def generate_data(self, n, seed):
        return NotImplementedError

    def run(self, model, seed):
        # generate training and test data
        training_features, training_labels = self.generate_data(self.n1, seed)
        test_features, test_labels = self.generate_data(self.n2, seed+1000)
        # fit model and make predictions
        model.fit(training_features, training_labels)
        predictions = model.predict(test_features).flatten()
        # show error rate
        error_rate = numpy.mean(numpy.abs(test_labels-predictions))
        return error_rate

    def __str__(self):
        return NotImplementedError


class Comparator:
    def __init__(self, scenario, model1, model2, model3, nsim):
        self.scenario = scenario
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.nsim = nsim

    def compare(self):
        errors = [self.single_compare(k) for k in range(self.nsim)]

        def avg_error(error):
            return sum(error)/len(error)

        avg_error1 = avg_error([e[0] for e in errors])
        avg_error2 = avg_error([e[1] for e in errors])
        avg_error3 = avg_error([e[2] for e in errors])

        better_prob = sum([e[0] <= e[1] and e[0] <= e[2] for e in errors]) / len(errors)

        print("{0}: avg_error1={1:.1f}  avg_error2={2:.1f} avg_error3={3:.1f} => {4:.1f}"
              .format(self.scenario.__str__(), avg_error1*100, avg_error2*100, avg_error3*100, better_prob*100))
        return avg_error1, avg_error2

    def single_compare(self, seed):

        error1 = self.scenario.run(self.model1, seed)
        error2 = self.scenario.run(self.model2, seed)
        error3 = self.scenario.run(self.model3, seed)
        print("{0}: error1={1:.1f}  error2={2:.1f} error3={3:.1f}".format(self.scenario.__str__(), error1*100, error2*100, error3*100))
        return error1, error2, error3


class PatternScenario(BaseScenario):
    def __init__(self, n1, n2, p, w0, w1):
        super().__init__(n1, n2, p)
        self.w0 = w0
        self.w1 = w1

    def __str__(self):
        return "n1={0} p={1} w0={2} w1={3}".format(self.n1, self.p, self.w0, self.w1)

    def generate_data(self, n, seed):
        def f(width):
            r = [0 for i in range(self.p)]
            pos = numpy.random.randint(width, self.p-width, size=1)
            r[(pos-width):(pos+width+1)] = [1 for i in range(width*2+1)]
            return r

        def g(t):
            if t == 1:
                return f(self.w1)
            else:
                return f(self.w0)

        def flip(v):
            if numpy.random.binomial(1, 0.9) == 1:
                return v
            else:
                return 1 - v

        numpy.random.seed(seed)
        truth = numpy.random.randint(low=0, high=2, size=n)
        features = numpy.array([g(i) for i in truth])
        labels = numpy.array([flip(v) for v in truth])
        return features, labels


#scenario = PatternScenario(800, 5000, 500, 388472) #25, 20
#scenario = PatternScenario(400, 5000, 500, 388472) #15, 10



