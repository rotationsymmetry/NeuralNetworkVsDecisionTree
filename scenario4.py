from pattern import PatternScenario, Comparator
from sknn.mlp import Classifier, Layer
from sklearn import tree


p = 1000

s = PatternScenario(400, 10000, p, 20, 25)

nn_clf = Classifier(
    layers=[
        Layer("Sigmoid", units=p, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500)

gini_clf = tree.DecisionTreeClassifier()
entropy_clf = tree.DecisionTreeClassifier(criterion="entropy")

comparator = Comparator(s, nn_clf, gini_clf, entropy_clf, 100)
comparator.compare()
