from pattern import PatternScenario, Comparator
from sknn.mlp import Classifier, Layer
from sklearn import tree


s = PatternScenario(400, 5000, 200, 10, 15, 234)

nn_clf = Classifier(
    layers=[
        Layer("Sigmoid", units=500, weight_decay=0.001),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=2)

gini_clf = tree.DecisionTreeClassifier()
entropy_clf = tree.DecisionTreeClassifier(criterion="entropy")

comparator = Comparator(s, nn_clf, gini_clf, entropy_clf, 2)
comparator.compare()