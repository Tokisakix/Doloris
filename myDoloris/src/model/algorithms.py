from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def get_model(name, params=None):
    params = params or {}

    if name == "logistic_regression":
        return LogisticRegression(**params)

    elif name == "random_forest":
        return RandomForestClassifier(**params)

    elif name == "knn":
        return KNeighborsClassifier(**params)

    elif name == "svm":
        return SVC(**params)

    elif name == "decision_tree":
        return DecisionTreeClassifier(**params)

    elif name == "sgd":
        return SGDClassifier(**params)

    elif name == "mlp":
        return MLPClassifier(**params)
