import numpy as np
from sklearn import tree
from sklearn.linear_model import Perceptron
from hw3_utils import abstract_classifier, abstract_classifier_factory

def euclidean_distance(x1, x2):
    diff_squared = (x1 - x2)**2
    distance_squared = diff_squared.sum(axis=0)
    return np.sqrt(distance_squared)

class knn_classifier(abstract_classifier):
    def __init__(self, k, train_set, train_labels):
        self._k = k
        self._train_set = train_set
        self._train_labels = train_labels

    def classify(self, features):
        num_samples = len(self._train_labels)
        distances = np.ndarray(num_samples)
        for i in range(num_samples):
            distances[i] = euclidean_distance(features, self._train_set[i])
        knn = np.argsort(distances)[:self._k]
        knn_labels = self._train_labels[knn]
        majority = np.round(knn_labels.mean())
        return bool(majority)

class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(self.k, data, labels)

class sklearn_classifier(abstract_classifier):
    def __init__(self, classifier, train_set, train_labels):
        self.classifier = classifier
        self.classifier.fit(train_set, train_labels)

    def classify(self, features):
        feature_mat = features.reshape((1, -1))
        prediction = self.classifier.predict(feature_mat)
        return prediction[0]

class tree_factory(abstract_classifier_factory):
    def train(self, data, labels):
        t = tree.DecisionTreeClassifier()
        return sklearn_classifier(t, data, labels)

class perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels):
        p = Perceptron(tol=1e-4)
        return sklearn_classifier(p, data, labels)

class contest_classifier(abstract_classifier):
    def __init__(self, tree_depth, full_weight, data, labels):
        self.full_weight = full_weight
        self.full_tree = tree.DecisionTreeClassifier(max_depth=tree_depth)
        self.partial_tree = tree.DecisionTreeClassifier(max_depth=tree_depth)

        self.full_tree.fit(data, labels)
        separate = self._separate(data)
        partial_data = data[separate]
        partial_labels = labels[separate]
        self.partial_tree.fit(partial_data, partial_labels)

    @classmethod
    def _separate(self, data):
        separate1 = (data[:, 3] <= 0.374) & \
                    (data[:, 16] > 0.003) & \
                    (data[:, 74] > -0.002)
        separate2 = (data[:, 3] > 0.374) & \
                    (data[:, 98] <= 0.364) & \
                    (data[:, 16] > 0.017)
        return separate1 | separate2

    def classify(self, features):
        features_mat = features.reshape((1, -1))
        p_full = self.full_tree.predict(features_mat)[0]
        separate = self._separate(features_mat)
        if separate[0]:
            p_partial = self.partial_tree.predict(features_mat)[0]
        else:
            p_partial = p_full

        return bool(np.round(self.full_weight * p_full + (1-self.full_weight)*p_partial))

class contest_factory(abstract_classifier_factory):
    def __init__(self, tree_depth, tree_weight):
        self.tree_depth = tree_depth
        self.tree_weight = tree_weight

    def train(self, data, labels):
        return contest_classifier(self.tree_depth, self.tree_weight, data, labels)
