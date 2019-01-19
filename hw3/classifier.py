import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile
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
        t = tree.DecisionTreeClassifier(criterion='entropy')
        return sklearn_classifier(t, data, labels)

class perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels):
        p = Perceptron(tol=1e-4)
        return sklearn_classifier(p, data, labels)

class contest_classifier(abstract_classifier):
    def __init__(self, data, labels, **kwargs):
        self.args = kwargs
        self.ada = AdaBoostClassifier(n_estimators=50)
        self.tree = tree.DecisionTreeClassifier(max_depth=8)
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.perceptron = Perceptron(tol=1e-3)

        self.sp_knn = SelectPercentile(percentile=24)
        self.sp_tree = SelectPercentile(percentile=kwargs['tree_per'])
        self.sp_ada = SelectPercentile(percentile=85)
        self.sp_percep = SelectPercentile(percentile=35)

        data_knn = self.sp_knn.fit_transform(data, labels)
        data_tree = self.sp_tree.fit_transform(data, labels)
        data_ada = self.sp_ada.fit_transform(data, labels)
        data_percep = self.sp_percep.fit_transform(data, labels)

        validation_size = 100
        train_data = data_tree[:validation_size]
        train_labels = labels[:validation_size]
        validation_data = data_tree[validation_size:]
        validation_labels = labels[validation_size:]
        self.knn.fit(data_knn, labels)
        self.tree.fit(train_data, train_labels)
        self.ada.fit(data_ada, labels)
        self.perceptron.fit(data_percep, labels)
        self.prune(self.tree, 0, validation_data, validation_labels)

    def prune(self, tree, index, validation_data, validation_labels):
        inner_tree = tree.tree_
        left_child = inner_tree.children_left[index]
        right_child = inner_tree.children_right[index]
        if left_child != -1:
            self.prune(tree, left_child, validation_data, validation_labels)
        if right_child != -1:
            self.prune(tree, right_child, validation_data, validation_labels)
        predictions_no_prune = tree.predict(validation_data)
        errors_no_prune = (predictions_no_prune ^ validation_labels).sum()

        inner_tree.children_left[index] = -1
        inner_tree.children_right[index] = -1
        predicitions_prune = tree.predict(validation_data)
        errors_prune = (predicitions_prune ^ validation_labels).sum()

        if errors_prune > errors_no_prune:
            inner_tree.children_left[index] = left_child
            inner_tree.children_right[index] = right_child

    def classify(self, features):
        features_mat = features.reshape((1, -1))
        features_knn = self.sp_knn.transform(features_mat)
        features_tree = self.sp_tree.transform(features_mat)
        features_ada = self.sp_ada.transform(features_mat)
        features_percep = self.sp_percep.transform(features_mat)

        w1 = self.args.get('w1', 1)
        w2 = self.args.get('w2', 1)
        w3 = self.args.get('w3', 1)

        p1 = int(self.knn.predict(features_knn)[0])
        p2 = int(self.ada.predict(features_ada)[0])
        p3 = int(self.perceptron.predict(features_percep)[0])
        p3 = int(self.tree.predict(features_tree)[0])

        avg = (w1*p1 + w2*p2 + w3*p3)/(w1 + w2 + w3)
        return bool(avg)

class contest_factory(abstract_classifier_factory):
    def __init__(self, **kwargs):
        self.train_args = kwargs

    def train(self, data, labels):
        return contest_classifier(data, labels, **self.train_args)
