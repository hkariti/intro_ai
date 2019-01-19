import classifier
import cross_validation

def experiment6(k_values=[1, 3, 5, 7, 13], num_folds=2):
    filename = "experiments6.csv"
    output_file = open(filename, 'w')
    for k in k_values:
        print("Testing classifier with k={}".format(k))
        knn = classifier.knn_factory(k)
        accuracy, error = cross_validation.evaluate(knn, num_folds)
        output_file.write("{},{:.3f},{:.3f}\n".format(k, accuracy, error))

def experiment_tree(num_folds=2):
    print("Testing tree")
    tree = classifier.tree_factory()
    accuracy, error = cross_validation.evaluate(tree, num_folds)
    return accuracy, error

def experiment_perceptron(num_folds=2):
    print("Testing perceptron")
    perceptron = classifier.perceptron_factory()
    accuracy, error = cross_validation.evaluate(perceptron, num_folds)
    return accuracy, error

def experiment12(num_folds=2):
    tree_results = experiment_tree(num_folds)
    perceptron_results = experiment_perceptron(num_folds)
    output_file = open('experiments12.csv', 'w')
    output_file.write("1,{:.3f},{:.3f}\n".format(*tree_results))
    output_file.write("2,{:.3f},{:.3f}\n".format(*perceptron_results))

if __name__ == '__main__':
    experiment6()
    experiment12()
