import classifier
import cross_validation

def experiment_k_values(k_values=[1, 3, 5, 7, 13], num_folds=2):
    filename = "experiments6.csv"
    output_file = open(filename, 'w')
    for k in k_values:
        print("Testing classifier with k={}".format(k))
        knn = classifier.knn_factory(k)
        accuracy, error = cross_validation.evaluate(knn, num_folds)
        output_file.write("{},{:.3f},{:.3f}\n".format(k, accuracy, error))
