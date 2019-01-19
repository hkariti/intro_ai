import pickle
import numpy as np

def split_crosscheck_groups(dataset, num_folds):
    samples, labels = dataset
    labels = np.array(labels)
    dataset_size = samples.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    fold_size = int(dataset_size/num_folds)
    for i in range(num_folds):
        start_fold = i * fold_size
        end_fold = (i + 1) * fold_size
        fold_indices = indices[start_fold:end_fold]
        fold = (samples[fold_indices], labels[fold_indices])

        filename = "ecg_fold_{}.data".format(i+1)
        data_file = open(filename, 'wb')
        pickle.dump(fold, data_file)
        data_file.close()

def load_k_fold_data(fold_id):
    filename = "ecg_fold_{}.data".format(fold_id)
    data_file = open(filename, 'rb')
    fold = pickle.load(data_file)
    return fold

def pop_test_fold(samples, labels, test_fold):
    """
    Return the given samples without the set used for testing
    """
    num_folds = samples.shape[0]
    train_folds = list(range(num_folds))
    train_folds.pop(test_fold)
    train_samples = np.concatenate(samples[train_folds])
    train_labels = np.concatenate(labels[train_folds])

    return train_samples, train_labels

def test_classifier(classifier, test_samples, test_labels):
    fold_size = len(test_labels)
    output_labels = np.ndarray(fold_size, dtype='bool')
    for sample_id in range(fold_size):
        sample = test_samples[sample_id]
        output_labels[sample_id] = classifier.classify(sample)
    mean_error = np.abs(output_labels ^ test_labels).mean()

    return mean_error

def evaluate(classifier_factory, k):
    fold_samples = []
    fold_labels = []
    for i in range(k):
        samples, labels = load_k_fold_data(i + 1)
        fold_samples.append(samples)
        fold_labels.append(labels)
    # Convert to numpy arrays for easier indexing
    fold_samples = np.array(fold_samples)
    fold_labels = np.array(fold_labels)

    error_rates = []
    accuracy_rates = []
    for test_fold in range(k):
        train_samples, train_labels = pop_test_fold(fold_samples, fold_labels, test_fold)
        test_samples = fold_samples[test_fold]
        test_labels = fold_labels[test_fold]

        classifier = classifier_factory.train(train_samples, train_labels)
        mean_error = test_classifier(classifier, test_samples, test_labels)
        error_rates.append(mean_error)
        accuracy_rates.append(1 - mean_error)

    accuracy = np.mean(accuracy_rates)
    error = np.mean(error_rates)
    return accuracy, error
