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

def evaluate(classifier_factory, k):
    fold_samples = []
    fold_labels = []
    fold_ids = list(range(k))
    for i in range(k):
        samples, labels = load_k_fold_data(i + 1)
        fold_samples.append(samples)
        fold_labels.append(labels)
    # Convert to numpy arrays for easier indexing
    fold_samples = np.array(fold_samples)
    fold_labels = np.array(fold_labels)
    fold_size = len(fold_labels[0])

    error_rates = []
    for test_fold in range(k):
        train_folds = fold_ids[:]
        train_folds.pop(test_fold)
        train_samples = np.concatenate(fold_samples[train_folds])
        train_labels = np.concatenate(fold_labels[train_folds])
        test_samples = fold_samples[test_fold]
        test_labels = fold_labels[test_fold]

        classifier = classifier_factory.train(train_samples, train_labels)
        output_labels = np.array(fold_size, dtype='bool')
        for sample_id in range(fold_size):
            sample = test_samples[sample_id]
            output_labels[sample_id] = classifier.classify(sample)
        mean_error = np.abs(output_labels - test_labels).mean()
        error_rates.append(mean_error)
