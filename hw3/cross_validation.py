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
