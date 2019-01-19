import sys

import hw3_utils
import cross_validation

folds = 2
if len(sys.argv) > 1:
    folds = int(sys.argv[1])

dataset = hw3_utils.load_data()
print("Splitting dataset to {} folds".format(folds))
cross_validation.split_crosscheck_groups((dataset[0], dataset[1]), folds)
