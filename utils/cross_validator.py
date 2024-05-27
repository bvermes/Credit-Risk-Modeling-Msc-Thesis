from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state
import numpy as np

class TaxIDStratifiedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        tax_ids = X["tax_id"].unique()
        if self.shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(tax_ids)
        n_folds = self.n_splits
        fold_sizes = np.full(n_folds, tax_ids.size // n_folds, dtype=int)
        fold_sizes[:tax_ids.size % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = X[X["tax_id"].isin(tax_ids[start:stop])].index
            yield test_indices
            current = stop
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits