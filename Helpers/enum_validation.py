from enum import Enum
from functools import partial
from Helpers.validations import k_cross_validate, strat_k_cross_validate, train_test_split


class Validation(Enum):
    K_FOLD = partial(k_cross_validate)
    STRAT_K_FOLD = partial(strat_k_cross_validate)
    TRAIN_TEST = partial(train_test_split)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


