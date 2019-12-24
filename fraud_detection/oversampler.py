import utils as ut
from imblearn.over_sampling import SMOTENC


def smotenc_over_sampler(x_data_set,
                         y_data_set,
                         categorical_features_dims,
                         num_jobs=8,
                         random_seed=42):
    """Generate oversampling for training data set using SMOTENC technique.

    Source:
    https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTENC.html

    Args:
        x_data_set (pandas data frame):
        y_data_set (pandas vector):
        categorical_features_dims (list):
        num_jobs (int):
        random_seed (int):

    Returns:
        X and Y datasets balanced (using SMOTENC oversampling technique).
    """
    ut.save_log('{0} :: {1}'.format(smotenc_over_sampler.__module__,
                                    smotenc_over_sampler.__name__))

    model = SMOTENC(categorical_features=categorical_features_dims,
                    random_state=random_seed,
                    n_jobs=num_jobs)

    x_data_set_balanced, y_data_set_balanced = \
        model.fit_resample(x_data_set, y_data_set)

    return x_data_set_balanced, y_data_set_balanced
