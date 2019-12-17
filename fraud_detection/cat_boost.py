import utils as ut
from catboost import CatBoostClassifier


def train_cat_boost(x_data_set,
                    y_data_set,
                    categorical_features_list,
                    output_cat_boost_file_name='../data/catBoost_model'):
    """You can call this function to train the cat boost using the
    hyper parameters passed.
    """
    ut.save_log('{0} :: {1}'.format(train_cat_boost.__module__,
                                    train_cat_boost.__name__))

    model_cat_boost = get_model_cat_boost()

    model_cat_boost.fit(x_data_set,
                        y_data_set,
                        verbose=False,
                        plot=True,
                        cat_features=categorical_features_list)

    model_cat_boost.save_model(fname=output_cat_boost_file_name)


def get_model_cat_boost(depth_tree=5,
                        learning_rate=0.1,
                        reg_l2=2,
                        evaluation_metric='F1',
                        device_type='GPU',
                        random_seed=42):
    """Define Cat Boost model using the hyper parameters
    defined in config file.

    Args:
        - depth_tree (int):
        - learning_rate (int):
        - reg_l2 (int):
        - evaluation_metric (str):
        - device_type (str):
        - random_seed (int):

    Returns:

    """
    ut.save_log('{0} :: {1}'.format(get_model_cat_boost.__module__,
                                    get_model_cat_boost.__name__))

    model = CatBoostClassifier(
        depth=depth_tree,
        learning_rate=learning_rate,
        l2_leaf_reg=reg_l2,
        eval_metric=evaluation_metric,
        task_type=device_type,
        random_seed=random_seed)

    return model


def predict_cat_boost(x_data_set,
                      cat_boot_file_name='../data/catBoost_model'):
    """Use this model to make predictions using the data set to
    predict outliers using Cat Boost model.


    """
    ut.save_log('{0} :: {1}'.format(predict_cat_boost.__module__,
                                    predict_cat_boost.__name__))

    model_cat_boost = get_model_cat_boost()
    model_cat_boost.load_model(fname=cat_boot_file_name)
    predictions = model_cat_boost.predict(x_data_set)

    return predictions


def make_grid_search_cat_boost(x_data_set,
                               y_data_set,
                               learning_rate_list,
                               depth_tree_list,
                               leaf_reg_list):
    """We've used the cat boost model to train using balanced data set.

    Source: https://catboost.ai/docs/concepts/about.html

    Args:
        x_data_set (pandas data frame):
        y_data_set (pandas data frame):
        learning_rate_list (float list):
        depth_tree_list (int list):
        leaf_reg_list (int list):

    Returns:

    """
    ut.save_log('{0} :: {1}'.format(make_grid_search_cat_boost.__module__,
                                    make_grid_search_cat_boost.__name__))

    model = CatBoostClassifier()

    grid = {'learning_rate': learning_rate_list,
            'depth': depth_tree_list,
            'l2_leaf_reg': leaf_reg_list}

    grid_search_result = \
        model.grid_search(grid,
                          X=x_data_set,
                          y=y_data_set,
                          plot=True)

    return grid_search_result
