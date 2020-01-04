from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_fscore_support as score

import utils
import config


def train(X_data,
          y_data,
          categorical_features_list,
          depth_tree,
          learning_rate,
          l2_regularization,
          number_of_iterations,
          cat_boost_file_name='../data/catBoost_model'):
    """Fit the CatBoost model using the training data.
        The model weights are saved in output file.

    Args:
        X_data: a matrix dataframe
        y_data: column outcome value to use in the train
        categorical_features_list: categorical features in X_data
        number_of_iterations: number of iterations
        depth_tree: depth tree to limit the catboost size
        learning_rate: learning rate to adjust the weights
        l2_regularization: regularization metric
        cat_boost_file_name: file name to export the trained model

    Returns:
        model: CatBoost model
    """
    utils.save_log('{0} :: {1}'.format(
        train.__module__,
        train.__name__))

    model_cat_boost = create_model(depth_tree=depth_tree,
                                   learning_rate=learning_rate,
                                   reg_l2=l2_regularization,
                                   iterations=number_of_iterations)

    model_cat_boost.fit(X_data,
                        y_data,
                        verbose=False,
                        plot=True,
                        cat_features=categorical_features_list)

    model_cat_boost.save_model(fname=cat_boost_file_name)

    return model_cat_boost


def create_model(iterations=5000,
                 depth_tree=4,
                 learning_rate=0.0135,
                 reg_l2=2,
                 evaluation_metric='F1'):
    """Create a CatBoost model.

    Args:
        iterations: number of iterations
        depth_tree: depth tree to limit the catboost size
        learning_rate: learning rate to adjust the weights
        reg_l2: regularization metric
        evaluation_metric (str):

    Returns:
        model: Isolation Forest model
    """
    utils.save_log('{0} :: {1}'.format(
        create_model.__module__,
        create_model.__name__))

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth_tree,
        learning_rate=learning_rate,
        l2_leaf_reg=reg_l2,
        eval_metric=evaluation_metric,
        task_type=config.device_type,
        random_seed=config.random_seed)

    return model


def predict(data, y_value: None, cat_boost_file_name='../data/catBoost_model'):
    """Generate predictions using the Cat Boost model.

    Args:
        data (Pandas dataframe): a matrix dataframe
        cat_boost_file_name: input file name of Cat Boost model
        y_value:

    Returns:
        predictions: Model outcomes (predictions)
    """
    utils.save_log('{0} :: {1}'.format(
        predict.__module__,
        predict.__name__))

    model_cat_boost = create_model()

    model_cat_boost.load_model(fname=cat_boost_file_name)

    predictions = model_cat_boost.predict(data)

    if y_value is not None:
        y_value = y_value.toPandas()
        export_valid_performance(y_label=y_value,
                                 y_predictions=predictions)

    return predictions


def export_valid_performance(y_label,
                             y_predictions,
                             depth_tree=5,
                             learning_rate=0.1,
                             regularization_l2=2,
                             output_file='../data/catBoost_perfomance.txt'):
    """Generate file Cat Boost model performance in validation step.

    Args:
        y_label : FraudResult
        y_predictions: CatBoost Predictions
        depth_tree: Configure CatBoost
        learning_rate: Configure CatBoost
        regularization_l2: Configure CatBoost
        output_file: output file name to export performance
    """
    utils.save_log('{0} :: {1}'.format(
        export_valid_performance.__module__,
        export_valid_performance.__name__))\

    precision, recall, f_score, _ = score(y_label, y_predictions)

    output_parser = open(output_file, 'w')
    output_parser.write('LABELS\t\tFraudResult\t\t\t\t | \tCatBoost\n')
    output_parser.write('------------------------------------------\n')
    output_parser.write('precision: \t{}\t\t | \t{}\n'.
                        format(precision[0], precision[1]))
    output_parser.write('recall: \t\t{}\t\t | \t{}\n'.
                        format(recall[0], recall[1]))
    output_parser.write('f-score: \t\t{}\t\t | \t{}\n'.
                        format(f_score[0], f_score[1]))
    output_parser.write('------------------------------------------\n')
    output_parser.write('CAT-BOOST CONFIGURATION--------------------\n')
    output_parser.write('depth: {} - LR {} - L2: {}\n'.
                        format(depth_tree,
                               learning_rate,
                               regularization_l2))
    output_parser.close()
