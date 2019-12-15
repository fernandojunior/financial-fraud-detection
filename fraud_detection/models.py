import pickle

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTENC
from catboost import CatBoostClassifier

import config as cfg
import handler as hdl


def train_isolation_forest():
    hdl.outside_log(train_isolation_forest.__module__,
                    train_isolation_forest.__name__)
    set_model_if()
    cfg.model_if = get_model_if()
    cfg.model_if.fit(cfg.x_train_numerical, cfg.y_train)
    with open('../data/model_if', 'wb') as file_model:
        pickle.dump(cfg.model_if, file_model)


def set_model_if():
    cfg.model_if = IsolationForest(behaviour='new',
                                   random_state=cfg.RANDOM_NUMBER,
                                   contamination=cfg.percent_contamination,
                                   n_jobs=cfg.N_JOBS)


def get_model_if():
    return cfg.model_if


def predict_isolation_forest():
    hdl.outside_log(predict_isolation_forest.__module__,
                    predict_isolation_forest.__name__)
    if not cfg.model_if:
        with open('../data/model_if', 'rb') as pickle_file:
            cfg.model_if = pickle.load(pickle_file)
    predictions = cfg.model_if.predict(cfg.x_data_temp[cfg.NUMERICAL_FEATURES])
    cfg.x_data_temp[cfg.IF_COLUMN_NAME] = predictions
    cfg.x_data_temp = cfg.x_data_temp.replace({cfg.IF_COLUMN_NAME: 1}, 0)
    cfg.x_data_temp = cfg.x_data_temp.replace({cfg.IF_COLUMN_NAME: -1}, 1)


def train_lscp():
    hdl.outside_log(train_lscp.__module__,
                    train_lscp.__name__)
    set_model_lscp()
    cfg.model_lscp = get_model_lscp()
    cfg.model_lscp.fit(cfg.x_train_numerical, cfg.y_train)
    with open('../data/model_lscp', 'wb') as file_model:
        pickle.dump(cfg.model_lscp, file_model)


def set_model_lscp():
    cfg.model_lscp = LSCP(detector_list=[set_model_bagging(),
                                         set_model_lof(),
                                         set_model_cblof()],
                          contamination=cfg.percent_contamination,
                          random_state=cfg.RANDOM_NUMBER)


def get_model_lscp():
    return cfg.model_lscp


def predict_lscp():
    hdl.outside_log(predict_lscp.__module__,
                    predict_lscp.__name__)
    if not cfg.model_lscp:
        with open('../data/model_lscp', 'rb') as pickle_file:
            cfg.model_lscp = pickle.load(pickle_file)
    predictions = cfg.model_lscp.predict(
        cfg.x_data_temp[cfg.NUMERICAL_FEATURES])
    cfg.x_data_temp[cfg.LSCP_COLUMN_NAME] = predictions


def train_knn():
    hdl.outside_log(train_knn.__module__,
                    train_knn.__name__)
    set_model_knn()
    cfg.model_knn = get_model_knn()
    cfg.model_knn.fit(cfg.x_train_numerical, cfg.y_train)
    with open('../data/model_knn', 'wb') as file_model:
        pickle.dump(cfg.model_knn, file_model)


def set_model_knn():
    cfg.model_knn = KNN(contamination=cfg.percent_contamination,
                        n_neighbors=cfg.NUM_NEIGHBORS,
                        method='mean',
                        n_jobs=cfg.N_JOBS)


def get_model_knn():
    return cfg.model_knn


def predict_knn():
    hdl.outside_log(predict_knn.__module__,
                    predict_knn.__name__)
    if not cfg.model_knn:
        with open('../data/model_knn', 'rb') as pickle_file:
            cfg.model_knn = pickle.load(pickle_file)
    predictions = cfg.model_knn.predict(
        cfg.x_data_temp[cfg.NUMERICAL_FEATURES])
    cfg.x_data_temp[cfg.KNN_COLUMN_NAME] = predictions


def smotenc_over_sampler():
    hdl.outside_log(smotenc_over_sampler.__module__,
                    smotenc_over_sampler.__name__)
    set_model_smotenc()
    clf_smotenc = get_model_smotenc()
    x_balanced, y_balanced = clf_smotenc.fit_resample(cfg.x_train, cfg.y_train)
    return x_balanced, y_balanced


def make_grid_search_cat_boost():
    hdl.outside_log(make_grid_search_cat_boost.__module__,
                    make_grid_search_cat_boost.__name__)

    model_cat_boost = []
    if not cfg.model_cat_boost:
        model_cat_boost = cfg.model_cat_boost

    grid = {'learning_rate': cfg.LEARNING_RATE_LIST,
            'depth': cfg.DEPTH_LIST,
            'l2_leaf_reg': cfg.LEAF_REG_LIST}

    grid_search_result = \
        model_cat_boost.grid_search(grid,
                                    X=cfg.x_train_balanced[
                                        hdl.get_categorical_features()],
                                    y=cfg.y_train_balanced,
                                    plot=True)
    return grid_search_result


def train_cat_boost():
    hdl.outside_log(train_cat_boost.__module__,
                    train_cat_boost.__name__)
    cfg.model_cat_boost = set_model_cat_boost()
    cfg.model_cat_boost.fit(cfg.x_train_balanced,
                            cfg.y_train_balanced,
                            verbose=False,
                            plot=True,
                            cat_features=hdl.get_categorical_features())
    cfg.model_cat_boost.save_model(fname=cfg.model_catboost_file)


def predict_cat_boost(mode):
    hdl.outside_log(predict_cat_boost.__module__,
                    predict_cat_boost.__name__)
    cfg.model_cat_boost = set_model_cat_boost()
    cfg.model_cat_boost.load_model(fname=cfg.model_catboost_file)
    cfg.predictions = cfg.model_cat_boost.predict(
        cfg.x_to_predict_catboost[cfg.ALL_FEATURES])
    cfg.x_to_predict_catboost['CatBoost'] = cfg.predictions
    if mode == 'VALID':
        cfg.x_to_predict_catboost['FraudResult'] = cfg.y_valid


def set_model_cat_boost():
    clf_cat_boost = CatBoostClassifier(
        depth=cfg.DEPTH_CATBOOST,
        learning_rate=cfg.LEARNING_RATE_CATBOOST,
        l2_leaf_reg=cfg.L2_CATBOOST,
        eval_metric=cfg.EVAL_METRIC,
        task_type=cfg.TYPE_DEVICE_CATBOOST,
        random_seed=cfg.RANDOM_NUMBER)
    return clf_cat_boost


def set_model_bagging():
    clf_feat_bag = FeatureBagging(contamination=cfg.percent_contamination,
                                  combination='max',
                                  n_estimators=cfg.NUM_ESTIMATORS,
                                  random_state=cfg.RANDOM_NUMBER,
                                  n_jobs=cfg.N_JOBS)
    return clf_feat_bag


def set_model_lof():
    clf_lof = LOF(contamination=cfg.percent_contamination,
                  n_neighbors=cfg.NUM_NEIGHBORS,
                  n_jobs=cfg.N_JOBS)
    return clf_lof


def set_model_cblof():
    clf_cblof = CBLOF(contamination=cfg.percent_contamination,
                      n_clusters=cfg.NUM_CLUSTERS,
                      random_state=cfg.RANDOM_NUMBER,
                      n_jobs=cfg.N_JOBS)
    return clf_cblof


def set_model_smotenc():
    cfg.model_smotenc = SMOTENC(
        categorical_features=cfg.categorical_features_dims,
        random_state=cfg.RANDOM_NUMBER,
        n_jobs=cfg.N_JOBS)


def get_model_smotenc():
    return cfg.model_smotenc
