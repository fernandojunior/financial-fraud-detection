import config as cfg
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.knn import KNN


def make_grid_search_cat_boost():
    model = CatBoostClassifier(eval_metric=cfg.EVAL_METRIC,
                               random_seed=cfg.RANDOM_NUMBER)

    grid = {'learning_rate': cfg.LEARNING_RATE_LIST,
            'depth': cfg.DEPTH_LIST,
            'l2_leaf_reg': cfg.LEAF_REG}

    grid_search_result = model.grid_search(grid,
                                           X=cfg.x_train_balanced[cfg.NUMERICAL_FEATURES],
                                           y=cfg.y_train_balanced,
                                           plot=True)
    return grid_search_result


def train_cat_boost():
    cfg.model_cat_boost = CatBoostClassifier(depth=5,
                                             learning_rate=0.1,
                                             l2_leaf_reg=1,
                                             eval_metric=cfg.EVAL_METRIC,
                                             random_seed=cfg.RANDOM_NUMBER)

    cfg.model_cat_boost.fit(cfg.x_train_balanced,
                            cfg.y_train_balanced,
                            verbose=False,
                            plot=True,
                            cat_features=cfg.CATEGORICAL_FEATURES)


def train_isolation_forest():
    cfg.model_isolation_forest = IsolationForest(behaviour='new',
                                                 random_state=cfg.RANDOM_NUMBER,
                                                 contamination=cfg.contamination_level,
                                                 n_jobs=cfg.N_JOBS)

    cfg.model_isolation_forest.fit(cfg.x_train_numerical, cfg.y_train)
    predictions = cfg.model_isolation_forest.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.IF_COLUMN_NAME] = predictions

    cfg.x_train = cfg.x_train.replace({cfg.IF_COLUMN_NAME: 1}, 0)
    cfg.x_train = cfg.x_train.replace({cfg.IF_COLUMN_NAME: -1}, 1)


def train_LSCP():
    detector_list = [LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS),
                     LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS)]
    cfg.model_lscp = LSCP(detector_list=detector_list,
                          random_state=cfg.RANDOM_NUMBER,
                          contamination=cfg.contamination_level)
    cfg.model_lscp.fit(cfg.x_train_numerical)
    predictions = cfg.model_lscp.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.LSCP_COLUMN_NAME] = predictions


def train_KNN():
    cfg.model_knn = KNN(n_jobs=cfg.N_JOBS,
                        contamination=cfg.contamination_level,
                        n_neighbors=cfg.NUM_NEIGHBORS,
                        method='mean')
    cfg.model_knn.fit(cfg.x_train_numerical, cfg.y_train)
    predictions = cfg.model_knn.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.KNN_COLUMN_NAME] = predictions
