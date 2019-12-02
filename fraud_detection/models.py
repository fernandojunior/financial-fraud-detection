import config as cfg
from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.knn import KNN


def train_isolation_forest(x_train=cfg.x_train):
    model = IsolationForest(behaviour='new', random_state=cfg.RANDOM_NUMBER,
                            contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS)
    model.fit(cfg.x_train_numerical, cfg.y_train)
    predictions = model.predict(cfg.x_train_numerical)
    x_train[cfg.IF_COLUMN_NAME] = predictions

    x_train = x_train.replace({cfg.IF_COLUMN_NAME: 1}, 0)
    x_train = x_train.replace({cfg.IF_COLUMN_NAME: -1}, 1)
    return x_train


def train_LSCP(x_train):
    detector_list = [LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS),
                     LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS)]
    model = LSCP(detector_list=detector_list, random_state=cfg.RANDOM_NUMBER,
                 contamination=cfg.contamination_level)
    model.fit(cfg.x_train_numerical)
    predictions = model.predict(cfg.x_train_numerical)
    x_train[cfg.LSCP_COLUMN_NAME] = predictions
    return x_train


def train_KNN(x_train):
    knn_clf = KNN(n_jobs=cfg.N_JOBS, contamination=cfg.contamination_level,
                  n_neighbors=cfg.NUM_NEIGHBORS, method='mean')
    knn_clf.fit(cfg.x_train_numerical, cfg.y_train)
    predictions = knn_clf.predict(cfg.x_train_numerical)
    x_train[cfg.KNN_COLUMN_NAME] = predictions
    return x_train
