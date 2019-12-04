import config as cfg
import handler as hdl

def train_isolation_forest():
    hdl.outside_log(train_isolation_forest.__module__,
                    train_isolation_forest.__name__)
    from sklearn.ensemble import IsolationForest

    cfg.model_isolation_forest = IsolationForest(behaviour='new',
                                                 random_state=cfg.RANDOM_NUMBER,
                                                 contamination=cfg.contamination_level,
                                                 n_jobs=cfg.N_JOBS)
    cfg.model_isolation_forest.fit(cfg.x_train_numerical, cfg.y_train)
    
def predict_isolation_forest():
    hdl.outside_log(predict_isolation_forest.__module__,
                    predict_isolation_forest.__name__)
    predictions = cfg.model_isolation_forest.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.IF_COLUMN_NAME] = predictions
    cfg.x_train = cfg.x_train.replace({cfg.IF_COLUMN_NAME: 1}, 0)
    cfg.x_train = cfg.x_train.replace({cfg.IF_COLUMN_NAME: -1}, 1)


def train_LSCP():
    hdl.outside_log(train_LSCP.__module__,
                    train_LSCP.__name__)
    from pyod.models.lscp import LSCP
    from pyod.models.lof import LOF

    detector_list = [LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS),
                     LOF(contamination=cfg.contamination_level, n_jobs=cfg.N_JOBS)]
    cfg.model_lscp = LSCP(detector_list=detector_list,
                          random_state=cfg.RANDOM_NUMBER,
                          contamination=cfg.contamination_level)
    cfg.model_lscp.fit(cfg.x_train_numerical)

def predict_LSCP():
    hdl.outside_log(predict_LSCP.__module__,
                    predict_LSCP.__name__)
    predictions = cfg.model_lscp.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.LSCP_COLUMN_NAME] = predictions


def train_KNN():
    hdl.outside_log(train_KNN.__module__,
                    train_KNN.__name__)
    from pyod.models.knn import KNN

    cfg.model_knn = KNN(n_jobs=cfg.N_JOBS,
                        contamination=cfg.contamination_level,
                        n_neighbors=cfg.NUM_NEIGHBORS,
                        method='mean')
    cfg.model_knn.fit(cfg.x_train_numerical, cfg.y_train)

def predict_KNN():
    hdl.outside_log(predict_KNN.__module__,
                    predict_KNN.__name__)
    predictions = cfg.model_knn.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.KNN_COLUMN_NAME] = predictions

def smotenc_oversampling():
    from imblearn.over_sampling import SMOTENC
    sm = SMOTENC(categorical_features=cfg.categorical_features_dims,
                 random_state=cfg.RANDOM_NUMBER, n_jobs=cfg.N_JOBS)
    x_smotenc, y_smotenc = sm.fit_sample(cfg.x_train[cfg.ALL_FEATURES], cfg.y_train)
    return x_smotenc, y_smotenc

def make_gridSearch_catBoost():
    hdl.outside_log(make_gridSearch_catBoost.__name__)
    from catboost import CatBoostClassifier

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


def train_catboost():
    hdl.outside_log(train_catboost.__name__)
    from catboost import CatBoostClassifier

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