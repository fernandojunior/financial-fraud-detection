import config as cfg
import handler as hdl

def train_isolation_forest():
    hdl.outside_log(train_isolation_forest.__module__,
                    train_isolation_forest.__name__)
    cfg.model_isolation_forest = cfg.if_outlier
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
    cfg.model_lscp = cfg.lscp_outlier
    cfg.model_lscp.fit(cfg.x_train_numerical)

def predict_LSCP():
    hdl.outside_log(predict_LSCP.__module__,
                    predict_LSCP.__name__)
    predictions = cfg.model_lscp.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.LSCP_COLUMN_NAME] = predictions

def train_KNN():
    hdl.outside_log(train_KNN.__module__,
                    train_KNN.__name__)
    cfg.model_knn = cfg.knn_outlier
    cfg.model_knn.fit(cfg.x_train_numerical, cfg.y_train)

def predict_KNN():
    hdl.outside_log(predict_KNN.__module__,
                    predict_KNN.__name__)
    predictions = cfg.model_knn.predict(cfg.x_train_numerical)
    cfg.x_train[cfg.KNN_COLUMN_NAME] = predictions

def smotenc_oversampling():
    hdl.outside_log(smotenc_oversampling.__module__,
                    smotenc_oversampling.__name__)
    sm = cfg.smotenc_oversampler
    x_smotenc, y_smotenc = sm.fit_sample(cfg.x_train[cfg.ALL_FEATURES], 
                                         cfg.y_train)
    return x_smotenc, y_smotenc

def make_gridSearch_catBoost():
    hdl.outside_log(make_gridSearch_catBoost.__module__,
                    make_gridSearch_catBoost.__name__)

    if not cfg.catboost_classifier:
        model = cfg.catboost_classifier

    grid = {'learning_rate': cfg.LEARNING_RATE_LIST,
            'depth': cfg.DEPTH_LIST,
            'l2_leaf_reg': cfg.LEAF_REG_LIST}

    grid_search_result = model.grid_search(grid,
                                           X=cfg.x_train_balanced[cfg.NUMERICAL_FEATURES],
                                           y=cfg.y_train_balanced,
                                           plot=True)
    return grid_search_result

def train_catboost():
    hdl.outside_log(train_catboost.__module__,
                    train_catboost.__name__)
    cfg.model_cat_boost = cfg.catboost_classifier
    cfg.model_cat_boost.fit(cfg.x_train_balanced,
                            cfg.y_train_balanced,
                            verbose=False,
                            plot=True,
                            cat_features=cfg.CATEGORICAL_FEATURES)
    cfg.model_cat_boost.save_model(fname=cfg.model_catboost_saved)