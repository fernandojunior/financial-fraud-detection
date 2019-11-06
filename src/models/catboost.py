cv_dataset = Pool(data=fraud_data_smotenc_x,
                  label=fraud_data_smotenc_y,
                  cat_features=cat_features)
iterations = 100
max_depth = 4
num_folds = 5

params = {"iterations": iterations,
          "depth": max_depth,
          "loss_function": "Logloss",
          "verbose": False,
          "roc_file": "roc-file"}

scores = cv(cv_dataset,
            params,
            fold_count=num_folds,
            plot="True")

model = CatBoostClassifier(max_depth=max_depth, verbose=False, max_ctr_complexity=1, iterations=iterations).fit(cv_dataset)