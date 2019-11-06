import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN


def balance_using_smotenc(data, all_features, label, categorical_positions, random_value=42):
    """
    :param data: The pyspark data used to balance.
    :param all_features: The list of attributes of this data set.
    :param label: Column identifying which feature corresponds to class label.
    :param categorical_positions: List with integer values indicating the categorical
    features positions.
    :param random_value: Seed used to generate random numbers.
    :return: X and Y data balanced in pandas data frame format.
    """
    sm = SMOTENC(categorical_features=categorical_positions, random_state=random_value)
    data_pd = data.toPandas()
    x_data, y_data = sm.fit_sample(data_pd[all_features], data_pd[label].ravel())
    x_data = pd.DataFrame(x_data, columns=all_features)
    y_data = pd.DataFrame(y_data, columns=[label])
    return x_data, y_data


def balance_using_only_numerical_features(data, numerical_features, label, strategy='smote', random_value=42):
    """
    The SMOTE, RandomOverSampler, and ADASYN balance the data using
    only numerical features. To use categorical features, please,
    use the SMOTENC variation.
    :param data: The pyspark data used to balance.
    :param numerical_features: The list of numerical attributes of this data set.
    :param label: Column identifying which feature corresponds to class label.
    :param strategy: Could be one of available strategy to do oversampling: the SMOTE, RandomOverSampler, or Adasyn.
    :param random_value: Seed used to generate random numbers.
    :return: X and Y data balanced in pandas data frame format.
    """
    x_smote = data.select(numerical_features).toPandas()
    y_smote = data.select(label).toPandas()

    if strategy == 'smote':
        model = SMOTE(random_state=random_value)
    elif strategy == 'random':
        model = RandomOverSampler(random_state=random_value)
    elif strategy == 'adasyn':
        model = ADASYN(random_state=random_value)
    else:
        return x_smote, y_smote

    x_data, y_data = model.fit_resample(x_smote, y_smote)
    return x_data, y_data
