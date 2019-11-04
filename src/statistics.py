from pyspark.sql.functions import mean


def get_fraud_proportion(data, verbose_mode=True):
    data.groupBy("FraudResult").count().show()
    outlier_fraction = data.filter("FraudResult == 1").count() / data.filter("FraudResult == 0").count()
    if verbose_mode:
        print('The data set are composed by {0:.3f}% of fraud data.'.format(outlier_fraction*100))
    return outlier_fraction


def get_tpr(tp, fn):
    '''
    :param tp: True Positive
    :param fn: False Negative
    :return: sensitivity, recall, hit rate, or true positive rate (TPR)
                TPR = TP / (TP + FN)
    '''
    return tp / (tp+fn)


def get_ppv(tp, fp):
    '''
    :param tp: True Positive
    :param fp: False Positive
    :return: precision or positive predictive value (PPV)
                PPF = TP / (TP + FP)
    '''
    return tp / (tp+fp)


def get_fpr(fp, tn):
    '''
    :param fp: False Positive
    :param tn: True Negative
    :return: fall-out or false positive rate (FPR)
                FPR = FP / (FP + TN)
    '''
    return fp / (fp+tn)


def get_f1score(tp, fp, fn):
    '''
    :param tp: True Positive
    :param fp: False Positive
    :param fn: False Negative
    :return: F1-score: is the harmonic mean of precision and sensitivity
                F1 = 2 * (PPV * TPR) / (PPV + TPR)
    '''
    ppv = get_ppv(tp, fp)
    tpr = get_tpr(fp, fn)
    f1score = 2 * (ppv * tpr) / (ppv + tpr)
    return f1score


def compute_score(confusion_matrix, metric='f1score'):
    '''
    :param confusion_matrix: Confusion matrix in the follow format:
    TP | FP
    -------
    FN | TN
    This matrix will be used to compute a specified metric above.
    :param metric: One of available metrics:
        - sensitivity, recall, hit rate, or true positive rate (TPR)
        - precision or positive predictive value (PPV)
        - fall-out or false positive rate (FPR)
        - F1 score
    :return:
        The follow metric choose for matrix confusion passed by param.
    '''
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    score = 0.0
    metric = metric.lower()
    if metric == 'sensitivity' or metric == 'recall' or metric == 'tpr':
        score = get_tpr(tp, fn)
    elif metric == 'precision' or metric == 'ppv':
        score = get_ppv(tp, fp)
    elif metric == 'fpr':
        score = get_fpr(fp, tn)
    elif metric == 'f1score' or metric == 'f1' or metric == 'f1-score':
        score = get_f1score(tp, fp, fn)
    return score


def print_description(data, feature_cols):
    print('Fraudulent data summary:')
    print(data.select(feature_cols).filter(data['FraudResult'] == 1).toPandas().describe())

    print('Genuine data summary:')
    print(data.select(feature_cols).filter(data['FraudResult'] == 0).toPandas().describe())

    mean_value = data.select(mean(data['value'])).collect()[0][0]
    print('Transactions mean value: US$ {0:.2f}'.format(mean_value))

    percentFraudByAvgValueLow = data.filter(
        "FraudResult == 1 and PositiveAmount < " +
        str(mean_value)).count() / data.filter('FraudResult == 1').count()
    percentGenByAvgValueLow = data.filter(
        "FraudResult == 0 and PositiveAmount < " +
        str(mean_value)).count() / data.filter('FraudResult == 0').count()

    percentFraudByAvgValueHigh = data.filter(
        "FraudResult == 1 and PositiveAmount > " +
        str(mean_value)).count() / data.filter('FraudResult == 1').count()
    percentGenByAvgValueHigh = data.filter(
        "FraudResult == 0 and PositiveAmount > " +
        str(mean_value)).count() / data.filter('FraudResult == 0').count()

    mean_val_fraud = data.filter("FraudResult == 1").select(mean(data['value'])).collect()[0][0]
    mean_val_not_fraud = data.filter("FraudResult == 0").select(mean(data['value'])).collect()[0][0]

    percentFraudByOperation = data.filter("FraudResult == 1 and Operation == 1 ").count() / data.filter(
        'FraudResult == 1').count()
    percentGenByOperation = data.filter("FraudResult == 0 and Operation == 1 ").count() / data.filter(
        'FraudResult == 0').count()

    print('Mean value for rigged transactions: US$ {0:.2f}.'.format(mean_val_fraud))
    print('Mean value for genuine transactions: US$ {0:.2f}.'.format(mean_val_not_fraud))

    print('Rigged data below the mean: {0:.2f}%.'.format(percentFraudByAvgValueLow * 100))
    print('Rigged data above the mean: {0:.2f}%.\n'.format(percentFraudByAvgValueHigh * 100))

    print('Genuine data below the mean: {0:.2f}%.'.format(percentGenByAvgValueLow * 100))
    print('Genuine data above the mean: {0:.2f}%.'.format(percentGenByAvgValueHigh * 100))

    print('Debit fraudulent transaction: {0:.1f}%.'.format(percentFraudByOperation * 100))
    print('Credit fraudulent transaction: {0:.1f}%.\n'.format(100 - (percentFraudByOperation * 100)))

    print('Debit genuine transaction: {0:.1f}%.'.format(percentGenByOperation * 100))
    print('Credit genuine transaction: {0:.1f}%.'.format(100 - (percentGenByOperation * 100)))

    print('Cross tab between Channel Id and Product Id')
    print(data.filter('FraudResult == 1').stat.crosstab("ChannelId", "ProductId").show())

    print('Cross tab for fraudulent transactions')
    print(data.filter('FraudResult == 1').stat.crosstab("ChannelId", "ProductId").show())

    print('Cross tab for genuine transactions')
    print(data.filter('FraudResult == 0').stat.crosstab("ChannelId","ProductId").show())


