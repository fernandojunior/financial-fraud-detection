from pyspark.sql.functions import mean

def get_fraud_proportion(data):
    data.groupBy("FraudResult").count().show()
    outlier_fraction = data.filter("FraudResult == 1").count() / data.filter("FraudResult == 0").count()
    print('The data set are composed by {0:.3f}% of fraud data.'.format(outlier_fraction*100))
    return outlier_fraction


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

    print('Rigged data below the mean: {0:.2f}%.'.format(percentFraudByAvgValueLow * 100))
    print('Rigged data above the mean: {0:.2f}%.\n'.format(percentFraudByAvgValueHigh * 100))

    print('Genuine data below the mean: {0:.2f}%.'.format(percentGenByAvgValueLow * 100))
    print('Genuine data above the mean: {0:.2f}%.'.format(percentGenByAvgValueHigh * 100))

