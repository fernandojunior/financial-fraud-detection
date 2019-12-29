# Project Specification : Fraud Detection

## Transaction Classification

Automatically identify if a financial transaction was fraudulent or not.

### Runing the System

* Immediately before running the program, the user is written to select a running type of the project: run, train, validate, and test.

* Once the running type is selected, the client needs to insert by scripting key and respective values about files used by the system.

### Input Files

* By insert running type, if the client's choice is run or train, can start the whole process of machine learn how to classify automatically if a transaction is genuine or fraud and generate an output file with the results. In this process, we've used the Xente dataset<sup>1</sup>, an e-commerce and financial service app serving 10,000+ customers in Uganda. This dataset includes a sample of approximately 140,000 transactions that occurred between 15 November 2018 and 15 March 2019. This dataset is provided following the organization described in the table below.

<sup>1</sup>: Available at: https://zindi.africa/competitions/xente-fraud-detection-challenge.

#### Input data: Xente dataset
| Name                 | Description                                                                                                       | Type         |
|----------------------|-------------------------------------------------------------------------------------------------------------------|--------------|
| TransactionId        | Unique transaction identifier on platform.                                                                         | Categorical  |
| BatchId              | Unique number identifying the customer on platform.                                                                | Categorical  |
| AccountId            | Unique number identifying the customer on platform.                                                                | Categorical  |
| SubscriptionId       | Unique number identifying the customer subscription.                                                               | Categorical  |
| CustomerId           | Unique identifier attached to Account.                                                                             | Categorical  |
| CurrencyCode         | Country currency.                                                                                                  | Categorical  |
| CountryCode          | Numerical geographical code of country.                                                                            | Categorical  |
| ProviderId           | Source provider of Item bought.                                                                                    | Categorical  |
| ProductId            | Item name being bought.                                                                                            | Categorical  |
| ProductCategory      | ProductIds are organized into these broader product categories.                                                    | Categorical  |
| ChannelId            | Identifies if customer used web,Android, IOS, pay later or checkout.                                               | Categorical  |
| Amount               | Value of the transaction. Positive for debits from customer account and negative for credit into customer account. | Float        |
| Value                | Absolute value of the amount.                                                                                      | Float        |
| TransactionStartTime | Transaction start time.                                                                                            | Object       |
| PricingStrategy      | Category of Xente's pricing structure for merchants.                                                               | Categorical    |
| FraudResult          | Fraud status of transaction 1-yes or 0-No.                                                                        | Class target |

New features are created from the Xente dataset, these features are described in the following:
#### Features created in "feature_engineering.py"
| Name                       | Description                                                                           | Type      |
|----------------------------|---------------------------------------------------------------------------------------|-----------|
| Operation                  | Transaction type 1 for debit and -1 for credit.                                        | Numerical |
| ValueStrategy              | Class identifying how multiple times the transaction value is bigger than the average. | Numerical |
| TransactionHour            | Hour time that the transaction happened.                                               | Numerical |
| TransactionDayOfWeek       | Day of week that the transaction happened.                                             | Numerical |
| TransactionDayOfYear       | Day of year that the transaction happened.                                             | Numerical |
| TransactionWeekOfYear      | Week of year that the transaction happened.                                            | Numerical |
| RatioValuespentByWeek      | Ratio between the transaction value and the week of year.                              | Numerical |
| RatioValueSpentByDayOfWeek | Ratio between the transaction value and the day of week.                               | Numerical |
| RatioValueSpentByDayOfYear | Ratio between the transaction value and the day of year.                               | Numerical |
| AverageValuePerProductId   | Average of transaction value for each product Id.                                      | Numerical |
| AverageValuePerProviderId  | Average of transaction value for each provider Id.                                     | Numerical |

#### Features created in "outliers_detector.py"
| Name            | Description                                                                                                             | Type        |
|-----------------|-------------------------------------------------------------------------------------------------------------------------|-------------|
| IsolationForest | Indicates if the instance is classified by IsolationForest algorithm as an outlier, 1 for an outlier, and 0 for normal. | Categorical |
| KNN             | Indicates if the instance is classified by KNN algorithm as an outlier, 1 for an outlier, and 0 for normal.             | Categorical |
| LSCP            | Indicates if the instance is classified by LSCP algorithm as an outlier, 1 for an outlier, and 0 for normal.            | Categorical |
| SumOfOutliers   | Sum all predictions made by outliers detection algorithms, corresponds to instance outlier intensity.                   | Categorical |

* Selecting validate, the client can evaluate the model using part of the training dataset. These features presented in this table are used to predict if the instance is fraud or genuine, in both phases: validation, and test. 

#### Features used to make prediction using "cat_boost.py"
| Name            | Description                                                                                                             | Type        |
|-----------------|-------------------------------------------------------------------------------------------------------------------------|-------------|
| TransactionId        | Unique transaction identifier on platform.                                                                         | Categorical  |
| BatchId              | Unique number identifying the customer on platform.                                                                | Categorical  |
| ProviderId           | Source provider of Item bought.                                                                                    | Categorical  |
| ProductId            | Item name being bought.                                                                                            | Categorical  |
| ProductCategory      | ProductIds are organized into these broader product categories.                                                    | Categorical  |
| ChannelId            | Identifies if customer used web,Android, IOS, pay later or checkout.                                               | Categorical  |
| Value                | Absolute value of the amount.                                                                                      | Float        |
| PricingStrategy      | Category of Xente's pricing structure for merchants.                                                               | Categorical    |
| Operation                  | Transaction type 1 for debit and -1 for credit.                                        | Numerical |
| ValueStrategy              | Class identifying how multiple times the transaction value is bigger than the average. | Numerical |
| TransactionHour            | Hour time that the transaction happened.                                               | Numerical |
| TransactionDayOfWeek       | Day of week that the transaction happened.                                             | Numerical |
| TransactionDayOfYear       | Day of year that the transaction happened.                                             | Numerical |
| TransactionWeekOfYear      | Week of year that the transaction happened.                                            | Numerical |
| RatioValuespentByWeek      | Ratio between the transaction value and the week of year.                              | Numerical |
| RatioValueSpentByDayOfWeek | Ratio between the transaction value and the day of week.                               | Numerical |
| RatioValueSpentByDayOfYear | Ratio between the transaction value and the day of year.                               | Numerical |
| AverageValuePerProductId   | Average of transaction value for each product Id.                                      | Numerical |
| AverageValuePerProviderId  | Average of transaction value for each provider Id.                                     | Numerical |
| IsolationForest | Indicates if the instance is classified by IsolationForest algorithm as an outlier, 1 for an outlier, and 0 for normal. | Categorical |
| KNN             | Indicates if the instance is classified by KNN algorithm as an outlier, 1 for an outlier, and 0 for normal.             | Categorical |
| LSCP            | Indicates if the instance is classified by LSCP algorithm as an outlier, 1 for an outlier, and 0 for normal.            | Categorical |
| SumOfOutliers   | Sum all predictions made by outliers detection algorithms, corresponds to instance outlier intensity.                   | Categorical |

* If the choice if a test, the client can start to test the model predicting the transaction type using a text input file.


## Setting Run Configuration

### File `config.py`

* It is possible to insert how many kernels are intended to use in a couple of functionalities.

* Set if will be used GPU or CPU to training the classification model.

## Code Review

### Use of Variables
* Code uses variables to avoid magic numbers
* Each variable name reflects the purpose of the value stored in it
* Once initiated, the purpose of each variable is maintained throughout the program
* No variables override `Python` built-in values (for example, `def`)

### Use of Functions
* Functions are used as tools to automate tasks which are likely to be repeated
* Functions produce the appropriate output (typically with a return statement) from the appropriate input (function parameters)
* No functions are longer than 18 lines of code (does not include blank lines, comments, or function definitions)

## Documentation
### README
- A `README` file is included detailing all steps required to successfully run the application.

### Comments
- Comments are present and effectively explain longer code procedures.

### Code Quality
- Code is formatted with consistent, logical, and easy-to-read formatting as described in the [PEP 8](https://www.python.org/dev/peps/pep-0008/).

## Suggestions to Make Your Project Stand Out!
- Create new features based on correlation features matrix.
- Incorporate news outlier detectors.
- Data persists when the app is closed and reopened, either through localStorage or an external database (e.g. Firebase).
- Include additional third-party data sources beyond the minimum required.
- Implement additional optimizations that improve the performance and user experience (keyboard shortcuts, autocomplete functionality, filtering of multiple fields, etc).
- Integrate all application components into a cohesive and enjoyable user experience.
