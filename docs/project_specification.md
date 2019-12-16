# Project specification : Fraud Detection

## Transaction Classification

Automatically identify if a financial transaction was fraudulent or not.

### Runing the system

* Immediately before running the program, user is wrote to select a running type of the project: run, train, validate, and test.

* Once a running type is selected, the client needs insert by scripting key and respectives values about files used by the system.

### Files

* By insert running type, if the client's choice is run or train, can start the whole process of machine learn how to classify automatically if a transaction is genuine or fraud and generate an output file with the results.

* Selecting validate, the client can evaluate the model using part of the training dataset.

* If the choice if a test, the client can start to test the model predicting the transaction type using a text input file.

## Code Review

### File config.py

* In this code file is possible to insert how many kernels is inteded to use in a couple of functionalities.

* Set if will be used GPU or CPU to training the classification model.

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
