[![Build Status](https://travis-ci.com/italoPontes/fraud_detection.svg?token=xCXQ5y8dztyVs3aHPJLA&branch=master)](https://travis-ci.com/italoPontes/fraud_detection)

# Fraud Detection
> The objective of this machine learning model is detect fraudulent transactions in the financial services sector.


## Stakeholders
> People involved in this project

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Scrum Master           | Fernando Felix | fernandofelix@copin.ufcg.edu.br   |
| Data Scientist       | Author                 | Dunfrey P. Aragão | dunfrey@gmail.com   |
| Data Scientist       | Author                 | Ítalo de Pontes Oliveira           | italodepontesoliveira93@gmail.com |


## Usage
Clone the repository:
```
git clone https://github.com/<@github_username>/fraud_detection.git
cd fraud_detection
```

### Prerequisites

We have tested the library in Ubuntu 19.04, 18.04 and 16.04, but it should be easy to compile in other platforms.

#### Python

The project was written in Python 3, and work with later as well.
Also, please read up the subsequent libraries that are used: [CatBoost](https://catboost.ai/), [Scikit Learn](https://scikit-learn.org/stable/), [PyOD](https://pyod.readthedocs.io/en/latest/), [Pandas](https://pandas.pydata.org/) and [PySpark](https://spark.apache.org/).

#### Libraries

Please make sure that it has installed all the required dependencies. 
A list of items to be installed using pip install can be running as following:
```
pip install -r requirements.txt
```

### Running

To run the whole workflow of the project, it is possible by the following command:

```
python main.py run \
  --input_train_file ../data/xente_fraud_detection_train.csv \
  --input_test_file ../data/xente_fraud_detection_test.csv \
  --output_balanced_train_x_file ../data/balanced_train_x.csv \
  --output_balanced_train_y_file ../data/balanced_train_y.csv \
  --output_valid_x_file ../data/valid_x.csv \
  --output_valid_y_file ../data/valid_y.csv \
  --output_valid_result_file ../data/valid_result.csv \
  --output_test_result_file ../data/xente_output_final.txt
```

* **input_train_file** & **input_test_file**: train and test datasets available by [Zindi](https://zindi.africa/competitions/xente-fraud-detection-challenge/data).
* **output_balanced_train_x_file** & **output_balanced_train_y_file**: 70% of the oversampled training dataset to work on the training step.
* **output_valid_x_file** & **output_valid_y_file**: 30% of the training dataset saved on memory to be used in validation step.
* **output_valid_result_file**: outcome from the classification model using validation data.
* **output_test_result_file**: outcome from the classification model using test data.


If that interested in work just a specific step (train, validate, or test), it is possible by:
```
python <mode> <--key> <keywords>
```

* **mode**: run, train, validate, and test
* **--key**: input_train_file, input_test_file, output_balanced_train_x_file, output_balanced_train_y_file, output_valid_x_file, output_valid_y_file, output_valid_result_file, output_test_result_file
* **keywords**: file name in your choice

For example:

```
python main.py test \
 --input_test_file ../data/xente_fraud_detection_test.csv \
 --output_test_result_file ../data/xente_output_final.txt
```

### Model Frequency

The entire process is running over 11'44". To training and predict the outcomes by detector outliers, it happens over 2'42," and to train dataset oversampling happens over by 4'10".

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Project Folder Structure and Files

* [data](./data/): contains data files generated and used by the classification model.
* [docs](./docs/): contains documentation of the project
* [fraud_detection](./fraud_detection/): main Python package with source of the model.
* [jupyter-notebook](./jupyter-notebook/): contains jupyter notebooks evaluation and modeling experimentation.
