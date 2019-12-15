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
> Describe how to reproduce the model

Clone the project from the analytics Models repo.
```
git clone https://github.com/<@github_username>/fraud_detection.git
cd fraud_detection
```

### Prerequisites

We have tested the library in Ubuntu 19.04, 18.04 and 16.04, but it should be easy to compile in other platforms.

#### Python

The project was wrote in Python 3, and work with later as well.

#### Libraries

We also use the libraries: [CatBoost](https://catboost.ai/), [Scikit Learn](https://scikit-learn.org/stable/), [PyOD](https://pyod.readthedocs.io/en/latest/), [Pandas](https://pandas.pydata.org/) and [PySpark](https://spark.apache.org/).


## Final Report (to be filled once the project is done)

### Model Frequency

> Describe the interval frequency and estimated total time to run

### Model updating

> Describe how your model may be updated in the future

### Maintenance

> Describe how your model may be maintained in the future

### Minimum viable product

> Describe a minimum configuration that would be able to create a minimum viable product.

### Early adopters

> Describe any potential paying users for this product if it was available today. Also state a point of contact for each of them.

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [docs](./docs): contains documentation of the project
* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [fraud_detection](./fraud_detection/): main Python package with source of the model.
