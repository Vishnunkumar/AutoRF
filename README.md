# AutoRF
Auto ML using random forest trees

## Installation

```
pip install -U AutoRF
```

## Usage

```
from AutoRF.AutoRF import AutoRF
from sklearn import datasets
from sklearn import linear_model, preprocessing, ensemble
import pandas as pd

# Loading data
br_d = pd.read_csv("Datasets/drug_class/drug200.csv")
X = br_d.drop(['Drug'], axis=1)
y = pd.get_dummies(br_d['Drug'])

# Initialize the object
rf_auto = AutoRF(X, y)
prep_x, prep_y = rf_auto.preprocess()
clf, te_x, te_y, prob = rf_auto.model(prep_x, prep_y)
rf_auto.scoring(clf, te_x, te_y, prob)
rf_auto.pipeline()

# Do with a single line using pipeline
rf_auto = AutoRF(X, y)
rf_auto.pipeline()

# Do parameter tuning
rf_auto.parameter_tuning(ensemble.RandomForestClassifier(), 
                         prep_x, prep_y, 
                         "Multi-Classification")

# Customize your parameter tuning also
param_dict = {'max_depth': [3, 8, 15],
              'max_features': [42, 85, 170],
              'n_estimators': [50, 100]}

rf_auto.parameter_tuning(ensemble.RandomForestClassifier(), 
                         prep_x, prep_y, 
                         "Multi-Classification",
                         param_dict=param_dict,
                         n_jobs=20)
                         
```
