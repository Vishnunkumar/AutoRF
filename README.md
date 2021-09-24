# AutoRF
Auto ML using random forest trees

## Installation

```
pip install AutoRF
```

## Simple Usage

- Pipeline method: Completely automated 
```
from AutoRF.AutoRF import AutoRF
from sklearn import datasets
import pandas as pd

br_d = datasets.fetch_california_housing()
X = pd.DataFrame(br_d.data)
y = pd.Series(br_d.target)
X.columns = br_d.feature_names
rf_auto = AutoRF(X, y)
rf_auto.pipeline()

--> 'model_saved to random_forest.joblib'
```

- Non-pipeline

```
from AutoRF.AutoRF import AutoRF
from sklearn import datasets
import pandas as pd

br_d = datasets.fetch_california_housing()
X = pd.DataFrame(br_d.data)
y = pd.Series(br_d.target)
X.columns = br_d.feature_names
rf_auto = AutoRF(X, y)

prep_x, prep_y = rf_auto.preprocess("nil") # leaving empty will use normalizer as scaler 
clf, te_x, te_y, prob = rf_auto.model(prep_x, prep_y, 1000) # last variable defines the threshold value after which the model will be treated as regression #default value = 100
met = rf_auto.scoring(clf, te_x, te_y, prob)
message = rf_auto.save_model(clf)
```
