import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_log_error, precision_score, make_scorer
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from pysimplelog import Logger

logger = Logger()

class AutoRF:

	"""docstring for Auto_ML"""

	def __init__(self, X, y):

		"""
		Initialize the dataset i.e features and target variable
		"""
	
		self.X = X
		self.y = y
        

	def preprocess(self, mode="Normalization"):

		"""
		preprocess the dataset by applying normalization, missing value imputation and encoding text to numeric

		"""

		self.mode = mode

		object_cols = [col for col in self.X.columns if self.X[col].dtype == "object"]

		if object_cols is not None:
			for o_col in object_cols:
				self.X[o_col] = self.X.fillna(self.X[o_col].mode())

		for col in list(self.X.columns):
			if object_cols is not None:
				if col not in object_cols:
					self.X[col] = self.X.fillna(self.X[col].mean())

			else:
				self.X[col] = self.X.fillna(self.X[col].mean())

		self.X = pd.get_dummies(self.X, columns=object_cols, drop_first=True)

		if self.mode == "Normalization":

			normalizer = Normalizer()
			self.X = normalizer.fit_transform(self.X)

		else:

			scaler = self.mode
			self.X = scaler.fit_transform(self.X)

		return self.X, self.y


	def model(self, train, target, estimator='RF', threshold=100):

		"""
		Inputs are the preprocessed data from the preprocess function and output is the validation data
		along with the trained model
		prob: "Regression/Classification"
		"""

		self.train = train
		self.target = target
		self.threshold = threshold
		self.estimator = estimator

		value_sets = list(set(self.target))
		
		if self.estimator == 'RF':
			if len(value_sets) < self.threshold:
				if len(value_sets) == 1:
					prob = "Binary-Classification"
				else:
					prob = "Multi-Classification"

				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, stratify=self.target, shuffle=True)
				rf_c = RandomForestClassifier()
				clf = rf_c.fit(tr_x, tr_y)

				return clf, te_x, te_y, prob


			elif len(value_sets) > self.threshold:
				prob = "Regression"
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
				rf_r = RandomForestRegressor()
				clf = rf_r.fit(tr_x, tr_y)

				return clf, te_x, te_y, prob
		
		else:
			tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
			rf_r = self.estimator
			clf = rf_r.fit(tr_x, tr_y)
			prob = str(self.estimator)
			
			return clf, te_x, te_y, prob
		

	def scoring(self, clf, te_x, te_y, prob):

		"""
		Returns the performance of the model
		"""
		
		self.clf = clf
		self.te_x = te_x
		self.te_y = te_y
		self.prob = prob

		if self.prob == "Regression":

			preds = clf.predict(self.te_x)
			met = mean_absolute_error(self.te_y, preds)
			metr = {}
			metr['MAE'] = met
			
			return metr

		elif self.prob == "Multi-Classification":
			
			preds = clf.predict(self.te_x)
			met = f1_score(self.te_y, preds, average='weighted')
			
			metr = {}
			metr['f1-weighted'] = met

			return metr

		else:
			
			preds = clf.predict(self.te_x)
			met = f1_score(self.te_y, preds)
			
			metr = {}
			metr['f1'] = met

			return met
		

	def save_model(self, clf):

		"""
		Saves the trained model
		"""

		self.clf = clf

		joblib.dump(clf, "random_forest.joblib")

		return logger.info("Model has been saved to random_forest.joblib")
	
	def parameter_tuning(self, clf, train, target, prob, param_dict='', n_jobs=10):
		
		"""
		Only applicable for Random Forest models
		"""
		
		self.param_dict = param_dict
		self.clf = clf
		self.train = train
		self.target = target
		self.prob = prob
		self.n_jobs = n_jobs
		mf = int(self.train.shape[1])
		
		if self.param_dict == '':
			
			self.param_dict = {"max_depth": [3, 8, 15],
					   "max_features": [int(mf/4), int(mf/2), mf],
					   "n_estimators": [mf, int(mf*3)]}

			if self.prob == 'Regression':
				
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
				scorer = make_scorer(mean_absolute_error)
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
			
			elif self.prob == "Multi-Classification":
				
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, stratify=self.target, shuffle=True)
				scorer = make_scorer(f1_score(average="weighted"))
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
			
			else:
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
				scorer = make_scorer(f1_score)
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
		
		else:
			if self.prob == 'Regression':
				
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
				scorer = make_scorer(mean_absolute_error)
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
			
			elif self.prob == "Multi-Classification":
				
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, stratify=self.target, shuffle=True)
				scorer = make_scorer(f1_score(average="weighted"))
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
			
			else:
				tr_x, te_x, tr_y, te_y = train_test_split(self.train, self.target, test_size=0.2, shuffle=True)
				scorer = make_scorer(f1_score)
				grid_search = GridSearchCV(clf, param_grid=self.param_dict, n_jobs=self.n_jobs, scoring=scorer)
				grid_search.fit(tr_x, tr_y.reshape(tr_y.shape[0],))
				
		
		return grid_search.best_estimator_, grid_search.best_score_, self.param_dict, self.n_jobs, scorer
		
	def pipeline(self, typ="Normalization", th=100, estim='RF'):
		
		"""
		Pipeline from preprocessing to training
		"""

		self.typ = typ
		self.th = th
		self.estim = estim
		
		logger.info("Pipeline has been initiated")
		prep_x, prep_y = self.preprocess(self.typ)
		clf, te_x, te_y, prob = self.model(prep_x, prep_y, self.estim, self.th)
		met = self.scoring(clf, te_x, te_y, prob)
		message = self.save_model(clf)
		logger.info("Pipeline is complete")

		return met, prob, clf
