import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

## need to get data
DF_TRAIN = pd.read_csv('training.csv')

class Model:

  def __init__(self):
    self.raw_data = DF_TRAIN
    self.num_vars = ['VehYear', 'VehAge_Purch', 'VehOdo', 'VehBCost', 'WarrantyCost']
    self.cat_vars = ['Auction',
            'Make',
            #'Model',
            'Transmission', 'Nationality', 'Size', 'TopThreeAmericanName',
            'price_to_auction', 'price_to_retail', 'price_to_current_auction', 'price_to_current_retail',
            'PRIMEUNIT', 'AUCGUART', 'VNST', 'IsOnlineSale']

  def prepare_data(self, df):
    df = self.convert_price(df)
    df = self.cols_fillna(df)
    df = self.get_years(df)
    df = df[self.num_vars + self.cat_vars + ['IsBadBuy']]
    df = df.dropna()
    dummy_vars = []
    for c in self.cat_vars:
      dummies = pd.get_dummies(df[c], prefix = c)
      df = pd.concat([df, dummies], axis=1)
      dummy_vars = dummy_vars + (list(dummies.columns.values))
    X_cols = dummy_vars + self.num_vars
    X = df[X_cols]
    y = df['IsBadBuy']
    return X, y

  def get_test_training(self, X, y):
    self.X_training, self.X_test, self.y_training,  self.y_test =  train_test_split(X, y, test_size =.9)
    #ros = RandomOverSampler(random_state=42)
    #self.X_training, self.y_training = ros.fit_sample(self.X_training, self.y_training)

  def rf_cv(self, X, y):
    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    est = RandomForestClassifier(n_estimators= 20)
    cv = GridSearchCV(est, param_grid)
    cv.fit(X, y)
    return cv.best_params_

  def train(self, X, y):
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_sample(X, y)
    best_params = self.rf_cv(X, y)
    rf_model = RandomForestClassifier(n_estimators = 20,
                                      max_features = best_params['max_features'],
                                      min_samples_split = best_params['min_samples_split'],
                                      min_samples_leaf = best_params['min_samples_leaf'],
                                      criterion = best_params['criterion'],
                                      bootstrap = best_params['bootstrap'],
                                      max_depth = best_params['max_depth']
                                      )
    rf_model.fit(X, y)
    return rf_model, rf_model.score(self.X_test, self.y_test)

  def model_perf(self, grids = [.01, .015, .02], sample_perc = .005, iterations = 10, mode = 'regular'):
    avg_scores = []
    for inc in grids:
      scores = []
      for i in range(0, iterations):
        test_ratio = 1 - inc if mode == 'regular' else 1 - inc - sample_perc
        X_training1, X_training2, y_training1, y_training2 = train_test_split(self.X_training, self.y_training, test_size = test_ratio)
        rf_model, score = self.train(X_training1, y_training1)
        if mode != 'regular':
          cols = X_training1.columns
          X_training2['pred'] = rf_model.predict(X_training2)
          X_training2['actual'] = y_training2

          total_n = int(len(self.X_training.index) * sample_perc)
          p_n = min(X_training2['pred'].sum(), int(total_n/2))
          n_n = total_n - p_n

          X_training2_p = X_training2.loc[X_training2.pred == 1]
          X_training2_n = X_training2.loc[X_training2.pred == 0]
          df_p = X_training2_p.sample(n = p_n)
          df_n = X_training2_n.sample(n = n_n)
          X_training1 = X_training1.append(df_p[cols])
          X_training1 = X_training1.append(df_n[cols])
          y_training1 = y_training1.append(df_p['actual'])
          y_training1 = y_training1.append(df_n['actual'])
          rf_model, score = self.train(X_training1, y_training1)

        scores = scores + [score]
      avg_scores = avg_scores + [np.mean(score)]
    return avg_scores

  def convert_price(self, df):
    df['price_to_auction'] = 'm'
    df.loc[(df.VehBCost <= df.MMRAcquisitionAuctionAveragePrice),'price_to_auction'] = 'l'
    df.loc[(df.VehBCost >= df.MMRAcquisitionAuctionCleanPrice),'price_to_auction'] = 'm'
    df.loc[(df.VehBCost > df.MMRAcquisitionAuctionAveragePrice) &
             (df.VehBCost < df.MMRAcquisitionAuctionCleanPrice),'price_to_auction'] = 'h'
    df['price_to_retail'] = 'm'
    df.loc[(df.VehBCost <= df.MMRAcquisitionRetailAveragePrice),'price_to_retail'] = 'l'
    df.loc[(df.VehBCost >= df.MMRAcquisitonRetailCleanPrice),'price_to_retail'] = 'm'
    df.loc[(df.VehBCost > df.MMRAcquisitionRetailAveragePrice) &
             (df.VehBCost < df.MMRAcquisitonRetailCleanPrice),'price_to_retail'] = 'h'
    df['price_to_current_auction'] = 'm'
    df.loc[(df.VehBCost <= df.MMRCurrentAuctionAveragePrice),'price_to_current_auction'] = 'l'
    df.loc[(df.VehBCost >= df.MMRCurrentAuctionCleanPrice),'price_to_current_auction'] = 'm'
    df.loc[(df.VehBCost > df.MMRCurrentAuctionAveragePrice) &
              (df.VehBCost < df.MMRCurrentAuctionCleanPrice),'price_to_current_auction'] = 'h'
    df['price_to_current_retail'] = 'm'
    df.loc[(df.VehBCost <= df.MMRCurrentRetailAveragePrice),'price_to_current_retail'] = 'l'
    df.loc[(df.VehBCost >= df.MMRCurrentRetailCleanPrice),'price_to_current_retail'] = 'm'
    df.loc[(df.VehBCost > df.MMRCurrentRetailAveragePrice) &
             (df.VehBCost < df.MMRCurrentRetailCleanPrice),'price_to_current_retail'] = 'h'
    return df

  def cols_fillna(self, df):
    df['AUCGUART'] = df['AUCGUART'].fillna('GREEN')
    df['PRIMEUNIT'] = df['PRIMEUNIT'].fillna('NO')
    return df

  def get_years(self, df):
    df['PurchYear'] = df['PurchDate'].apply(lambda x: x.split('/')[2])
    df['VehAge_Purch'] = df['PurchYear'].astype(int) - df['VehYear'].astype(int)
    return df

if __name__ == "__main__":
  rfmodel = Model()
  X, y = rfmodel.prepare_data(rfmodel.raw_data)
  rfmodel.get_test_training(X, y)
  print rfmodel.model_perf()
  print rfmodel.model_perf(mode = 'pursuit')
