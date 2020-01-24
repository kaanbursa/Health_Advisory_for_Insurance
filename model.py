import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import warnings
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFECV

from itertools import combinations



class Regressor():

    def __init__(self):
        self.linreg = LinearRegression()
        self.cros_val = KFold(n_splits=10, random_state=1, shuffle=True)
        self.scaler = MinMaxScaler()

    def vanilla_lin_reg(self,X,y):
        X = self.scaler.fit_transform(xset)

        #X_train, X_test, y_train, y_test = train_test_split(X,y)

        r2_baseline = np.mean(cross_val_score(self.linreg,X,y,scoring='r2',cv=self.cros_val))
        mse_baseline = np.mean(cross_val_score(self.linreg,X,y,scoring='mse',cv=self.cros_val))
        return r2_baseline,mse_baseline

    def run_poly(self,X,y,degree,plot = False):
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        poly = PolynomialFeatures(degree)
        X_fin = poly.fit_transform(X_train)
        X_t = poly.fit_transform(X_test)
        poly_reg = self.linreg.fit(X_fin,y_train)
        y_pred = poly_reg.predict(X_t)

        if plot:
            plt.plot(ypred, color='b',label="Predicted")
            plt.plot(X['Life_Expectancy'], color='g',label="True Life Expectancy")
            plt.legend()
            plt.shot()
        return r2_score(y_test,y_pred), mean_squared_error(y_test,y_pred)

    def interactions(self, df,target,baseline):
        features = df.drop([target],axis=1)
        y = df[target]
        interactions_l = []
        data = features.copy()
        X = self.scaler.fit_transform(data)
        for comb in combs:
            data['interaction'] = data[comb[0]] * data[comb[1]]
            #data.drop(['Life_Expectancy'],axis=1,inplace=True)
            score = np.mean(cross_val_score(linreg, data, y, scoring='r2', cv=cros_val))
            print(score)
            if score > baseline: interactions_l.append((comb[0], comb[1], round(score,3)))
        return sorted(interactions, key=lambda inter: inter[2], reverse=True)[:3]
