import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def train_val(y_train, y_train_pred, y_test, y_pred, model_name):
    
    scores = {
        f'{model_name}_train': {
            "R2" : r2_score(y_train, y_train_pred),
            "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        f'{model_name}_test':  {
            "R2" : r2_score(y_test, y_pred),
            "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))
        }
    }
    
    return pd.DataFrame(scores)



class FeatureImportances:
    """
    it takes coef: np.array from model.coef_
    it takes X   : encoded Feature DataFrame for using column names
    it takes str_model_name : e.g. 'linear_model'
    return pd.DataFrame for feature importances SORTED
    """
    
    def __init__(self, coef, X, str_model_name=""):
        self.coef = coef
        self.X    = X
        self.str_model_name = str_model_name
    
    def show_feature_importances_df(self):
        feature_importances = np.abs(self.coef)
        importances_df = pd.DataFrame({'Feature': self.X.columns, 'Importance': feature_importances})
        importances_df = importances_df.sort_values(by='Importance', ascending=False)
        return importances_df
    
    def barplot_feature_importances(self, n_bars=10):
        importances_df = self.show_feature_importances_df()
        sns.barplot(importances_df['Feature'].head(n_bars), importances_df['Importance'].head(n_bars))
        plt.xlabel('Feature')
        plt.ylabel('Absolute Coefficient')
        plt.title(f'{self.str_model_name} Feature Importances]')
        plt.xticks(rotation=90)
        plt.show()
        

def adjusted_r2_score(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adjusted_r2

def residual_analysis_df(df, y_test, y_pred, cols , log_transformed=True):
    """
    This function, is to make a comparison and analysis dataframe of residuals
    
    (df): it takes a dataframe df where the observations are stored (generally called X)
    
    (cols): it takes cols --> array or list of column names from dataframe X 
        to add analysis dataframe fo comparison
    
    it takes y_test and y_pred, generally numpy arrays
    
    it takes log_transformed: bool, default=True if target feature is log transformed
    
    returns a dataframe, where each row is X_test[cols] + y_test + y_pred + residuals
        with a sorted view by absolute value of residuals
    """  
    if log_transformed:
        y_test = np.exp(y_test)
        y_pred = np.exp(y_pred)
    residuals = y_test - y_pred
    indexes = y_test.index
    comp_df = pd.DataFrame(columns=['Real_Price', 'Predicted_Price', 'Residuals'], index=indexes)
    comp_df['Real_Price'] = y_test
    comp_df['Predicted_Price'] = y_pred
    comp_df['Residuals'] = np.abs(residuals)
    comp_df.sort_values(by='Residuals', inplace=True, ascending=False)
    comp_indexes = comp_df.index
    for col in cols:
        comp_df[col] = pd.Series()
    for comp_index in comp_indexes:
        comp_df.loc[comp_df.index==comp_index, cols] = df[df.index==comp_index][cols].values
        
    return comp_df