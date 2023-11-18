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