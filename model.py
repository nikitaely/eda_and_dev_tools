import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from IPython.core.interactiveshell import InteractiveShell


dataset = "https://raw.githubusercontent.com/nikitaely/eda_and_dev_tools/main/online_shoppers_intention.csv"

if __name__ == '__main__':
    df = pd.read_csv(dataset)
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)
    df = df.fillna(0)

    numeric = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
           'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    categorical = ['VisitorType', 'Weekend', 'Month', 'TrafficType', 'Browser', 'OperatingSystems', 'Region', 'SpecialDay']


    X = df[numeric]
    y = df['Revenue'].astype(int)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

    parameters = {#'depth'         : [1,4,7,10],
                 'learning_rate' : [0.01,0.05,0.1]}

    gs_cat = GridSearchCV(estimator=CatBoostClassifier(verbose=False),
                     param_grid=parameters,
                     cv=3,
                     n_jobs=-1,
                     verbose=2,
                     scoring='f1')
    gs_cat.fit(Xtrain, ytrain)




    gs_cat.best_estimator_.save_model('trained_model.cbm')
