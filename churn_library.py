"""
Author: PP

This notebook for predicting customer churn
"""
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(20, 10))
    churn_plot = df['Churn'].hist()
    plt.savefig("./images/eda/churn_plot.png", dpi=300)

    plt.figure(figsize=(20, 10))
    customer_plot = df['Customer_Age'].hist()
    plt.savefig("./images/eda/customer_plot.png", dpi=300)

    plt.figure(figsize=(20, 10))
    marital_plot = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/marital_plot.png", dpi=300)

    plt.figure(figsize=(20, 10))
    totoal_transaction = sns.histplot(
        df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("./images/eda/transaction_plot.png", dpi=300)

    plt.figure(figsize=(20, 10))
    corr_plot = sns.heatmap(
        df_numeric.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig("./images/eda/heatmap_plot.png", dpi=300)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for cat in category_lst:
        col_name = f'{cat}_Churn'
        df[col_name] = df[cat].map(df.groupby(cat)['Churn'].mean())
    df = df[response]
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    x = df.drop(['Churn'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)
    plt.savefig("./images/eda/importance.png", dpi=300)
