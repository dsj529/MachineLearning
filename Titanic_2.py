'''
Comparison of multiple classification algorithms on the Titanic survivors dataset

This code is based on work by Kaggle users Helge Bjorland and Anisotropic,
mostly in the feature engineering sections.

Author: Dale Josephs
Created: 20 Mar 2018

'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Classifier models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Other SKLearn tools
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# a set of visualization aides
def plot_histogram(df, vars, n_rows, n_cols):
    fig = plt.figure(figsize=(16, 12))
    for i, var_name in enumerate(vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title('Skew: {}'.format(round(float(df[var_name].skew()),)))
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
    fig.tight_layout()
    plt.show()

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()

def plot_correlation_map(df):
    corr = train.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9},
                    ax=ax, annot=True, annot_kws={'fontsize': 12})

def describe_more(df):
    var = []
    l = []
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels

def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)

def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(model.feature_importances_,
                       columns=['Importance'],
                       index=X.columns)
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[:15].plot(kind='barh')
    print(model.score(X, y))

##
# Next, load the Titanic data
train = pd.read_csv('/home/dsj529/Documents/PyProjects/Titanic/train.csv')
test = pd.read_csv('/home/dsj529/Documents/PyProjects/Titanic/test.csv')
full_data = [train, test]

# feature engineering, based on work by Sina, Anisotropic, and Helgejo
# some helper dicts for binning passenger titles together
titles_dict = {"Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer",
               "Rev": "Officer", "the Countess": "Royalty", "Dona": "Royalty",
               "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr", "Mrs": "Mrs",
               "Miss": "Miss", "Master": "Mr", "Lady": "Royalty"}
title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Officer': 4, 'Royalty': 5}

# first, add engineered columns.
for dataset in full_data:
    # FamilySize is a combination of SibSp and Parch (and self)
    dataset['FamilyQty'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Find singleton passengers
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilyQty'] == 1, 'IsAlone'] = 1
    # Remove NULL values from the embarkation data
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove NULL values from fare data;
    # Create new training column of Categorical fare data
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
#    train['CategoricalFare'] = pd.qcut(train['Fare'], 4, duplicates='drop')
    # Remove NULL values from age data
    # Create new training column of categorical age data
    dataset['Age'] = dataset['Age'].fillna(dataset.Age.mean())
#    train['CategoricalAge'] = pd.cut(train['Age'], 5)
    # separate and bin passenger titles
    dataset['Title'] = dataset['Name'].map(
        lambda name: name.split(',')[1].split('.')[0].strip())
    dataset['Title'] = dataset.Title.map(titles_dict)
    dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    # next, start mapping categorical labels into numerical values
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C': 1, 'Q': 2})

    # bin fare data into nominal levels
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # bin passenger ages
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    # bin family sizes
    dataset['FamilySize'] = dataset['FamilyQty']
    dataset['FamilySize'] = dataset['FamilySize'].map(lambda x: 1 if x == 1 else x)
    dataset['FamilySize'] = dataset['FamilySize'].map(lambda x: 2 if 2 <= x <= 4 else x)
    dataset['FamilySize'] = dataset['FamilySize'].map(lambda x: 3 if x > 5 else x)

##
# Next some helper function to evaluate model performance and feature selection

def test_model1(model, params, train_X, train_y, test_X, test_y):
    '''This function tests a classifier model over a set of parameter values
       and returns information about the CV and classification report data.
       Parameters
       model: an estimator
       params: parameters for the grid search
       train_X: data to train the estimator
       train_y: labels to train the estimator
       test_X: data to test the estimator
       test_y: data to generrate the classification report.
       Returns clf.score.'''

    clf = GridSearchCV(model, params, cv=5, n_jobs=6)
    clf.fit(train_X, train_y)

    print(model.named_steps['clf'])

    print('\n')

    for fold in zip(clf.cv_results_['mean_test_score'],
                    clf.cv_results_['std_test_score'],
                    clf.cv_results_['params']):
        print("M: {:8.5f} Sd: {:8.5f} | {}".format(*fold))
    print('\nBest param values: {}\n'.format(str(clf.best_params_)))
    print('\n')

    pred_y = clf.predict(test_X)
    print(classification_report(test_y, pred_y))

#    return clf.score(test_X, test_y)

def test_model(model, drop_cols, train_X, train_y, test_X, plot_imps=False):
    train_X = train_X.drop(drop_cols, axis=1)
    test_X = test_X.drop(drop_cols, axis=1)
    clf = model.fit(train_X, train_y)
    pred_y1 = clf.predict(train_X)
    if plot_imps:
        plot_model_var_imp(clf, train_X, train_y)
    print(classification_report(train_y, pred_y1))
    return clf.predict(test_X)

def test_model2(model, use_cols, train_X, train_y, test_X, plot_imps=False):
    train_X = train_X[use_cols]
    clf = model.fit(train_X, train_y)
    pred_y1 = clf.predict(train_X)
    if plot_imps:
        plot_model_var_imp(clf, train_X, train_y)
    print(classification_report(train_y, pred_y1))
    return clf.predict(test_X[use_cols])
##
# Time to model stuff

train_X = train
train_X = train_X.drop(['Survived'], axis=1)
train_y = train['Survived']
test_X = test

tree = DecisionTreeClassifier()
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
tree_preds = test_model(tree, drop_cols, train_X, train_y, test_X, True)

sgd = SGDClassifier()
sgd_preds = test_model(sgd, drop_cols, train_X, train_y, test_X)

gnb = GaussianNB()
gnb_preds = test_model(gnb, drop_cols, train_X, train_y, test_X)

rfc = RandomForestClassifier()
rfc_preds = test_model(rfc, drop_cols, train_X, train_y, test_X, True)


## Test classification based on the highest-ranked features in the decision tree models
use_cols = ['Sex', 'Title', 'Pclass', 'Age', 'Fare']
tree_preds2 = test_model2(tree, use_cols, train_X, train_y, test_X, True)
sgd_preds2 = test_model2(sgd, use_cols, train_X, train_y, test_X)
gnb_preds2 = test_model2(gnb, use_cols, train_X, train_y, test_X)
rfc_preds2 = test_model2(rfc, use_cols, train_X, train_y, test_X, True)
#  fascinatingly, tree accuracy drops, non-tree accuracy rises, and everything seems
#  to converge at 84%.
