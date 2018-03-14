import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import os
print(os.getcwd())

data = pd.read_csv("emails.csv", encoding='latin-1')

train_X, test_X, train_y, test_y = train_test_split(data['text'],
                                                    data['spam'],
                                                    test_size=0.2,
                                                    random_state=10)

def test_model(model, params, train_X, train_y, test_X, test_y):
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
#    print(clf.get_params().keys())
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

    return clf.score(test_X, test_y)

## Now, test some classifier pipelines.

mnb = Pipeline([('vect', CountVectorizer(analyzer='word',
                                         tokenizer=str.split,
                                         stop_words='english',
                                         ngram_range=(1,2),
                                         strip_accents="unicode",
                                         lowercase=True)),
                ('clf', MultinomialNB(alpha=10))])
mnb_params = {'clf__alpha': [10, 1, 0.1, 0.01, 0.001]}
mnb_score = test_model(model=mnb, params=mnb_params, train_X=train_X, test_X=test_X,
                       train_y=train_y, test_y=test_y)

sgd = Pipeline([('vect', CountVectorizer(analyzer='word',
                                         tokenizer=str.split,
                                         stop_words='english',
                                         ngram_range=(1,2),
                                         strip_accents="unicode",
                                         lowercase=True)),
                ('clf', SGDClassifier())])
sgd_params = {'clf__alpha': [10, 1, 0.1, 0.01, 0.001],
              'clf__loss': ['hinge', 'log'],
              'clf__penalty': ['l2', 'elasticnet']}
sgd_score = test_model(model=sgd, params=sgd_params, train_X=train_X, test_X=test_X,
                       train_y=train_y, test_y=test_y)

rfc = Pipeline([('vect', CountVectorizer(analyzer='word',
                                         tokenizer=str.split,
                                         stop_words='english',
                                         ngram_range=(1,2),
                                         strip_accents="unicode",
                                         lowercase=True)),
                ('clf', RandomForestClassifier())])
rfc_params = {'clf__criterion': ['gini', 'entropy'],
              'clf__n_estimators': range(1,101,5)}
rfc_score = test_model(model=rfc, params=rfc_params, train_X=train_X, test_X=test_X,
                       train_y=train_y, test_y=test_y)
