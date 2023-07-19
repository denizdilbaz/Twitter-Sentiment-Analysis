from Interfaces.Classifier import ClassifierInterface
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class SVMClassifier(ClassifierInterface):
    def __init__(self, data):
        self.data = data

    def token(self):
        bow_counts = CountVectorizer(
            tokenizer=word_tokenize,
            ngram_range=(1, 4)  # arka arkaya 4 kelime
        )

        reviews_train, reviews_test = train_test_split(self.data, test_size=0.3, random_state=0)
        X_train_bow = bow_counts.fit_transform(reviews_train.tweet)
        X_test_bow = bow_counts.transform(reviews_test.tweet)

        y_train_bow = reviews_train['type']
        y_test_bow = reviews_test['type']

        return X_train_bow, X_test_bow, y_train_bow, y_test_bow

    def train(self):
        X_train_bow, X_test_bow, y_train_bow, y_test_bow = self.token()
        svmodel = SVC(kernel='linear')
        params = {'kernel': ['linear', 'rgf'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto'],
                  'shrinking': [False, True], 'decision_function_shape': ['ovo', 'ovr']}

        grid_search = GridSearchCV(svmodel, param_grid=params, cv=5)
        grid_search.fit(X_train_bow, y_train_bow)
        best_svmModel = grid_search.best_estimator_
        return X_train_bow, X_test_bow, y_train_bow, y_test_bow,best_svmModel


    def evaluate(self, X_train_bow, y_train_bow, best_svmModel):
        k = 10
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)

        accuracy_sonuclari = cross_val_score(best_svmModel, X_train_bow, y_train_bow, cv=kfold, scoring='accuracy')
        precision_sonuclari = cross_val_score(best_svmModel, X_train_bow, y_train_bow, cv=kfold,
                                              scoring='precision_micro')
        recall_sonuclari = cross_val_score(best_svmModel, X_train_bow, y_train_bow, cv=kfold, scoring='recall_micro')
        f1_sonuclari = cross_val_score(best_svmModel, X_train_bow, y_train_bow, cv=kfold, scoring='f1_micro')

        sonuclar = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                 'Mean Score': [np.mean(accuracy_sonuclari),
                                                np.mean(precision_sonuclari),
                                                np.mean(recall_sonuclari),
                                                np.mean(f1_sonuclari)]})

        sonuclar['Mean Score'] = sonuclar['Mean Score'].map('{:.2f}'.format)

        return sonuclar

    def predict(self,X_test_bow,y_test_bow, best_dtModel):
        y_pred = best_dtModel.predict(X_test_bow)
        accuracy = accuracy_score(y_test_bow, y_pred)
        precision = precision_score(y_test_bow, y_pred, average='micro')
        recall = recall_score(y_test_bow, y_pred, average='micro')
        f1 = f1_score(y_test_bow, y_pred, average='micro')

        testVerisiSonuclar = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                 'Score': [accuracy,
                                           precision,
                                           recall,
                                           f1]})

        testVerisiSonuclar['Score'] = testVerisiSonuclar['Score'].map('{:.2f}'.format)

        print(testVerisiSonuclar)


    def allStep(self):
        X_train_bow, X_test_bow, y_train_bow, y_test_bow,model = self.train()
        sonuclar = self.evaluate(X_train_bow,y_train_bow,model)
        testSonuclari = self.predict(X_test_bow,y_test_bow,model)
        return sonuclar,testSonuclari
