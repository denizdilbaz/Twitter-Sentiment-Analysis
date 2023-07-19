from Classifiers.LSTM import LstmClassifier
from Classifiers.SVM import  SVMClassifier
from Classifiers.LogisticRegression import LogisticRegressionClassifier
from Classifiers.DecisionTree import decisionTreeClassifier
from PreProcessor.PreProcessing import PreProcessing
import pandas as pd




data = pd.read_csv('DuyguAnalizi.csv')
data.columns = ['id', 'information', 'type', 'tweet']
preprocessor = PreProcessing(data)
data = preprocessor.allStep()
print(data)

svm_classifier = SVMClassifier(data)
svm_sonuc,svm_test_sonuc = svm_classifier.allStep()


logistic_classifier = LogisticRegressionClassifier(data)
logistic_sonuc, logistic_test_sonuc = logistic_classifier.allStep()



decisionTree_classifier = decisionTreeClassifier(data)
decisionTree_sonuc, decisionTree_test_sonuc = decisionTree_classifier.allStep()


lstm_classifier = LstmClassifier(data)
lossAndAccuracy = lstm_classifier.allStep()


print(svm_sonuc,svm_test_sonuc)
print(logistic_sonuc, logistic_test_sonuc)
print(decisionTree_sonuc,decisionTree_test_sonuc)
print(lossAndAccuracy)