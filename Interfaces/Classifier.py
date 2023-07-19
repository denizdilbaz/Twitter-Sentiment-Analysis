from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #Data testing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import pandas as pd
from keras import layers
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize

class ClassifierInterface(ABC):
    @abstractmethod
    def train(self):   #, X_train, y_train
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, Model):
        pass


