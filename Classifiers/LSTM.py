from Interfaces.Classifier import ClassifierInterface
import keras
from keras import layers
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

class LstmClassifier(ClassifierInterface):
    def __init__(self, data):
        self.data = data

    def model(self,X):
        Lstm = keras.Sequential([
            keras.layers.Embedding(500, 200, input_length=X.shape[1]),
            keras.layers.SpatialDropout1D(0.4),
            keras.layers.LSTM(176, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(72, activation='relu'),
            keras.layers.Dense(4, activation='softmax')
        ])
        Lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(Lstm.summary())
        return Lstm

    def token(self):
        tokenizer = Tokenizer(num_words=500, split=' ')
        tokenizer.fit_on_texts(self.data['tweet'].values)
        X = tokenizer.texts_to_sequences(self.data['tweet'].values)
        X = pad_sequences(X)
        return X

    def train(self):
        X = self.token()
        Lstm = self.model(X)
        y = pd.get_dummies(self.data['type'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        batch_size = 48
        Lstm.fit(X_train, y_train, epochs=10, batch_size=batch_size)
        return Lstm


    def evaluate(self, X_test, y_test,Lstm):
        return Lstm.evaluate(X_test,y_test)


    def allStep(self):
        Lstm = self.train()
        lossAndAccuracy =self.evaluate(Lstm = Lstm)
        return lossAndAccuracy
