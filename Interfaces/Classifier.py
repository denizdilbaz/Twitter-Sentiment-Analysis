from abc import ABC, abstractmethod


class ClassifierInterface(ABC):
    @abstractmethod
    def train(self):   #, X_train, y_train
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, Model):
        pass


