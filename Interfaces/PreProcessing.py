from abc import ABC, abstractmethod

class PreProcessingInterface(ABC):
    @abstractmethod
    def allStep(self):
        pass



