from abc import ABC, abstractmethod

class BaseSurvivalModel(ABC):

    def __init__(self, features):
        self.features = features
        self.model = None

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def predict_survival(self, df, times):
        pass
