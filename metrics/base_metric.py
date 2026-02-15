# metrics/base_metric.py

from abc import ABC, abstractmethod

class BaseMetric(ABC):

    @abstractmethod
    def compute(self, survival_train, survival_test, estimate, times):
        pass
