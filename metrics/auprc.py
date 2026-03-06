from survivors import metrics
from .base_metric import BaseMetric

class AUPRCMetric(BaseMetric):

    def compute(self, survival_train, survival_test, estimate, times):

        return metrics.auprc(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=estimate,
            times=times
        )
