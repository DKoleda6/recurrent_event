from survivors import metrics
from .base_metric import BaseMetric

class IBSRemainMetric(BaseMetric):

    def compute(self, survival_train, survival_test, estimate, times):

        return metrics.ibs_remain(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=estimate,
            times=times,
            axis=0
        )
