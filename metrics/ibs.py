from survivors import metrics
from .base_metric import BaseMetric

class IBSMetric(BaseMetric):

    def compute(self, survival_train, survival_test, estimate, times):

        mean_ibs = metrics.ibs_remain(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=estimate,
            times=times,
            axis=-1
        )

        ibs_by_time = metrics.ibs_remain(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=estimate,
            times=times,
            axis=1
        )

        return mean_ibs, ibs_by_time
