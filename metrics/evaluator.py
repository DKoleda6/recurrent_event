import numpy as np
import pandas as pd
from survivors.constants import get_y
from lifelines.utils import concordance_index
from metrics.recurrent_count_error import RecurrentCountError

class SurvivalEvaluator:
    def __init__(self, ibs_metric, auprc_metric):
        self.ibs_metric = ibs_metric
        self.auprc_metric = auprc_metric
        self.results = []

    def evaluate(
        self,
        model,
        model_name,
        train_df,
        test_df,
        features,
        times,
        duration_col="time",
        event_col="event",
    ):  
        predictions = model.predict_survival(test_df[features], times)
        estimate = predictions.values.astype(np.float64)

        survival_test = get_y(time=test_df[duration_col], cens=test_df[event_col].astype(bool))
        survival_train = get_y(time=train_df[duration_col], cens=train_df[event_col].astype(bool))

        mean_ibs, _ = self.ibs_metric.compute(survival_train, survival_test, estimate, times)
        mean_auprc = self.auprc_metric.compute(survival_train, survival_test, estimate, times)

        ci = concordance_index(
            survival_test["time"],
            np.trapz(estimate, times, axis=1),
            test_df[event_col]
        )

        recurrent_error = None
        if model_name != "Random Survival Forest":
            tr_pred = model.predict_cumulative_hazard(train_df, times)
            tr_max = np.quantile(tr_pred.max(), 0.9)

            pred = model.predict_cumulative_hazard(test_df, times)

            recerr = RecurrentCountError()
            recurrent_error = recerr.compute(
                survival_train=None,
                survival_test=test_df,
                estimate=pred / tr_max,
                times=times
            )

        self.results.append({
            "model": model_name,
            "IBS": mean_ibs,
            "AUPRC": mean_auprc,
            "C-index": ci,
            "recurrent_error": recurrent_error
        })
        return mean_ibs, mean_auprc, ci, recurrent_error


    def get_results_table(self):
        return pd.DataFrame(self.results)