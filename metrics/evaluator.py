import numpy as np
import pandas as pd
from survivors.constants import get_y
from lifelines.utils import concordance_index
from metrics.recurrent_count_error import RecurrentCountError
from metrics.iauc_re1 import IAUCRE1

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
        try:
            tr_pred = model.predict_cumulative_hazard(train_df, times)
            tr_max = np.quantile(tr_pred.max(), 0.9)
            pred = model.predict_cumulative_hazard(test_df, times)
            pred_array = pred.values if isinstance(pred, pd.DataFrame) else pred
            if pred_array.shape[0] == len(times) and pred_array.shape[1] == len(test_df):
                pred_array = pred_array.T

            recerr = RecurrentCountError()
            recurrent_error, obs_matrix, name_to_ind = recerr.compute(
                survival_train=None,
                survival_test=test_df,
                estimate=pred_array / tr_max,
                times=times
            )
        except Exception as e:
            print(f"Warning: Recurrent error failed for {model_name}: {str(e)}")
            recurrent_error = np.nan

        iaucre_metric = IAUCRE1()
        iauc_re = iaucre_metric.compute(
            survival_train=None,
            survival_test=test_df,
            estimate=pred_array / tr_max,
            times=times,
            obs_matrix=obs_matrix,
            name_to_ind=name_to_ind
        )

        self.results.append({
            "model": model_name,
            "IBS": mean_ibs,
            "AUPRC": mean_auprc,
            "C-index": ci,
            "recurrent_error": recurrent_error,
            "IAUC_RE": iauc_re
        })
        return mean_ibs, mean_auprc, ci, recurrent_error, iauc_re


    def get_results_table(self):
        return pd.DataFrame(self.results)