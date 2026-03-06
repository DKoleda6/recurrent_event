import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

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
        #feature_col,
        duration_col,
        event_col,
        times=None
    ):

        # time grid
        train_dur = train_df[duration_col]
        if times is None:
            horizon = np.quantile(train_dur, 0.95)
            times = np.linspace(0, horizon, 200)
        times = np.asarray(times, dtype=np.float64)

        # predictions
        predictions = model.predict_survival(
            test_df,
            times
        )
        estimate = predictions.values.astype(np.float64)

        test_dur = test_df[duration_col]

        # fixing format
        survival_test = np.array(
            list(zip(
                test_df[event_col].astype(bool),
                test_dur.astype(float)
            )),
            dtype=[("event", "?"), ("time", "f8")]
        )

        survival_train = np.array(
            list(zip(
                train_df[event_col].astype(bool),
                train_dur.astype(float)
            )),
            dtype=[("event", "?"), ("time", "f8")]
        )

        # IBS
        mean_ibs, _ = self.ibs_metric.compute(
            survival_train,
            survival_test,
            estimate,
            times
        )

        # AUPRC
        survival_test_cens = np.array(
            list(zip(
                ~test_df[event_col].astype(bool),
                test_dur.astype(float)
            )),
            dtype=[("cens", "?"), ("time", "f8")]
        )

        survival_train_cens = np.array(
            list(zip(
                ~train_df[event_col].astype(bool),
                train_dur.astype(float)
            )),
            dtype=[("cens", "?"), ("time", "f8")]
        )

        auprc = self.auprc_metric.compute(
            survival_train_cens,
            survival_test_cens,
            estimate,
            times
        )

        mean_auprc = np.mean(auprc)

        # concordance
        ci = concordance_index(
            test_dur,
            np.trapz(estimate, times, axis=1),
            test_df[event_col]
        )

        # storing results
        self.results.append({
            "model": model_name,
            "IBS": mean_ibs,
            "AUPRC": mean_auprc,
            "C-index": ci
        })

        return mean_ibs, mean_auprc, ci


    def get_results_table(self):
        return pd.DataFrame(self.results)