import numpy as np
import pandas as pd
from .base_metric import BaseMetric

class IAUCRE(BaseMetric):
    def compute(self, survival_train, survival_test, estimate, times, obs_matrix, name_to_ind):
        test_df = survival_test.copy()
        if not isinstance(estimate, pd.DataFrame):
            estimate = pd.DataFrame(estimate, index=test_df.index, columns=times)

        auc_re_per_time = []
        for t_ind, t in enumerate(times):
            obs_counts = obs_matrix[t_ind]
            row_preds = estimate[t].values
            pred_df = pd.DataFrame({"name": test_df["name"].values, "pred": row_preds})
            pred_counts = pred_df.groupby("name")["pred"].sum().values

            numerator = 0
            denominator = 0
            n_people = len(obs_counts)
            for i in range(n_people):
                for j in range(n_people):
                    if obs_counts[i] > obs_counts[j]:
                        denominator += 1
                        if pred_counts[i] >= pred_counts[j]:
                            numerator += 1

            auc_re = 0.5 if denominator == 0 else numerator / denominator
            auc_re_per_time.append(auc_re)

        integral = np.trapz(auc_re_per_time, times)
        iauc_re = integral / (times[-1] - times[0]) if len(times) > 1 else 0.5
        
        return iauc_re