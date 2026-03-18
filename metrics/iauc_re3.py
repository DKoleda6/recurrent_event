import numpy as np
import pandas as pd
from .base_metric import BaseMetric

class IAUCRE3(BaseMetric):
    def compute(self, survival_train, survival_test, estimate, times):
        test_df = survival_test.copy()
        if not isinstance(estimate, pd.DataFrame):
            estimate = pd.DataFrame(estimate, index=test_df.index, columns=times)

        auc_re_per_time = []
        person_names = test_df["name"].unique()
        pred_by_person = []

        n_people = len(person_names)
        name_to_ind = {name: i for i, name in enumerate(person_names)}
        person_ind = test_df["name"].map(name_to_ind).values
        obs_matrix = np.zeros((len(times), n_people))

        for row_i, row in test_df.iterrows():
            if row["event"] == 1:
                t_mask = times >= row["time"]
                obs_matrix[t_mask, person_ind[row_i]] += 1

        first_indices = test_df.groupby("name").head(1).index
        estimate_first = estimate.loc[first_indices]

        for t in times:
            pred_sum = pd.Series(
                estimate_first[t].values,
                index=test_df.loc[first_indices, "name"]
            )
            pred_counts = pred_sum.reindex(person_names).values
            pred_by_person.append(pred_counts)

        pred_by_person = np.array(pred_by_person)

        for t_ind in range(len(times)):
            obs_counts = obs_matrix[t_ind]
            pred_counts = pred_by_person[t_ind]
            obs_i = np.tile(obs_counts[:, np.newaxis], (1, len(obs_counts)))
            obs_j = np.tile(obs_counts[np.newaxis, :], (len(obs_counts), 1))
            pred_i = np.tile(pred_counts[:, np.newaxis], (1, len(pred_counts)))
            pred_j = np.tile(pred_counts[np.newaxis, :], (len(pred_counts), 1))

            valid_mask = obs_i > obs_j
            correct_mask = pred_i >= pred_j
            denominator = np.sum(valid_mask)
            numerator = np.sum(valid_mask & correct_mask)

            auc_re = 0.5 if denominator == 0 else numerator / denominator
            auc_re_per_time.append(auc_re)

        integral = np.trapz(auc_re_per_time, times)
        iauc_re = integral / (times[-1] - times[0]) if len(times) > 1 else 0.5
        
        return iauc_re