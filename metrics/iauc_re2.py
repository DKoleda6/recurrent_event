import numpy as np
import pandas as pd
from .base_metric import BaseMetric

class IAUCRE2(BaseMetric):
    def compute(self, survival_train, survival_test, estimate, times):
        test_df = survival_test.copy()
        test_df.reset_index(drop=True, inplace=True)
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
            t_mask = times >= row["stop"]
            if row["event"] == 1:
                obs_matrix[t_mask, person_ind[row_i]] += 1
            # else:
            #     obs_matrix[t_mask, person_ind[row_i]] = np.nan

        for t in times:
            person_scores = {}
            for name in person_names:
                person_episodes = test_df[test_df["name"] == name].copy()
                total_risk = 0.0
                for idx, ep in person_episodes.iterrows():
                    stop_ep = ep["stop"]
                    
                    if stop_ep <= t: #add the episodes that passed
                        t_prev = max([tp for tp in times if tp <= stop_ep])
                        total_risk += estimate.loc[idx, t_prev]
                    else: #add the still active episode
                        total_risk += estimate.loc[idx, t]
                        break
                person_scores[name] = total_risk
            pred_counts = np.array([person_scores[name] for name in person_names]) #array of values for time t
            pred_by_person.append(pred_counts) #matrix that consists of all the arrays at times t0, t1, ...

        pred_by_person = np.array(pred_by_person)

        for t_ind in range(len(times)):
            obs_counts = obs_matrix[t_ind]
            pred_counts = pred_by_person[t_ind]
            obs_i = np.tile(obs_counts[:, np.newaxis], (1, len(obs_counts)))
            obs_j = np.tile(obs_counts[np.newaxis, :], (len(obs_counts), 1))
            pred_i = np.tile(pred_counts[:, np.newaxis], (1, len(pred_counts)))
            pred_j = np.tile(pred_counts[np.newaxis, :], (len(pred_counts), 1))

            valid_mask = obs_i > obs_j
            correct_mask = pred_i > pred_j
            denominator = np.sum(valid_mask)
            numerator = np.sum(valid_mask & correct_mask)

            auc_re = 0.5 if denominator == 0 else numerator / denominator
            auc_re_per_time.append(auc_re)

        integral = np.trapz(auc_re_per_time, times)
        iauc_re = integral / (times[-1] - times[0]) if len(times) > 1 else 0.5
        
        return iauc_re