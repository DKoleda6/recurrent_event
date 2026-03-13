import numpy as np
import pandas as pd
from .base_metric import BaseMetric

class RecurrentCountError(BaseMetric):

    def compute(self, survival_train, survival_test, estimate, times):

        test_df = survival_test.copy()

        if not isinstance(estimate, pd.DataFrame):
            estimate = pd.DataFrame(
                estimate,
                index=test_df.index,
                columns=times
            )

        errors = []

        names = test_df["name"].unique()
        n_people = len(names)

        name_to_ind = {name: i for i, name in enumerate(names)}
        person_ind = test_df["name"].map(name_to_ind).values

        obs_matrix = np.zeros((len(times), n_people))

        for row_i, row in test_df.iterrows():
            if row["event"] == 1:
                t_mask = times >= row["time"]
                obs_matrix[t_mask, person_ind[row_i]] += 1

        for t_ind, t in enumerate(times):
            row_preds = estimate[times[t_ind]].values

            temp = pd.DataFrame({
                "name": test_df["name"].values,
                "pred": row_preds
            })

            pred_counts = temp.groupby("name")["pred"].sum().values

            # Observed cumulative events per person, takes too long (1 minute 7.7 seconds)
            '''obs_counts = (
                test_df.groupby("name")
                .apply(lambda df: ((df["time"] <= t) & (df["event"] == 1)).sum())
                .values
            )'''

            obs_counts = obs_matrix[t_ind]

            mse = np.mean((pred_counts - obs_counts) ** 2)
            errors.append(mse)

        integrated_error = np.trapz(errors, times) / (times[-1] - times[0])

        return integrated_error, obs_matrix, name_to_ind
