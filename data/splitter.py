import numpy as np


class SurvivalSplitter:

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_by_individual(self, df, id_col="name"):

        np.random.seed(self.random_state)

        unique_ids = df[id_col].unique()
        n_test = int(len(unique_ids) * self.test_size)

        shuffled_ids = np.random.permutation(unique_ids)

        test_ids = shuffled_ids[:n_test]
        train_ids = shuffled_ids[n_test:]

        train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
        test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)

        return train_df, test_df
