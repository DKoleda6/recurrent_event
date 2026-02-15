import numpy as np

class SurvivalTrainer:

    def __init__(self, model, metrics_list):
        self.model = model
        self.metrics_list = metrics_list

    def train(self, df):
        self.model.fit(df)

    def evaluate(self, df, survival_train, survival_test, times):

        predictions = self.model.predict_survival(df, times)
        estimate = predictions.values

        results = {}

        for metric in self.metrics_list:
            results[metric.__class__.__name__] = metric.compute(
                survival_train,
                survival_test,
                estimate,
                times
            )

        return results, predictions
