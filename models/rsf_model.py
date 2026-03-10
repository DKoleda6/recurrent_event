import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from models.base_model import BaseSurvivalModel


class RSFModel(BaseSurvivalModel):
    def __init__(self, features, mode="independent", use_episode=None, **params):
        super().__init__(features)
        self.params = params
        self.mode = mode
        self.use_episode = use_episode
        self.model = RandomSurvivalForest(**params)

    def fit(self, df):
        X = df[self.features]
        y = Surv.from_dataframe(
            event="event",
            time="time",
            data=df
        )
        self.model.fit(X, y)

    def predict_survival(self, df, times):
        surv_funcs = self.model.predict_survival_function(df[self.features])
        preds = np.zeros((len(surv_funcs), len(times)))
        for i, fn in enumerate(surv_funcs):
            preds[i, :] = fn(times)

        return pd.DataFrame(preds, columns=times)

    def predict_cumulative_hazard(self, df, times):
        surv = self.predict_survival(df, times)
        return -np.log(surv)