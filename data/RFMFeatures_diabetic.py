import numpy as np

class RFMFeatures:
    def __init__(self, id_col="name", episode_col="episode_col"):
        self.id_col = id_col
        self.episode_col = episode_col

    def build_past_values(self, feature_values):
        history_vals = []
        for i in range(len(feature_values)):
            history_vals.append(feature_values.iloc[:i].tolist())
        return history_vals

    def min_max_mean(self, history_list, oper):
        if not history_list or len(history_list) == 0:
            return 0.0
        try:
            if oper == "min":
                return np.min(history_list)
            elif oper == "max":
                return np.max(history_list)
            elif oper == "mean":
                return np.mean(history_list)
        except:
            return 0.0
        return 0.0

    def custom_rfm(self, df):
        df = df.copy()
        
        df['log_curr_dur_custody'] = np.log1p(df['curr_dur_hosp'])
        df['dur_deviation_from_avg'] = df['curr_dur_hosp'] - df['curr_dur_hosp_mean']
        def is_increasing(row):
            history = row['curr_dur_hosp_history']
            if len(history) == 0:
                return False
            return row['curr_dur_hosp'] > history[-1]
        df['hosp_stay_increasing'] = df.apply(is_increasing, axis=1)
        #df['hosp_stay_speed_trend'] = df.groupby(self.id_col)['time_since_last_arrest'].diff().fillna(0)
        return df

    def create(self, df, history_features, operations):
        df = df.copy().sort_values([self.id_col, self.episode_col])

        for feature in history_features:
            df[f"{feature}_history"] = df.groupby(self.id_col)[feature].transform(self.build_past_values)

        for feature in history_features:
            for oper in operations:
                df[f"{feature}_{oper}"] = df[f"{feature}_history"].apply(lambda x: self.min_max_mean(x, oper))

        df = self.custom_rfm(df)

        return df