import pandas as pd
import numpy as np
import ast

class DataProcessor:

    def __init__(self, filepath):
        self.filepath = filepath

    def safe_literal_eval(self, x):
        if pd.isna(x) or x in ['nan', 'NaN']:
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    def load_and_prepare(self):
        df = pd.read_csv(self.filepath)

        df['all_future_arrest_times'] = df['all_future_arrest_times'].apply(self.safe_literal_eval)
        df['all_future_events'] = df['all_future_events'].apply(self.safe_literal_eval)

        df_sorted = df.sort_values(['name', 'in_custody']).reset_index(drop=True)
        df_first_row = df_sorted.groupby('name').first().reset_index()
        df_first_row = df_first_row[
            df_first_row['all_future_arrest_times'].apply(len) > 0
        ]

        cox_df = self.build_cox_dataframe(df_first_row)

        return cox_df

    def build_cox_dataframe(self, df_first_row):

        cox_data = []

        for _, row in df_first_row.iterrows():
            dur_list = row['all_future_arrest_times']
            entry_list = [0] + dur_list[:-1]
            event_list = [1]*(len(dur_list)-1) + [0]

            for i in range(len(dur_list)):
                if dur_list[i] <= 0:
                    continue

                cox_data.append({
                    'name': row['name'],
                    'episode_col': i+1,
                    'entry': entry_list[i],
                    'dur': dur_list[i],
                    'event': event_list[i],
                    'age': row['age'],
                    'sex': row['sex'],
                    'race': row['race'],
                    #'custody_num': row['custody_num'],
                    #'sum_dur_custody': row['sum_dur_custody'],
                    'curr_dur_custody': row['curr_dur_custody']
                })

        cox_df = pd.DataFrame(cox_data)

        eps = 1e-6
        cox_df["entry"] = cox_df["entry"].astype(float)
        cox_df["dur"] = cox_df["dur"].astype(float)
        mask = cox_df['entry'] >= cox_df['dur']
        cox_df.loc[mask, 'dur'] = cox_df.loc[mask, 'entry'] + eps

        return cox_df
