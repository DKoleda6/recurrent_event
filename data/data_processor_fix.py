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
        df_sorted = df.sort_values(['name', 'in_custody']).reset_index(drop=True)

        cox_df = self.build_cox_dataframe(df_sorted)

        return cox_df

    def build_cox_dataframe(self, df_sorted):
        cox_data = []
        unique_people = df_sorted['name'].unique()

        for person_name in unique_people:
            person_rows = df_sorted[df_sorted['name'] == person_name].reset_index(drop=True)
            if len(person_rows) == 0:
                continue

            full_dur_list = person_rows.iloc[0]['all_future_arrest_times']
            if len(full_dur_list) < 1:
                continue

            covariate_rows = person_rows.iloc[1:(len(full_dur_list)+1)]
            if len(covariate_rows) < 1:
                continue

            entry = 0
            episode_col = 1

            for i in range(len(full_dur_list)):
                dur = full_dur_list[i]
                event = 1 if i < (len(full_dur_list) - 1) else 0
                covar_row = covariate_rows.iloc[i]
                
                cox_data.append({
                    'name': person_name,
                    'episode_col': episode_col,
                    'start': entry,
                    'stop': dur,
                    'event': event,
                    'age': covar_row['age'],
                    'time_since_last_arrest': covar_row['time_since_last_arrest'],
                    'average_dur_custody': covar_row['average_dur_custody'],
                    'curr_dur_custody': covar_row['curr_dur_custody']
                })

                entry = dur
                episode_col += 1

        cox_df = pd.DataFrame(cox_data)
        #cox_df['average_dur_custody'] = cox_df['average_dur_custody'].round(2)

        return cox_df
