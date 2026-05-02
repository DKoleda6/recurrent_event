import pandas as pd
import numpy as np

class DataProcessor:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_prepare(self):
        df = pd.read_csv(self.filepath)
        
        df_sorted = df.sort_values(['patient_nbr', 'DATE']).reset_index(drop=True)
        
        cox_df = self.build_cox_dataframe(df_sorted)

        return cox_df

    def build_cox_dataframe(self, df_sorted):
        cox_data = []
        
        unique_patients = df_sorted['patient_nbr'].unique()

        for patient_id in unique_patients:
            patient_rows = df_sorted[df_sorted['patient_nbr'] == patient_id].reset_index(drop=True)
            
            if len(patient_rows) == 0:
                continue

            episode_num = 1
            
            for idx, row in patient_rows.iterrows():
                cox_data.append({
                    'patient_id': patient_id,
                    'episode_col': episode_num,
                    #'start': row['start'],
                    #'stop': row['stop'],
                    'start': row['start_days'],
                    'stop': row['stop_days'],
                    'event': row['event'],
                    
                    'race': row['race'],
                    'gender': row['gender'],
                    'age': row['age'],
                    #'weight': row['weight'],
                    
                    'admission_type_id': row['admission_type_id'],
                    'discharge_disposition_id': row['discharge_disposition_id'],
                    'admission_source_id': row['admission_source_id'],
                    'curr_dur_hosp': row['time_in_hospital'],

                    'num_lab_procedures': row['num_lab_procedures'],
                    'num_procedures': row['num_procedures'],
                    'num_medications': row['num_medications'],
                    'number_outpatient': row['number_outpatient'],
                    'number_emergency': row['number_emergency'],
                    'number_inpatient': row['number_inpatient'],
                    'number_diagnoses': row['number_diagnoses'],
                    
                    'diabetesMed': row['diabetesMed'],
                    'insulin': row['insulin'],
                    'change': row['change'],
                    #'readmitted': row['readmitted']
                })
                
                episode_num += 1

        cox_df = pd.DataFrame(cox_data)
        eps = 1e-6
        cox_df["start"] = cox_df["start"].astype(float)
        cox_df["stop"] = cox_df["stop"].astype(float)
        mask = cox_df['start'] >= cox_df['stop']
        cox_df.loc[mask, 'stop'] = cox_df.loc[mask, 'start'] + eps

        return cox_df