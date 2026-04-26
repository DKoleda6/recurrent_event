import pandas as pd
import numpy as np

class DataProcessor:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_prepare(self):
        """Загружает данные и строит Cox-таблицу как в твоей рабочей функции"""
        # Загружаем CSV
        df = pd.read_csv(self.filepath)
        
        # Сортируем как в оригинале
        df_sorted = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
        
        # Строим эпизодическую таблицу
        cox_df = self.build_cox_dataframe(df_sorted)
        
        return cox_df

    def build_cox_dataframe(self, df_sorted):
        """Точная копия логики из build_episode_based_timeline"""
        df = df_sorted.copy()
        eps = 1e-6

        # Номер эпизода (1,2,3...)
        df["episode_col"] = df.groupby("patient_nbr").cumcount() + 1

        # Кумулятивное время в больнице
        cum_time = df.groupby("patient_nbr")["time_in_hospital"].cumsum()

        # Start = 0 для первого эпизода
        df["start"] = cum_time.groupby(df["patient_nbr"]).shift(fill_value=0)

        # Stop = start + текущая длительность
        df["stop"] = df["start"] + df["time_in_hospital"]

        # Исправляем некорректные интервалы
        mask = df['start'] >= df['stop']
        df.loc[mask, 'stop'] = df.loc[mask, 'start'] + eps

        # Событие: 1 если повторная госпитализация
        df["event"] = df["readmitted"].apply(lambda x: 1 if x in ["<30", ">30"] else 0)

        # Кодируем пол
        df["sex"] = df["gender"].map({"Female": 0, "Male": 1})

        # Оставляем только нужные столбцы + переименовываем
        cox_df = df[[
            "patient_nbr",
            "episode_col",
            "start",
            "stop",
            "event",
            "age",
            "sex",
            "race",
            "medical_specialty",
            "time_in_hospital"
        ]].rename(columns={
            "patient_nbr": "patient_id",
            "time_in_hospital": "curr_dur_hosp"
        })

        # Типы данных
        cox_df["start"] = cox_df["start"].astype(float)
        cox_df["stop"] = cox_df["stop"].astype(float)

        return cox_df