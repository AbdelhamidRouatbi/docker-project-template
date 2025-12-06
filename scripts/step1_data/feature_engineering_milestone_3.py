import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import wandb

import os
import pandas as pd
import numpy as np
import math

class FeatureEngineering:

    def __init__(self, data_path_csv="./ift6758/data/nhl/csv",
                 save_data_path="./ift6758/data/nhl/csv/processed"):
        self._data_path_csv = data_path_csv
        self._save_data_path = save_data_path

        self._cached_game = None
    

    def combine_df(self):
        csv_dir = self._data_path_csv
        dfs = []

        for fname in sorted(os.listdir(csv_dir)):
            if fname.lower().endswith(".csv"):
                full_path = os.path.join(csv_dir, fname)
                print(full_path)
                try:
                    df = pd.read_csv(full_path, dtype={"situation_code": "string"})
                    dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Could not read {fname}: {e}")

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    

    def assign_net(self, df: pd.DataFrame) -> pd.Series:
        period = pd.to_numeric(df["period"], errors="coerce")
        home_flag = df["home"].astype("boolean")

        odd_period = (period % 2 == 1)
        home_def_x = np.where(odd_period, 89, -89)
        away_def_x = np.where(odd_period, -89, 89)

        net_x = np.where(home_flag, home_def_x, away_def_x)
        return pd.Series(net_x, index=df.index)


    def calculate_distance_from_net(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        x = pd.to_numeric(df["x_coord"], errors="coerce")
        y = pd.to_numeric(df["y_coord"], errors="coerce")

        defend_net_x = self.assign_net(df)
        dx = defend_net_x - x
        dy = -y
        dist = np.sqrt(dx**2 + dy**2)

        median_dist = np.nanmedian(dist)
        needs_flip = median_dist > 80

        if needs_flip:
            defend_net_x = -defend_net_x
            dx = defend_net_x - x
            dy = -y
            dist = np.sqrt(dx**2 + dy**2)

        df["distance_from_net"] = dist
        df["shot_angle"] = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))

        return df




    def calculate_empty_net(self, df: pd.DataFrame):
        df = df.copy()
        sc = df["situation_code"].astype(str)
        home = df["home"].astype("boolean")

        first_digit = sc.str[0]
        last_digit  = sc.str[-1]

        empty_net = (
            ((last_digit == "0") & (~home)) |
            ((first_digit == "0") & (home))
        )

        df["empty_net"] = empty_net.astype(int)
        df["empty_net_goalie"] = df["goalie_name"].isna().astype(int)
        df["is_goal"] = (df["event_type"] == "GOAL").astype(int)

        return df



def main():
    fe = FeatureEngineering()
    df = fe.combine_df()
    df = fe.calculate_distance_from_net(df)
    df = fe.calculate_empty_net(df)
    os.makedirs(fe._save_data_path, exist_ok=True)
    
    save_path = os.path.join(fe._save_data_path, "master.csv")
    df.to_csv(save_path, index=False)

    save_path = os.path.join(fe._save_data_path, "test_sample.csv")
    df.head(20).to_csv(save_path,index=False)

    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()
