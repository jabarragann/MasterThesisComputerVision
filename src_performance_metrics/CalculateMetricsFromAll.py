from pathlib import Path
from scipy.signal import welch
import pandas as pd
import re
from src_performance_metrics.ExtractPerformanceMetrics import calculate_all_metrics


if __name__ == "__main__":

    path_list = [Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\06-KeyuCollection\2021-03-14_12h.24m.47s_keyu_P_manual_T_02'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\06-KeyuCollection\2021-03-14_12h.31m.16s_keyu_P_autonomy_T_02'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\08-AnirudhCollection\2021-03-15_18h.44m.15s_anirudhAutonomy01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\08-AnirudhCollection\2021-03-15_19h.03m.55s_anirudhmanual01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\07-JingCollection\2021-03-14_13h.02m.49s_jing-manual-01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\07-JingCollection\2021-03-14_13h.12m.14s_jing-autonomy-01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\05-ChihoCollections\2021-03-13_19h.42m.56s_Chiho_P_Manual_T_02'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\05-ChihoCollections\2021-03-13_19h.37m.20s_Chiho_P_Autonomy_T_02'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\04-GleboCollection\2021-03-12_20h.28m.48s_glebo_P_Manual_T_01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\04-GleboCollection\2021-03-12_20h.16m.57s_glebo_P_Autonomy_T_01'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\03-PauCollection\2021-03-12_13h.30m.25s_pau_P_Manual_T_02'),
                 Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\03-PauCollection\2021-03-12_13h.18m.14s_pau_P_Autonomy_T_02'),]

    metrics_list = ['completion_time', 'psm1_moving_time', 'psm2_moving_time', 'psm3_moving_time', 'clutching_time',
                   'psm1_velocity', 'psm2_velocity', 'psm3_velocity', 'clutching_events', 'tool_changing_events', 'percentage_blood',]
    conditions = ['autonomy', 'manual']
    columns = pd.MultiIndex.from_product([conditions, metrics_list], names=['conditions', 'metrics'])
    all_metrics_df = pd.DataFrame(columns=columns)
    all_metrics_df.index.name = 'user_id'

    for p in path_list:
        xdf_file = list(p.rglob("*.xdf"))
        assert len(xdf_file) == 1, "Check the eeg files are in the directory {:}".format(p.name)
        xdf_file = xdf_file[0]

        task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_raw\.xdf)', xdf_file.name)[0].lower()
        uid = re.findall('.+(?=_S[0-9]+_T[0-9]+_)', xdf_file.name)[0]

        assert task in ['manual','autonomy'], "Check the label of the eeg file"

        print(uid,task)

        results = calculate_all_metrics(p)
        results = results.reset_index(level=0, drop=True, inplace=False)
        idx = pd.IndexSlice
        all_metrics_df.loc[uid, idx[task, :]] = results.values.squeeze()

        x=0
    x = 0
    all_metrics_df = all_metrics_df.swaplevel(i=0, j=1, axis=1).sort_index(axis=1)
    f_path = Path(r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset')
    all_metrics_df.to_csv(f_path/'final_results.csv')