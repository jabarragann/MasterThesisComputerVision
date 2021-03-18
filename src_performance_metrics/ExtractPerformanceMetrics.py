import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_average_blood(file):
    df = pd.read_csv(file, index_col=[0])
    return df['blood_percentage'].mean()

def calculate_clutching_events(file):
    df = pd.read_csv(file)
    return df.loc[df['state'] == 'pressed'].shape[0]

def calculate_clutching_time(file):
    df = pd.read_csv(file)

    idx = df.loc[df['state'] == 'pressed'].index.values
    idx_one_ahead = idx + 1

    r = df.loc[idx_one_ahead,'ts'].values - df.loc[idx,'ts'].values
    return r.sum()

def calculate_tool_changes(file):
    df = pd.read_csv(file)
    return df.loc[df['value']== 'PSM3'].shape[0]


def calculate_completion_time(file):
    df = pd.read_csv(file)
    return df['ts'].values[-1] - df['ts'].values[0]

def calculate_velocity_arm(file):
    df = pd.read_csv(file)
    #Change to cm
    df.loc[:, ['x', 'y', 'z']] = df.loc[:, ['x', 'y', 'z']] * 100
    #calculate velocity
    diff = (df.shift(-1) - df)
    vel_df = diff.copy()
    vel_df['x'] = diff['x'] / diff['ts']
    vel_df['y'] = diff['y'] / diff['ts']
    vel_df['z'] = diff['z'] / diff['ts']
    vel_df = vel_df[['x','y','z']]
    vel_df = np.sqrt(np.square(vel_df).sum(axis=1))
    mean_vel = vel_df.mean()

    return mean_vel


def calculate_all_metrics(filesPath):
    resultsPath = filesPath / "session_performance_metrics.csv"

    indexTuples = [('time', 'completion_time'),  # done
                   ('time', 'psm1_moving_time'),
                   ('time', 'psm2_moving_time'),
                   ('time', 'psm3_moving_time'),
                   ('time', 'clutching_time'),  # done
                   ('motion', 'psm1_velocity'),  # done
                   ('motion', 'psm2_velocity'),  # done
                   ('motion', 'psm3_velocity'),  # done
                   ('events', 'clutching_events'),  # done
                   ('events', 'tool_changing_events'),  # done
                   ('blood', 'percentage_blood')]  # done

    index = pd.MultiIndex.from_tuples(indexTuples)
    index.set_names(['type', 'name'], inplace=True)
    results = pd.DataFrame(index=index, columns=["value"])

    for f in filesPath.rglob("*.txt"):
        # print(f.name)
        if "mtml_cartesian" in f.name:
            results.loc[('time', 'completion_time')] = calculate_completion_time(f)
        if "blood_percentage" in f.name:
            results.loc[('blood', 'percentage_blood')] = calculate_average_blood(f)
        if "clutch_events" in f.name:
            results.loc[('events', 'clutching_events')] = calculate_clutching_events(f)
            try:
                results.loc[('time', 'clutching_time')] = calculate_clutching_time(f)
            except Exception:
                print("no clutching time information")
        if "teleop_events" in f.name:
            results.loc[('events', 'tool_changing_events')] = calculate_tool_changes(f)
        if "psm" in f.name:
            if "psm1" in f.name:
                results.loc[('motion', 'psm1_velocity')] = calculate_velocity_arm(f)
            elif "psm2" in f.name:
                results.loc[('motion', 'psm2_velocity')] = calculate_velocity_arm(f)
            elif "psm3" in f.name:
                results.loc[('motion', 'psm3_velocity')] = calculate_velocity_arm(f)

    results.to_csv(resultsPath)
    return results

def main():
    filesPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\08-AnirudhCollection\2021-03-15_19h.03m.55s_anirudhmanual01")
    results = calculate_all_metrics(filesPath)
    x=0

if __name__ == "__main__":
   main()