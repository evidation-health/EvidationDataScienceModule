#!/Users/bbradshaw/repos/evident/venv/bin/python

import pandas as pd
import numpy as np
import os
import itertools


data_dir = '/Users/bbradshaw/'
np.random.seed(42)

def event_window(event_date, window_size):
    """
    """
    event_dt = pd.to_datetime(event_date)
    begin_dt = event_dt - pd.Timedelta(days=window_size)
    end_dt = event_dt + pd.Timedelta(days=window_size)
    
    pre_window = pd.Series(pd.date_range(start=begin_dt, end=end_dt))
    post_window = pd.Series(pd.date_range(start=event_dt+pd.Timedelta(days=1), end=event_date))
    return pd.concat([pre_window, post_window])


def generate_shock_distribution(df, feature_mean, shock_mean, feature_std, shock_std, feature_name):
    """
    """
    f_i_mean = max(np.random.normal(feature_mean, feature_std, 1)[0], feature_mean*0.2)
    f_i_std = abs(np.random.normal(feature_std, max(0, 0.1*f_i_mean), 1)[0])
    df = df.copy()
    df['baseline_dist'] = np.random.normal(f_i_mean, f_i_std, 29)
    df['shock'] = np.random.normal(shock_mean, shock_std, 29)
    df['attenuation_weight'] = df.relative_idx.apply(lambda x: 1 / (1+abs(x**2)))+np.random.normal(0, 0.5, 29)
    df['attenuation_weight'] = df.attenuation_weight.rolling(5, win_type='triang', center=True, min_periods=1).mean()
    df[feature_name] = df.baseline_dist + (df.shock)*df.attenuation_weight
    df[feature_name] = df[feature_name].ffill().bfill().astype(int)

    return df[['date', feature_name]]


if __name__ == '__main__':
    # Generate the date of event, distibuted normally around an "anchor date"
    anchor_date = pd.to_datetime('2019-02-13')
    event_dates = [anchor_date + pd.Timedelta(days=np.random.normal(0, 30)) for i in range(3000)]

    # Create a table that contains the reported ILI peak symptom date
    user_events = pd.DataFrame({'user_id': list(range(3000)), 'event_date': event_dates})
    user_events['event_date'] = user_events.event_date.dt.date

    # Create a 29 day window indexed on the event date, for each user
    user_windows = pd.DataFrame(
        {'date': list(itertools.chain.from_iterable([event_window(d, 14) for d in user_events.event_date.values]))}
    )
    user_windows['user_id'] = list(itertools.chain.from_iterable([np.repeat(i, 29) for i in range(3000)]))
    user_windows = user_windows.merge(user_events, on='user_id')
    user_windows['event_date'] = pd.to_datetime(user_windows.event_date)
    user_windows['relative_idx'] = (user_windows.date - user_windows.event_date) / pd.Timedelta(days=1)

    # Generate user features
    walk_steps = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 10000, -2000, 2000, 1500, 'steps_sum'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    walk_steps.loc[walk_steps.steps_sum<0, 'steps_sum'] = 0

    sleep_disturbances = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 0, 4, 1, 2, 'sleep_disturbances'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    sleep_disturbances.loc[sleep_disturbances.sleep_disturbances<0, 'sleep_disturbances'] = 0

    resting_heart_rate = (
        user_windows
        .groupby('user_id')
        .apply(lambda df: generate_shock_distribution(df, 60, 5, 6, 2, 'resting_heart_rate'))
        .reset_index()
        .drop('level_1', axis=1)
    )
    resting_heart_rate.loc[resting_heart_rate.resting_heart_rate<38, 'resting_heart_rate'] = 38
    resting_heart_rate.loc[resting_heart_rate.resting_heart_rate>110, 'resting_heart_rate'] = 110
    
    features = (
        walk_steps
        .merge(sleep_disturbances, on=['user_id', 'date'])
        .merge(resting_heart_rate, on=['user_id', 'date'])
    )
    
    features.to_csv(os.path.join(data_dir, 'features.csv'), index=False)
    user_events.to_csv(os.path.join(data_dir, 'user_events.csv'), index=False)
