import pandas as pd
import numpy as np
import os

def read_sas_write_hdf(read_paths, write_dir, hdf_store, downcast=True, verbose=True):
    """Read raw SAS files from source and write to binary feather format"""
    for data_name, path in read_paths.items():
        tmp = pd.read_sas(path)
        tmp.columns = [x.lower() for x in tmp.columns] # pythonize column names
        tmp['seqn'] = tmp['seqn'].astype(np.int64)
        if verbose:
            print('Writing data: {}'.format(data_name))
            tmp.columns = [x.lower() for x in tmp.columns]
        tmp.to_hdf(os.path.join(write_dir, hdf_store), data_name)

def plot_user_steps(pax_df, seqn_id=None, day_of_study=1, window_len=30):
    """
    Docstring here.
    """
    if seqn_id is None:
        seqn_id = np.random.choice(pax_df.seqn.unique(), 1)[0]
    plt_df = pax_df[(pax_df.seqn==seqn_id) & (pax_df.paxday==day_of_study)]
    try:
        assert len(plt_df) == 1440 # Make sure there are 1440 rows in the dataframe
    except AssertionError as e:
            e.args += (
                """
                Error: Unexpected number of rows returned: ({}) when using parameters seqn_id={} 
                and day_of_study={}, expected 1440
                """.format(len(plt_df), seqn_id, day_of_study), )
            raise
    plt_df.set_index('minute_of_day', inplace=True)
    plt_df.paxstep.rolling(
        window=window_len,
        win_type='triang'
    ).mean().plot(figsize=(25,8), title='seqn: {}, study day: {}'.format(seqn_id, day_of_study))