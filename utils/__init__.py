import pandas as pd
import numpy as np
import os

def read_sas_write_hdf(read_paths, write_dir, hdf_store):
    """
    Read raw SAS files from source and write to binary feather format.
    
    Inputs:
        read_paths (dict): Dictionary of name: url key-value pairs to download
        write_dir (str): Path to hdf store
        hdf_store (str): Name of hdf store to write to
        
    Returns:
        None. Write downloaded data to hdf store.
    """
    for data_name, path in read_paths.items():
        tmp = pd.read_sas(path)
        tmp.columns = [x.lower() for x in tmp.columns] # pythonize column names
        tmp['seqn'] = tmp['seqn'].astype('object')
        tmp.columns = [x.lower() for x in tmp.columns]
        tmp.to_hdf(os.path.join(write_dir, hdf_store), data_name)

def plot_user_steps(pax_df, seqn_id=None, day_of_study=1, window_len=30):
    """
    Plot a single user-day's walk step intensity values.
    
    Inputs:
        pax_df (Pandas DataFrame): DataFrame of all user step data
        seqn_id (str): If not None, plot this specific user's step intensity
        day_of_study (int): Day of study to plot
        window_len (int): Choice of rolling window for walk step intensity
        
    Returns:
        None. Plots user walk steps inline in notebook.
    """
    if seqn_id is None:
        seqn_id = np.random.choice(pax_df.seqn.unique(), 1)[0]
    plt_df = pax_df[(pax_df.seqn==seqn_id) & (pax_df.paxday==day_of_study)]
    try:
        assert len(plt_df) == 1440 # Make sure there are 1440 rows in the dataframe
    except AssertionError as e:
            e.args += (
                """
                Error: Unexpected number of rows returned: ({})
                when using parameters seqn_id={} 
                and day_of_study={}, expected 1440
                """.format(len(plt_df), seqn_id, day_of_study), )
            raise
    plt_df.set_index('minute_of_day', inplace=True)
    plt_df.paxstep.rolling(
        window=window_len,
        win_type='triang'
    ).mean().plot(figsize=(25,8), title='seqn: {}, study day: {}'.format(seqn_id, day_of_study))